#!/usr/bin/env python3
"""SSE chat server for DiffusionGemma.

One model instance, one generation worker, everything else queues. That is the
whole design: the model needs 17GB resident and peaks near 28GB on long
prompts, so a second concurrent generation on a 36GB machine is an OOM, not a
slowdown. Concurrency is handled by making clients wait in a visible queue
rather than by running them in parallel.

Usage:
    .venv/bin/python server.py [--host 0.0.0.0] [--port 8000] [--max-context N]

Prints a LAN URL with an access token on startup.
"""

import argparse
import asyncio
import atexit
import hashlib
import io
import json
import os
import queue
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel

import docvision
import extract

from backends import Cancelled, RemoteError, load_registry
from context_guard import (
    HARD_CONTEXT_LIMIT,
    SOFT_CONTEXT_LIMIT,
    ContextOverflow,
    clamp_limit,
    estimate_peak_gb,
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "conversations.db"
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
SCHEMA_DIR = BASE_DIR / "schemas"
MODELS_PATH = BASE_DIR / "models.json"
MAX_QUEUE_DEPTH = 8

# Remote backends do no work on this machine, so the queue that protects the
# local model's memory would only add latency. They get their own lane with a
# cap that is about being polite to the upstream API, not about RAM.
MAX_REMOTE_CONCURRENCY = 4

# One tile of a table needs a few hundred tokens; this caps a runaway reply.
EXTRACT_MAX_TOKENS = 700
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Picked instead of a schema when the image is not a table and should just be
# read. Not a filename, so it can never collide with one in schemas/.
READ_MODE_SENTINEL = "__read__"

# Reasoning plus answer share this budget.
READ_MAX_TOKENS = 1400

READ_ANALYSIS_PROMPT = (
    "이 이미지를 분석해라. 글자를 전부 옮겨 적지 말고, 아래 순서로 답한다.\n\n"
    "1. 무엇인가 — 어떤 문서·화면인지, 어떤 구조인지, 무엇을 위한 것인지.\n"
    "2. 핵심 내용 — 중요한 항목과 수치만. 전체 전사는 하지 않는다.\n"
    "3. 특이사항 — 이상하거나 빠졌거나 눈에 띄는 점.\n\n"
    "읽을 수 없는 부분은 추측해서 채우지 말고 읽을 수 없다고 명시해라. "
    "확실한 것과 짐작한 것을 구분해서 써라."
)


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()
_db = None


def init_db():
    global _db
    _db = sqlite3.connect(DB_PATH, check_same_thread=False)
    _db.row_factory = sqlite3.Row
    _db.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            thinking        TEXT,
            created_at      REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
        """
    )
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so new columns
    # have to be added explicitly or a database from before this feature keeps
    # the old shape and every insert fails.
    have = {r["name"] for r in _db.execute("PRAGMA table_info(messages)")}
    for column, decl in (("attachment", "TEXT"), ("extraction", "TEXT"),
                         ("backend", "TEXT")):
        if column not in have:
            _db.execute(f"ALTER TABLE messages ADD COLUMN {column} {decl}")
    # NULL means "whatever the server's default is", which is what every
    # conversation from before backends were selectable should get.
    have = {r["name"] for r in _db.execute("PRAGMA table_info(conversations)")}
    if "backend" not in have:
        _db.execute("ALTER TABLE conversations ADD COLUMN backend TEXT")
    _db.commit()


def db_run(sql, params=(), fetch=None):
    with _db_lock:
        cur = _db.execute(sql, params)
        if fetch == "one":
            row = cur.fetchone()
            _db.commit()
            return row
        if fetch == "all":
            rows = cur.fetchall()
            _db.commit()
            return rows
        _db.commit()
        return cur.lastrowid


# ---------------------------------------------------------------------------
# generation worker
# ---------------------------------------------------------------------------


class Job:
    def __init__(self, conversation_id, loop, kind="chat", payload=None, backend_id=None):
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self.loop = loop
        self.kind = kind  # "chat" | "extract"
        self.payload = payload or {}
        self.backend_id = backend_id
        self.events = asyncio.Queue()
        self.cancelled = threading.Event()
        self.created_at = time.time()

    def emit(self, event, data):
        """Called from a worker thread; hands the event to the asyncio loop."""
        self.loop.call_soon_threadsafe(self.events.put_nowait, (event, data))


# ---------------------------------------------------------------------------
# job runners -- what happens around a backend, for every backend
# ---------------------------------------------------------------------------
#
# The database writes and the order of the SSE events are the same whichever
# model answered. Only the token source differs, so it is the only thing the
# backend supplies. Both lanes below call these.


def run_chat_job(job, backend):
    rows = db_run(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
        (job.conversation_id,),
        fetch="all",
    )
    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
    if not messages:
        job.emit("error", {"message": "conversation has no messages"})
        return

    if job.cancelled.is_set():
        job.emit("error", {"message": "cancelled before start"})
        return

    try:
        result = backend.stream_chat(messages, emit=job.emit, cancelled=job.cancelled)
    except ContextOverflow as e:
        job.emit("error", {"message": str(e)})
        return
    except RemoteError as e:
        job.emit("error", {"message": str(e)})
        return

    _finish_message(job, backend, result)


def run_extract_job(job, backend):
    if job.payload.get("mode") == "read":
        _run_read_job(job, backend)
        return

    image_path = job.payload["image_path"]
    schema = job.payload["schema"]
    passes = job.payload.get("passes", extract.DEFAULT_PASSES)

    img = docvision.load_image(image_path)
    job.emit("start", {
        "backend": backend.id,
        "mode": "extract",
        "schema": schema.get("name", "schema"),
        "image_size": list(img.size),
    })

    def on_progress(info):
        job.emit("progress", info)

    t0 = time.time()
    try:
        result = extract.extract_image(
            backend.image_reader(job.cancelled), img, schema,
            passes=passes, on_progress=on_progress,
        )
    except Cancelled:
        job.emit("cancelled", {})
        return
    except RemoteError as e:
        job.emit("error", {"message": str(e)})
        return

    flagged = [
        i for i, rec in enumerate(result["records"])
        if extract.worst_status(rec) != "ok"
    ]
    result["flagged_rows"] = flagged
    if result.get("aborted"):
        summary = f"추출 중단: {result['empty_reason']}"
    else:
        summary = (
            f"{result['stats']['rows']}행 추출 "
            f"(타일 {result['stats']['tiles']}개 × {passes}패스, "
            f"모델 호출 {result['stats']['model_calls']}회). "
            f"검토 필요 {len(flagged)}행."
        )

    msg_id = db_run(
        "INSERT INTO messages (conversation_id, role, content, extraction, backend, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (job.conversation_id, "assistant", summary,
         json.dumps(result, ensure_ascii=False), backend.id, time.time()),
    )
    db_run("UPDATE conversations SET updated_at=? WHERE id=?",
           (time.time(), job.conversation_id))

    job.emit("done", {
        "message_id": msg_id,
        "mode": "extract",
        "backend": backend.id,
        "extraction": result,
        "stats": {
            "peak_gb": getattr(backend, "last_peak_gb", None),
            "wall_seconds": round(time.time() - t0, 1),
            **result["stats"],
        },
    })


def _run_read_job(job, backend):
    """Free-form read of an image, for anything that is not a table.

    No tiling and no consensus: those exist to pin down field values, and
    neither majority-votes usefully over prose. This is explicitly the
    unverified path, and the UI labels it as such.
    """
    img = docvision.load_image(job.payload["image_path"])
    question = job.payload.get("question") or READ_ANALYSIS_PROMPT
    try:
        result = backend.stream_read_image(
            img, question, emit=job.emit, cancelled=job.cancelled
        )
    except RemoteError as e:
        job.emit("error", {"message": str(e)})
        return
    _finish_message(job, backend, result)


def _finish_message(job, backend, result):
    """Persist an answer and close out the stream. Shared by chat and read."""
    if job.cancelled.is_set() and not result["answer"]:
        job.emit("cancelled", {})
        return

    msg_id = db_run(
        "INSERT INTO messages (conversation_id, role, content, thinking, backend, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (job.conversation_id, "assistant", result["answer"],
         result["thinking"] or None, backend.id, time.time()),
    )
    db_run(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        (time.time(), job.conversation_id),
    )
    job.emit("done", {
        "message_id": msg_id,
        "backend": backend.id,
        "cancelled": job.cancelled.is_set(),
        "stats": result["stats"],
    })


# ---------------------------------------------------------------------------
# lane 1: the local model -- one at a time, always
# ---------------------------------------------------------------------------


class LocalEngine:
    """Owns the single worker thread that may touch the in-process model.

    The model needs 17GB resident and peaks near 28GB on long prompts, so a
    second concurrent generation on a 36GB machine is an OOM, not a slowdown.
    Everything that runs on the local backend queues behind this one thread.
    """

    def __init__(self, backend):
        self.backend = backend
        self.jobs = queue.Queue()
        self.pending = []  # jobs waiting, for queue-position reporting
        self.pending_lock = threading.Lock()
        self.current = None
        self.total_jobs = 0

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="generation")
        self.thread.start()

    def submit(self, job):
        with self.pending_lock:
            if len(self.pending) >= MAX_QUEUE_DEPTH:
                return False
            self.pending.append(job)
            position = len(self.pending) - 1 + (1 if self.current else 0)
        self.jobs.put(job)
        job.emit("queued", {"position": position})
        return True

    def _broadcast_positions(self):
        with self.pending_lock:
            snapshot = list(self.pending)
        for i, job in enumerate(snapshot):
            job.emit("queued", {"position": i + 1})

    def queue_info(self):
        with self.pending_lock:
            return {"waiting": len(self.pending), "busy": self.current is not None}

    def _run(self):
        while True:
            job = self.jobs.get()
            with self.pending_lock:
                if job in self.pending:
                    self.pending.remove(job)
                self.current = job
            self._broadcast_positions()
            try:
                # Extraction runs on this same worker on purpose. It issues many
                # image calls, and letting it overlap with a chat generation
                # would put two model runs in flight and break the OOM
                # guarantee the queue exists to provide.
                if job.kind == "extract":
                    run_extract_job(job, self.backend)
                else:
                    run_chat_job(job, self.backend)
            except Exception as e:  # keep the worker alive no matter what
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                with self.pending_lock:
                    self.current = None
                self.backend.clear_cache()
                self.total_jobs += 1


# ---------------------------------------------------------------------------
# lane 2: remote models -- concurrent, and never behind the local queue
# ---------------------------------------------------------------------------


class RemoteRunner:
    """Runs jobs whose backend talks to someone else's machine.

    Deliberately not the local queue: a Gemini request costs a socket here, and
    making it wait out a 30-second local generation would be latency invented
    for nothing. The backends stream synchronously, so each job gets a thread
    and keeps the same `job.emit` contract the local worker uses.
    """

    def __init__(self, limit=MAX_REMOTE_CONCURRENCY):
        self.limit = limit
        self.semaphore = None  # created on the loop, on first use
        self.active = 0
        self.total_jobs = 0
        # asyncio keeps only a weak reference to a running task, so a task
        # nobody holds can be collected mid-flight. Hold them until they end.
        self.tasks = set()

    def info(self):
        return {"remote_active": self.active, "remote_limit": self.limit}

    def submit(self, job, backend):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit)
        task = asyncio.create_task(self._run(job, backend))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return True

    async def _run(self, job, backend):
        async with self.semaphore:
            self.active += 1
            try:
                runner = run_extract_job if job.kind == "extract" else run_chat_job
                await asyncio.to_thread(runner, job, backend)
            except Exception as e:
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                self.active -= 1
                self.total_jobs += 1


def submit_job(job, backend):
    """Route a job to the lane its backend belongs to."""
    if backend.kind == "mlx-local":
        if local_engine is None:
            job.emit("error", {"message": "로컬 모델이 로드되지 않았습니다"})
            return False
        return local_engine.submit(job)
    return remote_runner.submit(job, backend)


registry = None
local_engine: LocalEngine = None
remote_runner: RemoteRunner = None
ACCESS_TOKEN = None
AUTH_MODE = "open"


# ---------------------------------------------------------------------------
# safety: refuse to start if something else already holds the model
# ---------------------------------------------------------------------------


LOCK_PATH = BASE_DIR / ".model.lock"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_model_lock():
    """Refuse to start if anything else already holds the model.

    Two model instances need ~34GB and will OOM this 36GB machine, so this is
    a hard stop rather than a warning. Covers our own server via a PID lock and
    mlx_vlm.server via a process scan, since that one does not take our lock.
    """
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
        except ValueError:
            pid = None
        if pid and _pid_alive(pid):
            sys.exit(
                f"refusing to start: server.py is already running as pid {pid}.\n"
                "Two model instances need ~34GB and will OOM this 36GB machine.\n"
                f"Stop it first, or delete {LOCK_PATH.name} if that pid is stale."
            )
        LOCK_PATH.unlink(missing_ok=True)  # stale lock from a crashed run

    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        out = ""
    me = os.getpid()
    for line in out.splitlines():
        pid_str, _, cmd = line.strip().partition(" ")
        if not pid_str.isdigit() or int(pid_str) == me:
            continue
        if "mlx_vlm.server" in cmd:
            sys.exit(
                f"refusing to start: mlx_vlm.server is running as pid {pid_str}.\n"
                f"  {cmd[:110]}\n"
                "It holds its own copy of the model; running both will OOM.\n"
                "Stop it first."
            )

    LOCK_PATH.write_text(str(me))
    atexit.register(release_model_lock)


def release_model_lock():
    try:
        if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
            LOCK_PATH.unlink()
    except OSError:
        pass


def check_port_free(host, port):
    """Fail before the 3-second model load rather than after it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("" if host == "0.0.0.0" else host, port))
    except OSError:
        holder = ""
        try:
            out = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()
            if len(out) > 1:
                holder = f"\n  held by: {out[1]}"
        except Exception:
            pass
        sys.exit(f"port {port} is already in use.{holder}\nPick another with --port.")
    finally:
        s.close()


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


def require_auth(session: str = Cookie(default=None)):
    """The one gate every endpoint depends on.

    Two modes, chosen with --auth. In "token" mode this is the original
    behaviour: a random token, exchanged once for an httpOnly cookie. In
    "open" mode there is no gate at all -- anyone who reaches the port is in.

    Open is the default because the server is meant to be reached through its
    Tailscale Funnel URL and knowing that URL is currently the whole of the
    access control. Be clear-eyed about what that is worth: a *.ts.net name is
    published in Certificate Transparency logs the moment Tailscale issues its
    certificate, so it is discoverable, not secret. Run with --auth token for
    anything left up unattended.
    """
    if AUTH_MODE == "open":
        return True
    if session != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    return True


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Without this, browsers heuristically cache the UI assets and keep
    serving a stale app.js/style.css after an edit. `no-cache` still allows a
    304 via the ETag StaticFiles already sends, so it costs one small request.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response


class AuthBody(BaseModel):
    token: str


class NewConversation(BaseModel):
    title: str | None = None
    backend: str | None = None


class ConversationPatch(BaseModel):
    backend: str


class ChatBody(BaseModel):
    conversation_id: str
    content: str
    # Lets the client switch model mid-thread without a round trip to PATCH
    # first; the conversation's own setting is the fallback.
    backend: str | None = None


@app.post("/api/auth")
def auth(body: AuthBody):
    if AUTH_MODE == "open":
        return JSONResponse({"ok": True, "auth": "open"})
    if not secrets.compare_digest(body.token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="bad token")
    r = JSONResponse({"ok": True})
    r.set_cookie(
        "session", ACCESS_TOKEN, httponly=True, samesite="lax", max_age=30 * 86400
    )
    return r


def ui_version():
    """Stamp identifying the UI assets currently on disk.

    A tab opened before an edit keeps running the old code with no sign that
    it is stale -- which is how a fixed bug went on showing "all rows passed"
    for a zero-row result. The client compares this against the version it
    booted with.
    """
    stamps = []
    for asset in ("index.html", "app.js", "style.css"):
        path = WEB_DIR / asset
        if path.exists():
            stamps.append(int(path.stat().st_mtime))
    return max(stamps) if stamps else 0


@app.get("/api/status")
def status(_=Depends(require_auth)):
    local = registry.local
    info = local_engine.queue_info() if local_engine else {"waiting": 0, "busy": False}
    total = (local_engine.total_jobs if local_engine else 0) + remote_runner.total_jobs
    return {
        "ui_version": ui_version(),
        "auth": AUTH_MODE,
        "default_backend": registry.default_id,
        # Kept flat and named as before so an old tab still renders; they now
        # describe the local backend specifically, which may not be in use.
        "model": local.model_name if local else None,
        "max_context": local.max_context if local else None,
        "hard_context_limit": HARD_CONTEXT_LIMIT,
        "max_queue_depth": MAX_QUEUE_DEPTH,
        "last_peak_gb": local.last_peak_gb if local else None,
        "local_loaded": local is not None,
        "total_jobs": total,
        **info,
        **remote_runner.info(),
    }


@app.get("/api/backends")
def list_backends(_=Depends(require_auth)):
    """Everything the client needs to render the picker, including the ones it
    must grey out. A backend with no API key stays visible and says why --
    finding out at send time would be worse."""
    return registry.describe()


@app.get("/api/conversations")
def list_conversations(_=Depends(require_auth)):
    rows = db_run(
        "SELECT id, title, backend, created_at, updated_at FROM conversations"
        " ORDER BY updated_at DESC",
        fetch="all",
    )
    return [dict(r) for r in rows]


@app.post("/api/conversations")
def create_conversation(body: NewConversation, _=Depends(require_auth)):
    cid = uuid.uuid4().hex
    now = time.time()
    backend = _valid_backend_id(body.backend)
    db_run(
        "INSERT INTO conversations (id, title, backend, created_at, updated_at)"
        " VALUES (?,?,?,?,?)",
        (cid, body.title or "새 채팅", backend, now, now),
    )
    return {"id": cid, "title": body.title or "새 채팅", "backend": backend,
            "created_at": now, "updated_at": now}


@app.patch("/api/conversations/{cid}")
def update_conversation(cid: str, body: ConversationPatch, _=Depends(require_auth)):
    if registry.get(body.backend) is None:
        raise HTTPException(status_code=404, detail="unknown backend")
    row = db_run("SELECT id FROM conversations WHERE id=?", (cid,), fetch="one")
    if not row:
        raise HTTPException(status_code=404, detail="conversation not found")
    db_run("UPDATE conversations SET backend=? WHERE id=?", (body.backend, cid))
    return {"ok": True, "backend": body.backend}


def _valid_backend_id(backend_id):
    """None (meaning: use the server default) or a configured id. An unknown
    id is dropped rather than rejected -- a stale tab should not 400."""
    if backend_id and registry.get(backend_id):
        return backend_id
    return None


def backend_for(conversation_id, override=None):
    """The backend a job should run on, and whether it can actually run."""
    if override and registry.get(override):
        chosen = registry.get(override)
    else:
        row = db_run(
            "SELECT backend FROM conversations WHERE id=?", (conversation_id,),
            fetch="one",
        )
        chosen = registry.resolve(row["backend"] if row else None)
    if chosen is None:
        raise HTTPException(status_code=503, detail="사용 가능한 백엔드가 없습니다")
    ok, reason = chosen.available()
    if not ok:
        raise HTTPException(status_code=503, detail=f"{chosen.label}: {reason}")
    return chosen


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str, _=Depends(require_auth)):
    db_run("DELETE FROM messages WHERE conversation_id=?", (cid,))
    db_run("DELETE FROM conversations WHERE id=?", (cid,))
    return {"ok": True}


@app.get("/api/conversations/{cid}/messages")
def get_messages(cid: str, _=Depends(require_auth)):
    rows = db_run(
        "SELECT id, role, content, thinking, attachment, extraction, created_at"
        " FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,),
        fetch="all",
    )
    out = []
    for r in rows:
        d = dict(r)
        if d.get("extraction"):
            d["extraction"] = json.loads(d["extraction"])
        out.append(d)
    return out


@app.get("/api/schemas")
def list_schemas(_=Depends(require_auth)):
    """Extraction schemas available on disk."""
    if not SCHEMA_DIR.exists():
        return []
    out = []
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        try:
            schema = extract.load_schema(path)
        except (ValueError, json.JSONDecodeError):
            continue  # a malformed schema should not break the picker
        out.append({
            "file": path.name,
            "name": schema.get("name", path.stem),
            "description": schema.get("description", ""),
            "fields": extract.field_names(schema),
        })
    return out


@app.get("/api/attachments/{name}")
def get_attachment(name: str, _=Depends(require_auth)):
    # The name comes from a URL, so anchor it to the upload directory rather
    # than trusting it to stay inside.
    path = (UPLOAD_DIR / name).resolve()
    if not str(path).startswith(str(UPLOAD_DIR.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


_jobs: dict[str, Job] = {}
JOB_TTL_SECONDS = 900


def _evict_stale_jobs():
    """A client that posts but never opens the stream would leak its Job."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for jid in [j for j, job in _jobs.items() if job.created_at < cutoff]:
        _jobs.pop(jid, None)


@app.post("/api/chat")
async def chat(body: ChatBody, request: Request, _=Depends(require_auth)):
    conv = db_run(
        "SELECT id, title FROM conversations WHERE id=?", (body.conversation_id,), fetch="one"
    )
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")

    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="empty message")

    backend = backend_for(body.conversation_id, body.backend)
    if body.backend and registry.get(body.backend):
        db_run("UPDATE conversations SET backend=? WHERE id=?",
               (body.backend, body.conversation_id))

    db_run(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
        (body.conversation_id, "user", content, time.time()),
    )
    # first user message names the conversation
    if conv["title"] == "새 채팅":
        title = content[:40] + ("..." if len(content) > 40 else "")
        db_run("UPDATE conversations SET title=? WHERE id=?", (title, body.conversation_id))

    job = Job(body.conversation_id, asyncio.get_running_loop(), backend_id=backend.id)
    if not submit_job(job, backend):
        raise HTTPException(
            status_code=503,
            detail=f"queue is full ({MAX_QUEUE_DEPTH} waiting); try again shortly",
        )
    _evict_stale_jobs()
    _jobs[job.id] = job
    return {"job_id": job.id}


@app.post("/api/extract")
async def start_extraction(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    schema_file: str = Form(...),
    passes: int = Form(extract.DEFAULT_PASSES),
    question: str = Form(""),
    backend_id: str = Form(""),
    _=Depends(require_auth),
):
    conv = db_run(
        "SELECT id, title FROM conversations WHERE id=?", (conversation_id,), fetch="one"
    )
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")

    backend = backend_for(conversation_id, backend_id or None)
    if not backend.capabilities.get("vision"):
        raise HTTPException(
            status_code=400,
            detail=f"{backend.label}는 이미지를 읽을 수 없습니다",
        )

    read_mode = schema_file == READ_MODE_SENTINEL
    schema = None
    if not read_mode:
        schema_path = (SCHEMA_DIR / schema_file).resolve()
        if not str(schema_path).startswith(str(SCHEMA_DIR.resolve())) or not schema_path.is_file():
            raise HTTPException(status_code=404, detail="unknown schema")
        try:
            schema = extract.load_schema(schema_path)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=400, detail=f"bad schema: {e}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지가 너무 큽니다 ({len(raw)//1024//1024}MB, 최대 25MB)",
        )

    UPLOAD_DIR.mkdir(exist_ok=True)
    name = hashlib.sha256(raw).hexdigest()[:32] + ".png"
    stored = UPLOAD_DIR / name
    if not stored.exists():
        try:
            # Re-encode rather than trusting the extension: this both validates
            # that the bytes really are an image and normalizes the format.
            with PILImage.open(io.BytesIO(raw)) as im:
                im.convert("RGB").save(stored, "PNG")
        except Exception:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다")

    passes = max(1, min(5, passes))
    question = (question or "").strip()
    if read_mode:
        label = f"[이미지 읽기] {question}" if question else "[이미지 읽기]"
    else:
        label = f"[이미지 추출] {schema.get('name', schema_file)}"
    db_run(
        "INSERT INTO messages (conversation_id, role, content, attachment, created_at)"
        " VALUES (?,?,?,?,?)",
        (conversation_id, "user", label, name, time.time()),
    )
    if conv["title"] == "새 채팅":
        db_run("UPDATE conversations SET title=? WHERE id=?", (label[:40], conversation_id))

    job = Job(
        conversation_id,
        asyncio.get_running_loop(),
        kind="extract",
        backend_id=backend.id,
        payload={
            "image_path": str(stored),
            "schema": schema,
            "passes": passes,
            "mode": "read" if read_mode else "extract",
            "question": question,
        },
    )
    if not submit_job(job, backend):
        raise HTTPException(
            status_code=503,
            detail=f"queue is full ({MAX_QUEUE_DEPTH} waiting); try again shortly",
        )
    _evict_stale_jobs()
    _jobs[job.id] = job
    return {"job_id": job.id, "attachment": name}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, _=Depends(require_auth)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    job.cancelled.set()
    return {"ok": True}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str, _=Depends(require_auth)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")

    async def gen():
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(job.events.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event in ("done", "error", "cancelled"):
                    break
        finally:
            _jobs.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    """Serve index.html with a mtime stamp on the asset URLs.

    A no-cache header alone does not help a browser that already holds a
    cached app.js from before the header existed, and every phone that has
    opened the page keeps its old copy. Changing the URL is what actually
    forces the fetch after an edit.
    """
    html = (WEB_DIR / "index.html").read_text()
    for asset in ("style.css", "app.js"):
        stamp = int((WEB_DIR / asset).stat().st_mtime)
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# tailscale exposure
# ---------------------------------------------------------------------------
#
# Funnel is the default way in. The alternative, `serve`, publishes the same
# HTTPS URL but only inside the tailnet; it is one flag away and is the right
# choice for anything left running unattended, especially with --auth open.
#
# Port 8443 rather than 443 on purpose: Funnel only accepts 443, 8443 and
# 10000, and this machine may already be funnelling something else on 443.
# Taking a different port leaves that mapping alone.

FUNNEL_PORTS = (443, 8443, 10000)


def tailscale_hostname():
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return (json.loads(out.stdout).get("Self") or {}).get("DNSName", "").rstrip(".")
    except Exception:
        return None


def tailscale_expose(mode, expose_port, local_port):
    """Point mode ("funnel"/"serve") at the local server. Returns the URL.

    Never fatal: a machine with no tailscale, or a tailnet without Funnel
    enabled, should still get a working LAN server and a clear warning.
    """
    if expose_port not in FUNNEL_PORTS and mode == "funnel":
        print(f"warning: funnel only allows ports {FUNNEL_PORTS}; "
              f"{expose_port} will be rejected")
    cmd = [
        "tailscale", mode, "--bg", f"--https={expose_port}",
        f"http://127.0.0.1:{local_port}",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("warning: tailscale not found; skipping external exposure")
        return None
    except subprocess.TimeoutExpired:
        print("warning: tailscale timed out; skipping external exposure")
        return None
    if out.returncode != 0:
        detail = (out.stderr or out.stdout or "").strip().splitlines()
        print(f"warning: tailscale {mode} failed: {detail[0] if detail else '?'}")
        return None

    host = tailscale_hostname()
    if not host:
        return None
    suffix = "" if expose_port == 443 else f":{expose_port}"
    return f"https://{host}{suffix}/"


def tailscale_withdraw(mode, expose_port):
    try:
        subprocess.run(
            ["tailscale", mode, f"--https={expose_port}", "off"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass


def main():
    global registry, local_engine, remote_runner, ACCESS_TOKEN, AUTH_MODE

    parser = argparse.ArgumentParser(description="DiffusionGemma SSE chat server")
    parser.add_argument("--model", default=None,
                        help="override the local backend's model id")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8842)
    parser.add_argument("--max-context", type=int, default=SOFT_CONTEXT_LIMIT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--canvas-length",
        type=int,
        default=None,
        help="tokens denoised per canvas; smaller commits text more often but "
        "costs throughput (default: the model's own canvas_length)",
    )
    parser.add_argument(
        "--unmasking-interval",
        type=int,
        default=2,
        help="send a denoising preview every Nth step; 0 disables the preview "
        "(each frame forces a GPU sync, so lower is smoother but slower)",
    )
    parser.add_argument(
        "--models", default=str(MODELS_PATH),
        help="backend configuration (default: models.json next to this file)",
    )
    parser.add_argument(
        "--no-local", action="store_true",
        help="serve remote backends only: no MLX import, no 17GB load, no "
        "model lock. Use when this machine should not hold the model",
    )
    parser.add_argument(
        "--auth", choices=("open", "token"), default=None,
        help="open (default): no gate, the URL is the whole secret. "
        "token: the original random-token cookie gate",
    )
    parser.add_argument(
        "--expose", choices=("funnel", "serve", "off"), default="funnel",
        help="funnel (default): reachable from the public internet. "
        "serve: HTTPS inside the tailnet only. off: LAN only",
    )
    parser.add_argument(
        "--expose-port", type=int, default=8443,
        help="tailscale port to publish on (443/8443/10000 for funnel)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DIFFUSIONGEMMA_TOKEN"),
        help="access token; generated if omitted. Implies --auth token",
    )
    args = parser.parse_args()

    AUTH_MODE = args.auth or ("token" if args.token else "open")

    check_port_free(args.host, args.port)

    max_context, warning = clamp_limit(args.max_context)
    if warning:
        print(f"warning: {warning}")

    runtime = {
        "max_context": max_context,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "canvas_length": args.canvas_length,
        "unmasking_interval": max(0, args.unmasking_interval),
        "extract_max_tokens": EXTRACT_MAX_TOKENS,
        "read_max_tokens": READ_MAX_TOKENS,
    }
    registry = load_registry(
        Path(args.models), runtime,
        include_local=not args.no_local,
        model_override=args.model,
    )

    ACCESS_TOKEN = args.token or secrets.token_urlsafe(16)

    init_db()

    remote_runner = RemoteRunner()

    local = registry.local
    took = None
    if local is not None:
        # The lock only matters when this process actually holds the weights.
        acquire_model_lock()
        print(f"Loading {local.model_name} ...", flush=True)
        took = local.load()
        local_engine = LocalEngine(local)
        local_engine.start()

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    lines = []
    if took is not None:
        lines.append(f"Loaded in {took:.1f}s.")
        lines.append(
            f"  context budget : {max_context:,} tokens "
            f"(~{estimate_peak_gb(max_context):.1f}GB peak)"
        )
        lines.append(f"  max queue depth: {MAX_QUEUE_DEPTH}")
        lines.append("  concurrency    : 1 local generation at a time (queued)")
    else:
        lines.append("Local model not loaded (--no-local).")
    lines.append(
        f"  remote lane    : up to {MAX_REMOTE_CONCURRENCY} concurrent, not queued"
    )
    for b in registry.describe():
        ok = "" if b["available"] else f"  — 사용 불가: {b['reason']}"
        star = "*" if b["default"] else " "
        lines.append(f"  {star} {b['id']:<22} {b['label']}{ok}")
    lines.append(f"  auth           : {AUTH_MODE}")

    token_query = "" if AUTH_MODE == "open" else f"?token={ACCESS_TOKEN}"

    public_url = None
    if args.expose != "off":
        public_url = tailscale_expose(args.expose, args.expose_port, args.port)
        if public_url:
            atexit.register(tailscale_withdraw, args.expose, args.expose_port)

    if public_url:
        reach = "public internet" if args.expose == "funnel" else "your tailnet"
        lines.append(f"\nReachable from the {reach}:\n  {public_url}{token_query}")
        if args.expose == "funnel" and AUTH_MODE == "open":
            lines.append(
                "  warning: funnel + --auth open means anyone who reaches this URL\n"
                "           can read every conversation. *.ts.net names appear in\n"
                "           public certificate transparency logs, so treat the URL\n"
                "           as discoverable, not secret. Use --auth token to close it."
            )
    lines.append(f"\nOn this network:\n  http://{lan_ip()}:{args.port}/{token_query}\n")

    print("\n".join(lines), flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
