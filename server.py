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
import json
import os
import queue
import re
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

import mlx.core as mx
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mlx_vlm import load
from mlx_vlm.generate.diffusion import stream_diffusion_generate_from_kwargs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs, should_add_special_tokens

from context_guard import (
    HARD_CONTEXT_LIMIT,
    PREFILL_STEP_SIZE,
    SOFT_CONTEXT_LIMIT,
    ContextOverflow,
    clamp_limit,
    estimate_peak_gb,
    get_tokenizer,
    render_and_fit,
)

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "conversations.db"
WEB_DIR = BASE_DIR / "web"
MAX_QUEUE_DEPTH = 8

# ---------------------------------------------------------------------------
# thinking-channel splitting
# ---------------------------------------------------------------------------

THOUGHT_START = "<|channel>"
THOUGHT_END = "<channel|>"
_MAX_MARKER = max(len(THOUGHT_START), len(THOUGHT_END))


class ChannelSplitter:
    """Split a token stream into ('thinking'|'answer', text) pairs.

    The model wraps its reasoning in <|channel>thought ... <channel|>. Markers
    can straddle a token boundary, so we hold back a short tail that might be
    the prefix of a marker before emitting.
    """

    # The start marker is followed by a channel name and a newline
    # ("<|channel>thought\n"). The name can land in a later chunk than the
    # marker, so it is consumed by scanning for that newline rather than by
    # matching the literal word.
    _MAX_LABEL = 64

    def __init__(self):
        self.buf = ""
        self.kind = "answer"
        self.awaiting_label = False

    def feed(self, chunk):
        self.buf += chunk
        return self._drain(final=False)

    def flush(self):
        out = self._drain(final=True)
        if self.buf and not self.awaiting_label:
            out.append((self.kind, self.buf))
        self.buf = ""
        return out

    def _drain(self, final):
        out = []
        while True:
            if self.awaiting_label:
                nl = self.buf.find("\n")
                if nl == -1:
                    if not final and len(self.buf) < self._MAX_LABEL:
                        return out  # the newline is still in flight
                    self.buf = ""  # malformed header; drop it
                    self.awaiting_label = False
                    return out
                self.buf = self.buf[nl + 1 :]
                self.awaiting_label = False

            marker = THOUGHT_END if self.kind == "thinking" else THOUGHT_START
            idx = self.buf.find(marker)
            if idx == -1:
                break
            head = self.buf[:idx]
            if head:
                out.append((self.kind, head))
            self.buf = self.buf[idx + len(marker) :]
            if self.kind == "thinking":
                self.kind = "answer"
            else:
                self.kind = "thinking"
                self.awaiting_label = True

        if not self.awaiting_label:
            # hold back a tail that could be the prefix of a marker
            keep = _MAX_MARKER - 1
            if len(self.buf) > keep:
                emit, self.buf = self.buf[:-keep], self.buf[-keep:]
                if emit:
                    out.append((self.kind, emit))
        return out


# A denoising snapshot is a picture of the whole canvas, not a position in a
# stream, so ChannelSplitter cannot be used on it. Markers are only stripped so
# the preview reads as text; which side of a marker a word falls on does not
# matter for a frame that is about to be replaced.
_CHANNEL_MARKER_RE = re.compile(
    f"{re.escape(THOUGHT_START)}[^\n]*\n?|{re.escape(THOUGHT_END)}"
)


def strip_channel_markers(text):
    """Drop thinking-channel markers from a whole-canvas snapshot."""
    return _CHANNEL_MARKER_RE.sub("", text)


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
    def __init__(self, conversation_id, loop):
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self.loop = loop
        self.events = asyncio.Queue()
        self.cancelled = threading.Event()
        self.created_at = time.time()

    def emit(self, event, data):
        """Called from the worker thread; hands the event to the asyncio loop."""
        self.loop.call_soon_threadsafe(self.events.put_nowait, (event, data))


class Engine:
    """Owns the model and the single worker thread that may touch it."""

    def __init__(
        self,
        model_name,
        max_context,
        max_tokens,
        temperature,
        canvas_length=None,
        unmasking_interval=2,
    ):
        self.model_name = model_name
        self.max_context = max_context
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.canvas_length = canvas_length
        self.unmasking_interval = unmasking_interval
        self.jobs = queue.Queue()
        self.pending = []  # jobs waiting, for queue-position reporting
        self.pending_lock = threading.Lock()
        self.current = None
        self.loaded_at = None
        self.last_peak_gb = None
        self.total_jobs = 0

    def load(self):
        t0 = time.time()
        self.model, self.processor = load(self.model_name)
        self.config = load_config(self.model_name)
        self.tokenizer = get_tokenizer(self.processor)
        self.loaded_at = time.time()
        return self.loaded_at - t0

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
                self._generate(job)
            except Exception as e:  # keep the worker alive no matter what
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                with self.pending_lock:
                    self.current = None
                mx.clear_cache()
                self.total_jobs += 1

    def _diffusion_stream(self, prompt, on_result):
        """Drive the diffusion denoiser, pushing every result to `on_result`.

        Not stream_generate(): on the diffusion path it collects every result
        into a list and only yields once generation has finished, so the whole
        answer lands at once and cancellation cannot be seen mid-generation.
        Live delivery needs the on_result callback, and that cannot be passed
        through stream_generate either -- it forwards **kwargs into a call that
        already binds on_result, which raises TypeError. So prepare the inputs
        the way stream_generate would and call the diffusion entry point
        directly, as mlx-vlm's own server does.

        Returns the generator; the caller must exhaust and close it.
        """
        inputs = prepare_inputs(
            self.processor,
            prompts=prompt,
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor
            ),
        )

        stream_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prefill_step_size": PREFILL_STEP_SIZE,
        }
        if self.canvas_length:
            stream_kwargs["diffusion_max_canvas_length"] = self.canvas_length
        if self.unmasking_interval:
            stream_kwargs["diffusion_show_unmasking"] = True
            stream_kwargs["diffusion_unmasking_interval"] = self.unmasking_interval

        return stream_diffusion_generate_from_kwargs(
            self.model,
            self.processor,
            self.tokenizer,
            inputs["input_ids"],
            None,  # pixel_values; this server is text-only
            inputs.get("attention_mask"),
            [],  # skip_special_token_ids
            stream_kwargs,
            on_result=on_result,
        )

    def _generate(self, job):
        if job.cancelled.is_set():
            job.emit("error", {"message": "cancelled before start"})
            return

        rows = db_run(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
            (job.conversation_id,),
            fetch="all",
        )
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        if not messages:
            job.emit("error", {"message": "conversation has no messages"})
            return

        try:
            prompt, n_tokens, dropped = render_and_fit(
                self.processor,
                self.config,
                messages,
                self.tokenizer,
                self.max_context,
                apply_chat_template,
            )
        except ContextOverflow as e:
            job.emit("error", {"message": str(e)})
            return

        job.emit(
            "start",
            {
                "prompt_tokens": n_tokens,
                "dropped_turns": dropped,
                "estimated_peak_gb": round(estimate_peak_gb(n_tokens), 1),
            },
        )

        splitter = ChannelSplitter()
        answer, thinking = [], []
        last = None
        t0 = time.time()

        def on_result(result):
            nonlocal last
            if job.cancelled.is_set():
                return False  # stops the denoiser at the next step
            if result.is_draft:
                job.emit(
                    "draft",
                    {
                        "text": strip_channel_markers(result.draft_text),
                        "step": result.diffusion_step,
                        "total": result.diffusion_total_steps,
                        "canvas": result.diffusion_canvas_index,
                    },
                )
                return True
            last = result
            for kind, text in splitter.feed(result.text):
                (thinking if kind == "thinking" else answer).append(text)
                job.emit("token", {"kind": kind, "text": text})
            return True

        results = self._diffusion_stream(prompt, on_result)
        try:
            for _ in results:  # with on_result set, nothing is ever yielded
                pass
        finally:
            results.close()

        for kind, text in splitter.flush():
            (thinking if kind == "thinking" else answer).append(text)
            job.emit("token", {"kind": kind, "text": text})

        answer_text = "".join(answer).strip()
        thinking_text = "".join(thinking).strip()

        if job.cancelled.is_set() and not answer_text:
            job.emit("cancelled", {})
            return

        msg_id = db_run(
            "INSERT INTO messages (conversation_id, role, content, thinking, created_at)"
            " VALUES (?,?,?,?,?)",
            (job.conversation_id, "assistant", answer_text, thinking_text or None, time.time()),
        )
        db_run(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (time.time(), job.conversation_id),
        )

        peak = getattr(last, "peak_memory", None)
        if peak:
            self.last_peak_gb = round(peak, 2)
        job.emit(
            "done",
            {
                "message_id": msg_id,
                "cancelled": job.cancelled.is_set(),
                "stats": {
                    "prompt_tokens": n_tokens,
                    "generation_tokens": getattr(last, "generation_tokens", None),
                    "prompt_tps": round(getattr(last, "prompt_tps", 0) or 0, 1),
                    "generation_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
                    "peak_gb": self.last_peak_gb,
                    "wall_seconds": round(time.time() - t0, 1),
                },
            },
        )


engine: Engine = None
ACCESS_TOKEN = None


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


class ChatBody(BaseModel):
    conversation_id: str
    content: str


@app.post("/api/auth")
def auth(body: AuthBody):
    if not secrets.compare_digest(body.token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="bad token")
    r = JSONResponse({"ok": True})
    r.set_cookie(
        "session", ACCESS_TOKEN, httponly=True, samesite="lax", max_age=30 * 86400
    )
    return r


@app.get("/api/status")
def status(_=Depends(require_auth)):
    info = engine.queue_info()
    return {
        "model": engine.model_name,
        "max_context": engine.max_context,
        "hard_context_limit": HARD_CONTEXT_LIMIT,
        "max_queue_depth": MAX_QUEUE_DEPTH,
        "last_peak_gb": engine.last_peak_gb,
        "total_jobs": engine.total_jobs,
        **info,
    }


@app.get("/api/conversations")
def list_conversations(_=Depends(require_auth)):
    rows = db_run(
        "SELECT id, title, created_at, updated_at FROM conversations"
        " ORDER BY updated_at DESC",
        fetch="all",
    )
    return [dict(r) for r in rows]


@app.post("/api/conversations")
def create_conversation(body: NewConversation, _=Depends(require_auth)):
    cid = uuid.uuid4().hex
    now = time.time()
    db_run(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?,?,?,?)",
        (cid, body.title or "새 채팅", now, now),
    )
    return {"id": cid, "title": body.title or "새 채팅", "created_at": now, "updated_at": now}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str, _=Depends(require_auth)):
    db_run("DELETE FROM messages WHERE conversation_id=?", (cid,))
    db_run("DELETE FROM conversations WHERE id=?", (cid,))
    return {"ok": True}


@app.get("/api/conversations/{cid}/messages")
def get_messages(cid: str, _=Depends(require_auth)):
    rows = db_run(
        "SELECT id, role, content, thinking, created_at FROM messages"
        " WHERE conversation_id=? ORDER BY id",
        (cid,),
        fetch="all",
    )
    return [dict(r) for r in rows]


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

    db_run(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
        (body.conversation_id, "user", content, time.time()),
    )
    # first user message names the conversation
    if conv["title"] == "새 채팅":
        title = content[:40] + ("..." if len(content) > 40 else "")
        db_run("UPDATE conversations SET title=? WHERE id=?", (title, body.conversation_id))

    job = Job(body.conversation_id, asyncio.get_running_loop())
    if not engine.submit(job):
        raise HTTPException(
            status_code=503,
            detail=f"queue is full ({MAX_QUEUE_DEPTH} waiting); try again shortly",
        )
    _evict_stale_jobs()
    _jobs[job.id] = job
    return {"job_id": job.id}


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


def main():
    global engine, ACCESS_TOKEN

    parser = argparse.ArgumentParser(description="DiffusionGemma SSE chat server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
        "--token",
        default=os.environ.get("DIFFUSIONGEMMA_TOKEN"),
        help="access token; generated if omitted",
    )
    args = parser.parse_args()

    check_port_free(args.host, args.port)
    acquire_model_lock()

    max_context, warning = clamp_limit(args.max_context)
    if warning:
        print(f"warning: {warning}")

    ACCESS_TOKEN = args.token or secrets.token_urlsafe(16)

    init_db()

    engine = Engine(
        args.model,
        max_context,
        args.max_tokens,
        args.temperature,
        canvas_length=args.canvas_length,
        unmasking_interval=max(0, args.unmasking_interval),
    )
    print(f"Loading {args.model} ...", flush=True)
    took = engine.load()
    engine.start()

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    url = f"http://{lan_ip()}:{args.port}/?token={ACCESS_TOKEN}"
    print(
        f"Loaded in {took:.1f}s.\n"
        f"  context budget : {max_context:,} tokens "
        f"(~{estimate_peak_gb(max_context):.1f}GB peak)\n"
        f"  max queue depth: {MAX_QUEUE_DEPTH}\n"
        f"  concurrency    : 1 generation at a time (queued)\n\n"
        f"Open on any device on this network:\n  {url}\n",
        flush=True,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
