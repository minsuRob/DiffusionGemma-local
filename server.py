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

from mlx_vlm import load, stream_generate
from mlx_vlm.generate.diffusion import stream_diffusion_generate_from_kwargs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs, should_add_special_tokens

import docvision
import extract

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
UPLOAD_DIR = BASE_DIR / "uploads"
SCHEMA_DIR = BASE_DIR / "schemas"
MAX_QUEUE_DEPTH = 8

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


class _Cancelled(Exception):
    """Raised inside the extraction reader so a cancel unwinds the pipeline."""

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
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so new columns
    # have to be added explicitly or a database from before this feature keeps
    # the old shape and every insert fails.
    have = {r["name"] for r in _db.execute("PRAGMA table_info(messages)")}
    for column, decl in (("attachment", "TEXT"), ("extraction", "TEXT")):
        if column not in have:
            _db.execute(f"ALTER TABLE messages ADD COLUMN {column} {decl}")
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
    def __init__(self, conversation_id, loop, kind="chat", payload=None):
        self.id = uuid.uuid4().hex
        self.conversation_id = conversation_id
        self.loop = loop
        self.kind = kind  # "chat" | "extract"
        self.payload = payload or {}
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
                # Extraction runs on this same worker on purpose. It issues many
                # image calls, and letting it overlap with a chat generation
                # would put two model runs in flight and break the OOM
                # guarantee the queue exists to provide.
                if job.kind == "extract":
                    self._extract(job)
                else:
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

    def _image_reader(self, job):
        """A (pil_image, prompt, max_tokens=None) -> Reply callable.

        Uses stream_generate rather than the diffusion path: _diffusion_stream
        pins pixel_values to None and has never been exercised with an image,
        while this route is the one the accuracy measurements were taken on.
        Extraction wants correctness, not live denoising previews.
        """
        def read(pil_image, prompt, tokens=None, think=False):
            if job.cancelled.is_set():
                raise _Cancelled()
            formatted = apply_chat_template(
                self.processor, self.config,
                [{"role": "user", "content": prompt}], num_images=1,
                # Only the structure probe asks for reasoning; a tile read with
                # thinking on burns its whole budget inside the thought channel
                # and returns nothing. See extract_image's docstring.
                enable_thinking=think,
            )
            chunks = []
            last = None
            for r in stream_generate(
                self.model, self.processor, formatted,
                image=pil_image,
                max_tokens=tokens or EXTRACT_MAX_TOKENS,
                temperature=0.0,
                prefill_step_size=PREFILL_STEP_SIZE,
            ):
                chunks.append(r.text)
                last = r
            peak = getattr(last, "peak_memory", None)
            if peak:
                self.last_peak_gb = round(peak, 2)
            # Thought is separated, not merged into the answer: mixing it in
            # would feed reasoning prose to the row parser.
            return extract.split_thought("".join(chunks))

        return read

    def _read_image(self, job):
        """Free-form read of an image, for anything that is not a table.

        No tiling and no consensus: those exist to pin down field values, and
        neither majority-votes usefully over prose. This is explicitly the
        unverified path, and the UI labels it as such.
        """
        img = docvision.load_image(job.payload["image_path"])
        # Analysis, not transcription. The order of these three is the order we
        # want the model to think in: establish what the thing is before
        # deciding which parts of it matter.
        question = job.payload.get("question") or READ_ANALYSIS_PROMPT
        job.emit("start", {"mode": "read", "image_size": list(img.size)})

        splitter = ChannelSplitter()
        answer, thinking = [], []
        last = None
        t0 = time.time()
        formatted = apply_chat_template(
            self.processor, self.config,
            [{"role": "user", "content": question}], num_images=1,
            enable_thinking=True,
        )
        for r in stream_generate(
            self.model, self.processor, formatted, image=img,
            # Reasoning and answer share this budget, so it is larger than the
            # chat default; a cut-off here would stop mid-thought silently.
            max_tokens=max(self.max_tokens, READ_MAX_TOKENS),
            temperature=self.temperature,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            if job.cancelled.is_set():
                break
            last = r
            for kind, text in splitter.feed(r.text):
                (thinking if kind == "thinking" else answer).append(text)
                job.emit("token", {"kind": kind, "text": text})
        for kind, text in splitter.flush():
            (thinking if kind == "thinking" else answer).append(text)
            job.emit("token", {"kind": kind, "text": text})

        answer_text = "".join(answer).strip()
        if job.cancelled.is_set() and not answer_text:
            job.emit("cancelled", {})
            return

        msg_id = db_run(
            "INSERT INTO messages (conversation_id, role, content, thinking, created_at)"
            " VALUES (?,?,?,?,?)",
            (job.conversation_id, "assistant", answer_text,
             "".join(thinking).strip() or None, time.time()),
        )
        db_run("UPDATE conversations SET updated_at=? WHERE id=?",
               (time.time(), job.conversation_id))
        peak = getattr(last, "peak_memory", None)
        if peak:
            self.last_peak_gb = round(peak, 2)
        job.emit("done", {
            "message_id": msg_id,
            "cancelled": job.cancelled.is_set(),
            "stats": {
                "prompt_tokens": getattr(last, "prompt_tokens", 0) or 0,
                "generation_tokens": getattr(last, "generation_tokens", None),
                "prompt_tps": round(getattr(last, "prompt_tps", 0) or 0, 1),
                "generation_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
                "peak_gb": self.last_peak_gb,
                "wall_seconds": round(time.time() - t0, 1),
            },
        })

    def _extract(self, job):
        if job.payload.get("mode") == "read":
            self._read_image(job)
            return

        image_path = job.payload["image_path"]
        schema = job.payload["schema"]
        passes = job.payload.get("passes", extract.DEFAULT_PASSES)

        img = docvision.load_image(image_path)
        job.emit("start", {
            "mode": "extract",
            "schema": schema.get("name", "schema"),
            "image_size": list(img.size),
        })

        def on_progress(info):
            job.emit("progress", info)

        t0 = time.time()
        try:
            result = extract.extract_image(
                self._image_reader(job), img, schema,
                passes=passes, on_progress=on_progress,
            )
        except _Cancelled:
            job.emit("cancelled", {})
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
            "INSERT INTO messages (conversation_id, role, content, extraction, created_at)"
            " VALUES (?,?,?,?,?)",
            (job.conversation_id, "assistant", summary,
             json.dumps(result, ensure_ascii=False), time.time()),
        )
        db_run("UPDATE conversations SET updated_at=? WHERE id=?",
               (time.time(), job.conversation_id))

        job.emit("done", {
            "message_id": msg_id,
            "mode": "extract",
            "extraction": result,
            "stats": {
                "peak_gb": self.last_peak_gb,
                "wall_seconds": round(time.time() - t0, 1),
                **result["stats"],
            },
        })

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
    info = engine.queue_info()
    return {
        "ui_version": ui_version(),
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


@app.post("/api/extract")
async def start_extraction(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    schema_file: str = Form(...),
    passes: int = Form(extract.DEFAULT_PASSES),
    question: str = Form(""),
    _=Depends(require_auth),
):
    conv = db_run(
        "SELECT id, title FROM conversations WHERE id=?", (conversation_id,), fetch="one"
    )
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")

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
        payload={
            "image_path": str(stored),
            "schema": schema,
            "passes": passes,
            "mode": "read" if read_mode else "extract",
            "question": question,
        },
    )
    if not engine.submit(job):
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
