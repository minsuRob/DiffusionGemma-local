#!/usr/bin/env python3
"""LLM backends for the chat server.

The server used to be one model: an MLX DiffusionGemma loaded in-process. This
module is the seam that made a second one possible. A backend owns exactly one
thing -- how to turn messages (and optionally an image) into a token stream --
and knows nothing about jobs, SSE, or the database. The server owns those and
drives whichever backend a conversation picked.

Two backends ship here:

    LocalDiffusionBackend   the original in-process MLX model. Default.
    GeminiBackend           Google's Generative Language API over HTTPS.

Adding a third (an OpenAI-compatible /v1 endpoint, say) means a class with the
same four methods and one entry in models.json; nothing in server.py changes.

The capability flags matter as much as the methods. A backend that cannot see
images must not be offered the attach button, and one that cannot show a
denoising canvas must not leave the UI waiting for a `draft` event that will
never arrive. Capabilities are how the client finds that out before it asks.
"""

import base64
import io
import json
import os
import re
import time

# ---------------------------------------------------------------------------
# thinking-channel splitting
# ---------------------------------------------------------------------------
#
# Lives here rather than in server.py because it is a property of a model's
# output format, not of the transport. The local model interleaves reasoning
# into one text stream and marks it; Gemini hands thoughts over as separate
# parts and needs none of this.

THOUGHT_START = "<|channel>"
THOUGHT_END = "<channel|>"
_MAX_MARKER = max(len(THOUGHT_START), len(THOUGHT_END))


class Cancelled(Exception):
    """Raised inside a reader callable so a cancel unwinds the pipeline."""


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
# the interface
# ---------------------------------------------------------------------------


class Backend:
    """One way of getting tokens out of a model.

    Every generating method takes `emit(event, data)` and `cancelled` (a
    threading.Event) and returns

        {"answer": str, "thinking": str, "stats": {...}}

    The caller writes the answer to the database and sends the `done` event;
    a backend only emits `start`, `token`, and -- if it can -- `draft`.
    All of these run on a worker thread, never on the event loop.
    """

    kind = "abstract"
    # Advertised to the client so the UI can disable what a backend cannot do.
    #   vision   -- accepts images (attachment, extraction, image reading)
    #   drafts   -- emits `draft` events (the denoising canvas preview)
    #   thinking -- can produce a separated reasoning channel
    capabilities = {"vision": False, "drafts": False, "thinking": False}

    def __init__(self, spec, runtime):
        self.id = spec["id"]
        self.label = spec.get("label") or spec["id"]
        self.model_name = spec.get("model") or spec["id"]
        self.context_limit = spec.get("context_limit")
        self.spec = spec
        self.runtime = runtime
        self.total_jobs = 0

    # -- availability ------------------------------------------------------

    def available(self):
        """(usable, reason). A backend with a missing key stays in the list
        and explains itself, rather than failing at send time."""
        return True, None

    def describe(self):
        ok, reason = self.available()
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "model": self.model_name,
            "available": ok,
            "reason": reason,
            "capabilities": dict(self.capabilities),
            "context_limit": self.context_limit,
            "remote": self.kind != "mlx-local",
        }

    # -- generation --------------------------------------------------------

    def stream_chat(self, messages, *, emit, cancelled):
        raise NotImplementedError

    def image_reader(self, cancelled):
        """A (pil_image, prompt, tokens=None, think=False) -> Reply callable,
        the contract extract.extract_image consumes. None if no vision."""
        return None

    def stream_read_image(self, image, question, *, emit, cancelled):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# local: MLX DiffusionGemma, in-process
# ---------------------------------------------------------------------------


class LocalDiffusionBackend(Backend):
    """The original engine's model half.

    Everything here is serialized by the server's single worker thread: the
    model needs 17GB resident and peaks near 28GB, so a second concurrent
    generation is an OOM rather than a slowdown. That constraint belongs to
    this backend alone -- remote ones are not queued behind it.
    """

    kind = "mlx-local"
    capabilities = {"vision": True, "drafts": True, "thinking": True}

    def __init__(self, spec, runtime):
        super().__init__(spec, runtime)
        self.max_context = runtime["max_context"]
        self.max_tokens = runtime["max_tokens"]
        self.temperature = runtime["temperature"]
        self.canvas_length = runtime.get("canvas_length")
        self.unmasking_interval = runtime.get("unmasking_interval", 2)
        self.last_peak_gb = None
        self.loaded_at = None
        if not self.context_limit:
            self.context_limit = self.max_context

    def load(self):
        # Imported here, not at module scope, so a remote-only server never
        # pays for (or requires) the MLX stack.
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        from context_guard import get_tokenizer

        t0 = time.time()
        self.model, self.processor = load(self.model_name)
        self.config = load_config(self.model_name)
        self.tokenizer = get_tokenizer(self.processor)
        self.loaded_at = time.time()
        return self.loaded_at - t0

    def clear_cache(self):
        import mlx.core as mx

        mx.clear_cache()

    def _note_peak(self, last):
        peak = getattr(last, "peak_memory", None)
        if peak:
            self.last_peak_gb = round(peak, 2)

    # -- chat --------------------------------------------------------------

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
        from mlx_vlm.generate.diffusion import stream_diffusion_generate_from_kwargs
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        from context_guard import PREFILL_STEP_SIZE

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
            None,  # pixel_values; the chat path is text-only
            inputs.get("attention_mask"),
            [],  # skip_special_token_ids
            stream_kwargs,
            on_result=on_result,
        )

    def stream_chat(self, messages, *, emit, cancelled):
        from mlx_vlm.prompt_utils import apply_chat_template

        from context_guard import estimate_peak_gb, render_and_fit

        # ContextOverflow propagates: the caller turns it into an `error`
        # event, because the advice it carries is about this server's budget.
        prompt, n_tokens, dropped = render_and_fit(
            self.processor,
            self.config,
            messages,
            self.tokenizer,
            self.max_context,
            apply_chat_template,
        )

        emit(
            "start",
            {
                "backend": self.id,
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
            if cancelled.is_set():
                return False  # stops the denoiser at the next step
            if result.is_draft:
                emit(
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
                emit("token", {"kind": kind, "text": text})
            return True

        results = self._diffusion_stream(prompt, on_result)
        try:
            for _ in results:  # with on_result set, nothing is ever yielded
                pass
        finally:
            results.close()

        for kind, text in splitter.flush():
            (thinking if kind == "thinking" else answer).append(text)
            emit("token", {"kind": kind, "text": text})

        self._note_peak(last)
        return {
            "answer": "".join(answer).strip(),
            "thinking": "".join(thinking).strip(),
            "stats": {
                "prompt_tokens": n_tokens,
                "generation_tokens": getattr(last, "generation_tokens", None),
                "prompt_tps": round(getattr(last, "prompt_tps", 0) or 0, 1),
                "generation_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
                "peak_gb": self.last_peak_gb,
                "wall_seconds": round(time.time() - t0, 1),
            },
        }

    # -- vision ------------------------------------------------------------

    def image_reader(self, cancelled):
        """Uses stream_generate rather than the diffusion path: _diffusion_stream
        pins pixel_values to None and has never been exercised with an image,
        while this route is the one the accuracy measurements were taken on.
        Extraction wants correctness, not live denoising previews.
        """
        import extract
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        from context_guard import PREFILL_STEP_SIZE

        default_tokens = self.runtime["extract_max_tokens"]

        def read(pil_image, prompt, tokens=None, think=False):
            if cancelled.is_set():
                raise Cancelled()
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
                max_tokens=tokens or default_tokens,
                temperature=0.0,
                prefill_step_size=PREFILL_STEP_SIZE,
            ):
                chunks.append(r.text)
                last = r
            self._note_peak(last)
            # Thought is separated, not merged into the answer: mixing it in
            # would feed reasoning prose to the row parser.
            return extract.split_thought("".join(chunks))

        return read

    def stream_read_image(self, image, question, *, emit, cancelled):
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        from context_guard import PREFILL_STEP_SIZE

        emit("start", {"backend": self.id, "mode": "read",
                       "image_size": list(image.size)})

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
            self.model, self.processor, formatted, image=image,
            # Reasoning and answer share this budget, so it is larger than the
            # chat default; a cut-off here would stop mid-thought silently.
            max_tokens=max(self.max_tokens, self.runtime["read_max_tokens"]),
            temperature=self.temperature,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            if cancelled.is_set():
                break
            last = r
            for kind, text in splitter.feed(r.text):
                (thinking if kind == "thinking" else answer).append(text)
                emit("token", {"kind": kind, "text": text})
        for kind, text in splitter.flush():
            (thinking if kind == "thinking" else answer).append(text)
            emit("token", {"kind": kind, "text": text})

        self._note_peak(last)
        return {
            "answer": "".join(answer).strip(),
            "thinking": "".join(thinking).strip(),
            "stats": {
                "prompt_tokens": getattr(last, "prompt_tokens", 0) or 0,
                "generation_tokens": getattr(last, "generation_tokens", None),
                "prompt_tps": round(getattr(last, "prompt_tps", 0) or 0, 1),
                "generation_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
                "peak_gb": self.last_peak_gb,
                "wall_seconds": round(time.time() - t0, 1),
            },
        }


# ---------------------------------------------------------------------------
# remote: Google Gemini
# ---------------------------------------------------------------------------


class RemoteError(RuntimeError):
    """An upstream API said no. The message is shown to the user as-is."""


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiBackend(Backend):
    """Google's Generative Language API.

    Nothing here is serialized: the work happens on Google's machines, so the
    only local cost is a socket. The server runs these off the queue that
    exists to protect the local model's memory.
    """

    kind = "gemini"
    capabilities = {"vision": True, "drafts": False, "thinking": True}

    # Streamed generation can idle for a while between chunks on a long
    # thinking turn; a short read timeout would kill a working request.
    CONNECT_TIMEOUT = 10.0
    READ_TIMEOUT = 300.0

    def __init__(self, spec, runtime):
        super().__init__(spec, runtime)
        self.api_key_env = spec.get("api_key_env", "GEMINI_API_KEY")
        self.base_url = spec.get("base_url", GEMINI_BASE).rstrip("/")
        self.max_tokens = spec.get("max_tokens") or runtime["max_tokens"]
        self.temperature = spec.get("temperature")
        if self.temperature is None:
            self.temperature = runtime["temperature"]
        if not self.context_limit:
            self.context_limit = 1_000_000

    # -- plumbing ----------------------------------------------------------

    def api_key(self):
        return os.environ.get(self.api_key_env, "").strip()

    def available(self):
        if not self.api_key():
            return False, f"{self.api_key_env} 환경변수가 없습니다"
        return True, None

    def _client(self):
        import httpx

        key = self.api_key()
        if not key:
            raise RemoteError(f"{self.api_key_env} 환경변수가 없습니다")
        return httpx.Client(
            timeout=httpx.Timeout(self.READ_TIMEOUT, connect=self.CONNECT_TIMEOUT),
            headers={"x-goog-api-key": key, "content-type": "application/json"},
        )

    def _fit(self, messages):
        """Drop the oldest turns until the history plausibly fits.

        Deliberately crude -- four characters per token, no tokenizer. The
        local backend measures exactly because a miss there is an OOM; here a
        miss is a 400 from an API with a million-token window, which no real
        conversation in this UI approaches.
        """
        budget = self.context_limit * 4
        total = sum(len(m["content"]) for m in messages)
        dropped = 0
        while len(messages) > 1 and total > budget:
            total -= len(messages[0]["content"])
            messages = messages[1:]
            dropped += 1
        return messages, dropped

    @staticmethod
    def _contents(messages, image_b64=None):
        out = []
        for m in messages:
            parts = [{"text": m["content"]}]
            out.append({
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": parts,
            })
        if image_b64 and out:
            out[-1]["parts"].append(
                {"inline_data": {"mime_type": "image/png", "data": image_b64}}
            )
        return out

    def _body(self, contents, *, max_tokens, think):
        return {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": max_tokens,
                # includeThoughts only controls whether thoughts come back, not
                # whether the model thinks. A budget of 0 is rejected outright
                # by some 2.5 models, so it is never sent.
                "thinkingConfig": {"includeThoughts": bool(think)},
            },
        }

    @staticmethod
    def _stats(usage, t0):
        wall = round(time.time() - t0, 1)
        gen = usage.get("candidatesTokenCount")
        thoughts = usage.get("thoughtsTokenCount") or 0
        billed = (gen or 0) + thoughts
        return {
            "prompt_tokens": usage.get("promptTokenCount"),
            "generation_tokens": billed or gen,
            "prompt_tps": None,
            "generation_tps": round(billed / wall, 1) if billed and wall else None,
            "peak_gb": None,  # nothing runs on this machine
            "wall_seconds": wall,
        }

    @staticmethod
    def _error_message(response, body):
        try:
            detail = json.loads(body)["error"]["message"]
        except Exception:
            detail = (body or "")[:300] or response.reason_phrase
        return f"Gemini {response.status_code}: {detail}"

    # -- chat --------------------------------------------------------------

    def stream_chat(self, messages, *, emit, cancelled):
        messages, dropped = self._fit(list(messages))
        url = f"{self.base_url}/models/{self.model_name}:streamGenerateContent?alt=sse"
        body = self._body(
            self._contents(messages), max_tokens=self.max_tokens, think=True
        )

        emit("start", {
            "backend": self.id,
            "prompt_tokens": None,  # only known once the response reports it
            "dropped_turns": dropped,
            "remote": True,
        })

        answer, thinking = [], []
        usage = {}
        t0 = time.time()

        with self._client() as client:
            with client.stream("POST", url, json=body) as response:
                if response.status_code >= 400:
                    raw = response.read().decode("utf-8", "replace")
                    raise RemoteError(self._error_message(response, raw))
                for line in response.iter_lines():
                    if cancelled.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usageMetadata") or usage
                    for kind, text in _gemini_parts(chunk):
                        (thinking if kind == "thinking" else answer).append(text)
                        emit("token", {"kind": kind, "text": text})

        return {
            "answer": "".join(answer).strip(),
            "thinking": "".join(thinking).strip(),
            "stats": self._stats(usage, t0),
        }

    # -- vision ------------------------------------------------------------

    def _generate_once(self, contents, *, max_tokens, think):
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        body = self._body(contents, max_tokens=max_tokens, think=think)
        with self._client() as client:
            response = client.post(url, json=body)
            if response.status_code >= 400:
                raise RemoteError(self._error_message(response, response.text))
            return response.json()

    def image_reader(self, cancelled):
        """Same callable shape extract.extract_image already consumes, so the
        tiling and consensus pipeline runs unchanged on a remote model."""
        import extract

        default_tokens = self.runtime["extract_max_tokens"]

        def read(pil_image, prompt, tokens=None, think=False):
            if cancelled.is_set():
                raise Cancelled()
            data = self._generate_once(
                self._contents(
                    [{"role": "user", "content": prompt}], _png_b64(pil_image)
                ),
                max_tokens=tokens or default_tokens,
                think=think,
            )
            answer, thought = [], []
            for candidate in data.get("candidates", []):
                for kind, text in _parts_of(candidate):
                    (thought if kind == "thinking" else answer).append(text)
            # Gemini hands thoughts over already separated, so split_thought is
            # only here to strip a stray marker the model may have written into
            # the text itself; it returns the Reply the pipeline expects.
            reply = extract.split_thought("".join(answer))
            if thought and not reply.thinking:
                reply = reply._replace(thinking="".join(thought).strip())
            return reply

        return read

    def stream_read_image(self, image, question, *, emit, cancelled):
        emit("start", {"backend": self.id, "mode": "read",
                       "image_size": list(image.size), "remote": True})

        url = f"{self.base_url}/models/{self.model_name}:streamGenerateContent?alt=sse"
        body = self._body(
            self._contents([{"role": "user", "content": question}], _png_b64(image)),
            max_tokens=max(self.max_tokens, self.runtime["read_max_tokens"]),
            think=True,
        )

        answer, thinking = [], []
        usage = {}
        t0 = time.time()
        with self._client() as client:
            with client.stream("POST", url, json=body) as response:
                if response.status_code >= 400:
                    raw = response.read().decode("utf-8", "replace")
                    raise RemoteError(self._error_message(response, raw))
                for line in response.iter_lines():
                    if cancelled.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usageMetadata") or usage
                    for kind, text in _gemini_parts(chunk):
                        (thinking if kind == "thinking" else answer).append(text)
                        emit("token", {"kind": kind, "text": text})

        return {
            "answer": "".join(answer).strip(),
            "thinking": "".join(thinking).strip(),
            "stats": self._stats(usage, t0),
        }


def _parts_of(candidate):
    for part in (candidate.get("content") or {}).get("parts") or []:
        text = part.get("text")
        if not text:
            continue
        yield ("thinking" if part.get("thought") else "answer"), text


def _gemini_parts(chunk):
    for candidate in chunk.get("candidates", []):
        yield from _parts_of(candidate)


def _png_b64(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

KINDS = {
    "mlx-local": LocalDiffusionBackend,
    "gemini": GeminiBackend,
}


class Registry:
    """The configured backends, in the order models.json lists them."""

    def __init__(self, backends, default_id):
        self.backends = backends
        self._by_id = {b.id: b for b in backends}
        self.default_id = default_id

    def __iter__(self):
        return iter(self.backends)

    def get(self, backend_id):
        return self._by_id.get(backend_id)

    def resolve(self, backend_id):
        """The backend a job should use, falling back to the default when a
        conversation names one that is no longer configured."""
        return self._by_id.get(backend_id) or self._by_id.get(self.default_id)

    @property
    def default(self):
        return self._by_id.get(self.default_id)

    @property
    def local(self):
        for b in self.backends:
            if b.kind == "mlx-local":
                return b
        return None

    def describe(self):
        out = []
        for b in self.backends:
            d = b.describe()
            d["default"] = b.id == self.default_id
            out.append(d)
        return out


def load_registry(path, runtime, *, include_local=True, model_override=None):
    """Build the backends from models.json.

    `include_local=False` drops the in-process model entirely -- no MLX import,
    no 17GB load, no model lock -- which is what --no-local is for.
    """
    config = json.loads(path.read_text())
    backends = []
    for spec in config.get("backends", []):
        kind = spec.get("kind")
        cls = KINDS.get(kind)
        if cls is None:
            raise ValueError(f"unknown backend kind {kind!r} in {path.name}")
        if kind == "mlx-local":
            if not include_local:
                continue
            if model_override:
                spec = {**spec, "model": model_override}
        backends.append(cls(spec, runtime))

    if not backends:
        raise ValueError("no usable backends configured")

    default_id = config.get("default")
    if not any(b.id == default_id for b in backends):
        # The configured default was the local model and it was skipped; fall
        # back to the first backend that can actually run.
        usable = [b for b in backends if b.available()[0]]
        default_id = (usable or backends)[0].id
    return Registry(backends, default_id)
