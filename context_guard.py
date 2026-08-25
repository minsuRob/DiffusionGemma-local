"""Shared context-length limits and history trimming for DiffusionGemma.

Measured on this machine (M5 Max / 36GB) with 1024-token outputs and
prefill_step_size=2048:

    input   peak      wall    gen speed
     77K    24.6GB     66s    34 tok/s
     96K    26.5GB     94s    27 tok/s
    116K    27.9GB    131s    25 tok/s

Linear fit: peak_GB ~= 18.0 + 0.085 * (tokens / 1000)

Three constraints set the ceiling:
  1. iogpu.wired_limit_mb defaults to ~27GB on a 36GB machine; 116K already
     exceeds it, so anything beyond that can start paging under load.
  2. At 120K only ~8GB is left for macOS and everything else.
  3. Latency degrades faster than memory: 200 tok/s at short context drops to
     25 tok/s at 116K, and a single turn takes over two minutes.
"""

SOFT_CONTEXT_LIMIT = 96_000
HARD_CONTEXT_LIMIT = 120_000

# Without chunked prefill, prompts over ~16K tokens fail with a Metal
# single-buffer allocation error (attention materializes one huge buffer).
PREFILL_STEP_SIZE = 2048

# Coefficients of the linear fit above, used to predict memory before running.
_PEAK_BASE_GB = 18.0
_PEAK_GB_PER_1K = 0.085


def estimate_peak_gb(n_tokens):
    """Predict peak memory for a prompt of n_tokens, from the measured fit."""
    return _PEAK_BASE_GB + _PEAK_GB_PER_1K * (n_tokens / 1000)


def clamp_limit(requested):
    """Clamp a requested context limit to HARD_CONTEXT_LIMIT.

    Returns (limit, warning_or_None).
    """
    if requested > HARD_CONTEXT_LIMIT:
        return HARD_CONTEXT_LIMIT, (
            f"requested context {requested:,} exceeds the measured safe ceiling; "
            f"clamped to {HARD_CONTEXT_LIMIT:,} "
            f"(~{estimate_peak_gb(HARD_CONTEXT_LIMIT):.1f}GB peak, leaving ~8GB for the system)"
        )
    if requested < 1000:
        return 1000, f"requested context {requested:,} is too small; raised to 1,000"
    return requested, None


def get_tokenizer(processor):
    """Pull the tokenizer out of an mlx-vlm processor."""
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


class ContextOverflow(Exception):
    """A single message alone exceeds the limit, so trimming cannot help."""

    def __init__(self, n_tokens, limit):
        self.n_tokens = n_tokens
        self.limit = limit
        super().__init__(
            f"this message alone renders to {n_tokens:,} tokens, over the "
            f"{limit:,} token limit; there is no history left to drop"
        )


def render_and_fit(processor, config, messages, tokenizer, limit, apply_chat_template):
    """Render messages to a prompt that fits within `limit` tokens.

    Drops the oldest user/assistant pairs until the rendered prompt fits.
    Measuring the *rendered* prompt matters: the chat template adds special
    tokens, and only the rendered length matches what prefill actually sees.

    Mutates `messages` in place so the caller's history stays in sync.

    Returns (prompt, n_tokens, dropped_turns).
    Raises ContextOverflow if the newest message alone is too large.
    """
    dropped = 0
    while True:
        prompt = apply_chat_template(processor, config, messages, num_images=0)
        n_tokens = len(tokenizer.encode(prompt))
        if n_tokens <= limit:
            return prompt, n_tokens, dropped

        # Only the newest user message is left and it still does not fit.
        if len(messages) <= 1:
            raise ContextOverflow(n_tokens, limit)

        # Drop the oldest exchange. Messages alternate user/assistant, so
        # removing two entries drops one complete turn; if the history starts
        # with a lone assistant message, removing one is correct.
        del messages[0]
        if messages and messages[0]["role"] == "assistant":
            del messages[0]
        dropped += 1
