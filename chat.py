#!/usr/bin/env python3
"""Interactive chat CLI for DiffusionGemma (MLX, local).

Usage:
    .venv/bin/python chat.py [--model MODEL] [--max-tokens N] [--max-context N]

Commands inside the chat:
    /reset   clear conversation history
    /stats   toggle per-turn generation stats
    /quit    exit
"""

import argparse
import re
import sys
import time

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from context_guard import (
    PREFILL_STEP_SIZE,
    SOFT_CONTEXT_LIMIT,
    ContextOverflow,
    clamp_limit,
    get_tokenizer,
    render_and_fit,
)

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"

THINKING_RE = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.S)


def main():
    parser = argparse.ArgumentParser(description="DiffusionGemma local chat")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-context",
        type=int,
        default=SOFT_CONTEXT_LIMIT,
        help=f"prompt token budget (default {SOFT_CONTEXT_LIMIT:,}, hard ceiling 120,000)",
    )
    args = parser.parse_args()

    max_context, warning = clamp_limit(args.max_context)
    if warning:
        print(f"warning: {warning}")

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    model, processor = load(args.model)
    config = load_config(args.model)
    tokenizer = get_tokenizer(processor)
    print(
        f"Loaded in {time.time() - t0:.1f}s. "
        f"Context budget {max_context:,} tokens. Type /quit to exit.\n"
    )

    messages = []
    show_stats = True

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = []
            print("(history cleared)")
            continue
        if user == "/stats":
            show_stats = not show_stats
            print(f"(stats {'on' if show_stats else 'off'})")
            continue

        messages.append({"role": "user", "content": user})
        try:
            prompt, n_tokens, dropped = render_and_fit(
                processor, config, messages, tokenizer, max_context, apply_chat_template
            )
        except ContextOverflow as e:
            print(f"error: {e}.")
            print("       Shorten the message, or restart with a larger --max-context.")
            messages.pop()  # do not keep an unusable turn in history
            continue

        if dropped:
            print(
                f"(dropped the {dropped} oldest turn(s) to fit the "
                f"{max_context:,} token budget)"
            )

        print("assistant> ", end="", flush=True)
        reply_parts = []
        last = None
        for result in stream_generate(
            model,
            processor,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            print(result.text, end="", flush=True)
            reply_parts.append(result.text)
            last = result
        print()

        reply = "".join(reply_parts)
        # keep history clean of <|channel>thought ... <channel|> reasoning
        reply_clean = THINKING_RE.sub("", reply).strip()
        messages.append({"role": "assistant", "content": reply_clean or reply})

        if show_stats and last is not None:
            stats = [f"ctx={n_tokens:,}/{max_context:,}"]
            for attr, label in [
                ("generation_tokens", "gen tok"),
                ("prompt_tps", "prompt tps"),
                ("generation_tps", "gen tps"),
                ("peak_memory", "peak GB"),
            ]:
                v = getattr(last, attr, None)
                if v is not None:
                    stats.append(f"{label}={v:.1f}" if isinstance(v, float) else f"{label}={v}")
            print(f"  [{', '.join(stats)}]\n")


if __name__ == "__main__":
    sys.exit(main())
