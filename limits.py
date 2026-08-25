#!/usr/bin/env python3
"""Probe DiffusionGemma input-length limits on this machine.

Feeds progressively longer prompts (with a needle-in-haystack question at the
end) and records prefill speed, memory, and whether the needle is retrieved.
Stops when generation fails (OOM / error). Writes limits_results.json.
"""

import gc
import json
import time
import traceback

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "autumn leaves drift slowly across the quiet meadow in the fading light. "
)
NEEDLE = "The secret passphrase is BLUE-FALCON-7291. "
QUESTION = "\n\nWhat is the secret passphrase mentioned above? Answer with just the passphrase."

# target approximate prompt token sizes
TARGETS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000]

# Chunked prefill: without this, prompts over ~16K tokens fail with a Metal
# single-buffer allocation error (attention materializes one huge buffer).
PREFILL_STEP_SIZE = 2048


def build_prompt(tokenizer, target_tokens):
    filler_tokens = len(tokenizer.encode(FILLER))
    reps = max(1, int(target_tokens / filler_tokens))
    half = reps // 2
    text = FILLER * half + NEEDLE + FILLER * (reps - half) + QUESTION
    return text


def main():
    print(f"Loading {MODEL} ...", flush=True)
    model, processor = load(MODEL)
    config = load_config(MODEL)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print("Loaded.\n", flush=True)

    results = []
    for target in TARGETS:
        text = build_prompt(tokenizer, target)
        messages = [{"role": "user", "content": text}]
        prompt = apply_chat_template(processor, config, messages, num_images=0)
        n_tokens = len(tokenizer.encode(prompt))
        print(f"=== target≈{target} actual={n_tokens} tokens ===", flush=True)

        entry = {"target": target, "prompt_tokens": n_tokens}
        try:
            t0 = time.time()
            parts = []
            last = None
            for result in stream_generate(
                model, processor, prompt, max_tokens=64, temperature=0.0,
                prefill_step_size=PREFILL_STEP_SIZE,
            ):
                parts.append(result.text)
                last = result
            wall = time.time() - t0
            output = "".join(parts).strip()
            entry.update(
                ok=True,
                wall_seconds=round(wall, 1),
                output=output[:200],
                needle_found="BLUE-FALCON-7291" in output,
            )
            for attr in ("prompt_tps", "generation_tps", "peak_memory"):
                v = getattr(last, attr, None)
                if v is not None:
                    entry[attr] = round(v, 2) if isinstance(v, float) else v
            print(f"ok wall={wall:.1f}s prompt_tps={entry.get('prompt_tps')} "
                  f"peak={entry.get('peak_memory')}GB needle={entry['needle_found']}",
                  flush=True)
            print(f"output: {output[:120]}", flush=True)
        except Exception as e:
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
            print(f"FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
            results.append(entry)
            break
        finally:
            gc.collect()
            mx.clear_cache()

        results.append(entry)
        print(flush=True)

    with open("limits_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved limits_results.json")


if __name__ == "__main__":
    main()
