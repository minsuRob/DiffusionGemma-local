#!/usr/bin/env python3
"""Benchmark DiffusionGemma across a variety of prompts.

Runs a fixed suite of prompts (reasoning, code, Korean, summarization, math),
records output quality samples and speed stats, and writes results to
bench_results.json.
"""

import json
import re
import time

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"

SUITE = [
    ("simple-fact", "What is the capital of Australia? Answer in one sentence.", 64),
    ("korean", "한국의 사계절을 각각 한 문장으로 설명해줘.", 256),
    ("reasoning", "If a train leaves at 3:15 PM traveling 80 km/h and another leaves the same station at 3:45 PM traveling 100 km/h in the same direction, at what time does the second train catch up? Show your reasoning.", 512),
    ("code", "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring.", 384),
    ("math", "Compute 17 * 23 + 456 / 8. Show each step.", 256),
    ("summarize", "Summarize in 3 bullet points: The Industrial Revolution began in Britain in the late 18th century. It marked a shift from hand production to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools, and the rise of the mechanized factory system. It also led to unprecedented population growth and urbanization.", 256),
    ("creative", "Write a 4-line poem about the ocean at night.", 128),
    ("json-output", 'Return ONLY valid JSON: an object with keys "name", "age", "city" for a fictional person.', 128),
]


def strip_thinking(text):
    """Remove <|channel>thought ... <channel|> reasoning sections."""
    return re.sub(r"<\|channel>.*?(<channel\|>|$)", "", text, flags=re.S).strip()


def run_one(model, processor, config, prompt_text, max_tokens):
    messages = [{"role": "user", "content": prompt_text}]
    prompt = apply_chat_template(processor, config, messages, num_images=0)
    parts = []
    last = None
    t0 = time.time()
    for result in stream_generate(
        model, processor, prompt, max_tokens=max_tokens, temperature=0.0
    ):
        parts.append(result.text)
        last = result
    wall = time.time() - t0
    raw = "".join(parts)
    out = {
        "output": strip_thinking(raw),
        "raw_output": raw,
        "wall_seconds": round(wall, 2),
    }
    for attr in ("prompt_tokens", "generation_tokens", "prompt_tps",
                 "generation_tps", "peak_memory"):
        v = getattr(last, attr, None)
        if v is not None:
            out[attr] = round(v, 2) if isinstance(v, float) else v
    return out


def main():
    print(f"Loading {MODEL} ...", flush=True)
    t0 = time.time()
    model, processor = load(MODEL)
    config = load_config(MODEL)
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s\n", flush=True)

    results = {"model": MODEL, "load_seconds": round(load_s, 1), "cases": {}}
    for name, prompt_text, max_tokens in SUITE:
        print(f"=== {name} ===", flush=True)
        r = run_one(model, processor, config, prompt_text, max_tokens)
        results["cases"][name] = {"prompt": prompt_text, **r}
        print(r["output"][:600])
        print(f"--> wall={r['wall_seconds']}s gen_tps={r.get('generation_tps')} "
              f"gen_tok={r.get('generation_tokens')} peak={r.get('peak_memory')}GB\n",
              flush=True)

    with open("bench_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved bench_results.json")


if __name__ == "__main__":
    main()
