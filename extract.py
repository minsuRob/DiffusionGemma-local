#!/usr/bin/env python3
"""Tiled, multi-pass data extraction from document images.

Two problems are being solved here.

*Accuracy* — the model shrinks any image to 224x224, so a dense page loses its
characters. Splitting into ~12-row tiles took a 45-row ledger from 62% to 95%
field accuracy at no extra wall-clock cost.

*Silence* — the errors that remain are substitutions like INV-65642 ->
INV-56642 and LFAH -> LLAH. They are correctly formatted and raise nothing, so
a single pass cannot tell you which fields to distrust. Reading each tile three
times from slightly different crops produces *different* errors each time, and
disagreement is the signal. Anything the passes do not unanimously agree on is
reported as such rather than being quietly emitted as fact.

CLI:
    .venv/bin/python extract.py page.png --schema schemas/ledger.json
"""

import argparse
import json
import pathlib
import re
import sys
import time
from collections import Counter, namedtuple

import docvision as dv

# What a reader hands back. `thinking` is the model's own reasoning channel,
# kept separate so it can be shown to the user instead of being silently
# dropped or, worse, mixed into the data being parsed.
Reply = namedtuple("Reply", "text thinking")

DEFAULT_PASSES = 3

# The structure probe writes prose, not table rows, so it needs a bigger budget
# than a tile read. Truncation here is invisible -- generation just stops -- so
# the margin is deliberate.
PROBE_MAX_TOKENS = 900

# Sending tiles one at a time is not an accident. Images cost far more memory
# than their token count suggests -- 8 in one call measured 19.3GB, 16 measured
# 22.0GB, and 32 reached 27.5GB and produced garbage. One per call stays at
# ~17.5GB.
MAX_MODEL_CALLS = 60

_STATUS_ORDER = ["ok", "resolved", "majority", "invalid", "conflict", "unresolved", "missing"]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def load_schema(path):
    schema = json.loads(pathlib.Path(path).read_text())
    if not schema.get("fields"):
        raise ValueError(f"{path}: schema needs a non-empty 'fields' list")
    names = [f["name"] for f in schema["fields"]]
    if schema.get("row_key") and schema["row_key"] not in names:
        raise ValueError(f"{path}: row_key '{schema['row_key']}' is not one of {names}")
    return schema


def field_names(schema):
    return [f["name"] for f in schema["fields"]]


def build_prompt(schema, structure=None):
    """Per-tile read instruction.

    Stays terse and forbids prose on purpose: reasoning happens in the probe
    and in the model's own thought channel, but the tile reply has to be
    parseable. `structure`, when the probe found one, tells the model what the
    columns actually are so it does not have to re-derive the layout from a
    12-row slice that may not include the header.
    """
    names = field_names(schema)
    header = "|".join(n.upper() for n in names)
    hints = []
    for f in schema["fields"]:
        bits = [f["name"]]
        if f.get("type"):
            bits.append(f["type"])
        if f.get("pattern"):
            bits.append(f"형식 {f['pattern']}")
        hints.append(" = ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0])

    context = ""
    if structure:
        parts = []
        if structure.get("doc_type"):
            parts.append(f"이 이미지는 {structure['doc_type']}이다.")
        if structure.get("columns"):
            parts.append(
                "표에 보이는 열은 왼쪽부터 " + ", ".join(structure["columns"]) + " 이다."
            )
        if structure.get("mapping"):
            pairs = "; ".join(f"{k} -> {v}" for k, v in structure["mapping"].items())
            parts.append(f"이 중 출력해야 할 열의 위치는 {pairs}. 나머지 열은 무시한다.")
        if structure.get("notes"):
            parts.append(f"주의: {structure['notes']}")
        if parts:
            context = "\n".join(parts) + "\n"

    return (
        context
        + "이미지의 표를 읽어라. 보이는 각 행을 다음 형식으로 정확히 한 줄씩 출력한다.\n"
        f"{header}\n"
        "규칙: 설명·머리말·코드블록 금지. 데이터 줄만. 값이 안 보이면 그 칸은 비워라. "
        "추측해서 채우지 마라.\n"
        "필드: " + "; ".join(hints)
    )


# ---------------------------------------------------------------------------
# structure probe
# ---------------------------------------------------------------------------


def build_probe_prompt(schema):
    names = field_names(schema)
    return (
        "이 이미지를 먼저 파악해라. 글자를 옮겨 적지 말고 구조만 판단한다.\n"
        "아래 항목을 정확히 이 키 이름으로, 한 줄에 하나씩 답한다.\n\n"
        "문서종류: (무슨 이미지인지 한 줄. 예: 거래 내역서 표, 영수증, 브라우저 스크린샷)\n"
        "표있음: 예 또는 아니오\n"
        "열목록: 표의 열 제목을 왼쪽부터 쉼표로. 제목이 없으면 각 열의 내용을 한 단어로.\n"
        f"열대응: 요청된 열({', '.join(names)})이 표의 몇 번째 열인지. "
        "형식은 이름=번호, 쉼표로 구분. 없는 열은 이름=없음.\n"
        "주의사항: 판독에 영향을 줄 만한 점. 없으면 없음.\n\n"
        "표가 아예 없으면 표있음: 아니오 라고만 하고 나머지는 없음으로 둔다. "
        "확실하지 않으면 추측하지 말고 모름이라고 적어라."
    )


_PROBE_KEYS = {
    "문서종류": "doc_type",
    "표있음": "has_table",
    "열목록": "columns",
    "열대응": "mapping",
    "주의사항": "notes",
}


def parse_probe(text, schema):
    """Turn the probe reply into a dict.

    Line-oriented `key: value` rather than JSON: a diffusion model that breaks
    a nested structure leaves nothing recoverable, whereas a missing line here
    just means that one field is unknown.
    """
    out = {"doc_type": "", "has_table": None, "columns": [], "mapping": {}, "notes": ""}
    for raw in text.splitlines():
        line = raw.strip().lstrip("*-• ").strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        field = _PROBE_KEYS.get(key.strip().replace(" ", ""))
        if not field:
            continue
        value = value.strip()
        if field == "has_table":
            out["has_table"] = value.startswith("예") or value.lower().startswith("yes")
        elif field == "columns":
            if value and value not in ("없음", "모름"):
                out["columns"] = [c.strip() for c in value.split(",") if c.strip()]
        elif field == "mapping":
            names = set(field_names(schema))
            for pair in value.split(","):
                name, _, pos = pair.partition("=")
                name, pos = name.strip(), pos.strip()
                if name in names and pos and pos not in ("없음", "모름"):
                    out["mapping"][name] = pos
        elif value not in ("없음", "모름"):
            out[field] = value
    return out


def probe_structure(reader, img, schema):
    """One look at the whole image before any tile is read.

    Returns (structure, reply). The caller decides whether to continue; this
    only reports what it saw.
    """
    reply = reader(img, build_probe_prompt(schema), PROBE_MAX_TOKENS, think=True)
    structure = parse_probe(reply.text, schema)
    structure["raw"] = reply.text.strip()
    structure["thinking"] = (reply.thinking or "").strip()
    return structure


def probe_rejects(structure, schema):
    """Reason to stop before tiling, or None to continue.

    Only an explicit 'no table' or a mapping that matched nothing stops the
    run. An uncertain probe is not allowed to veto -- being wrong here would
    silently skip a document the tiles could have read fine.
    """
    if structure.get("has_table") is False:
        doc = structure.get("doc_type") or "표가 아닌 이미지"
        return (
            f"이 이미지에는 표가 없습니다. 모델은 이것을 “{doc}”(으)로 판단했습니다."
        )
    if structure.get("columns") and not structure.get("mapping"):
        seen = ", ".join(structure["columns"])
        want = ", ".join(field_names(schema))
        return (
            f"표는 있지만 스키마와 맞는 열이 없습니다. "
            f"이미지에서 보이는 열은 [{seen}]이고, 스키마가 찾는 열은 [{want}]입니다."
        )
    return None


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_rows(text, schema):
    """Pull delimited rows out of a model reply, ignoring any prose around it."""
    names = field_names(schema)
    rows = []
    for raw in text.splitlines():
        line = raw.strip().strip("|").strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < len(names):
            continue
        parts = parts[: len(names)]
        # Skip a header the model echoed back.
        if all(p.upper() == n.upper() for p, n in zip(parts, names)):
            continue
        if not any(parts):
            continue
        rows.append(dict(zip(names, parts)))
    return rows


def normalize(value, field):
    """Canonical form for comparison: thousands separators and case do not
    constitute a disagreement between passes."""
    if value is None:
        return ""
    v = str(value).strip()
    if field.get("type") in ("integer", "number"):
        v = v.replace(",", "").replace(" ", "")
    else:
        v = v.upper()
    return v


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------


def _edit_distance(a, b, cap=3):
    """Levenshtein, abandoned once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _merge_misread_keys(order, buckets, max_distance=2):
    """Fold a one-off row key into the near-identical row most passes agreed on.

    Rows are grouped by their key, so a pass that misreads the key itself --
    INV-50580 read as INV-60580 -- spawns a second row instead of disagreeing
    about the first. Left alone that shows up as a phantom extra row *and*
    hides the fact that the key was misread. Merging recovers both: the row
    count is right, and the key now carries a visible 2/3 vote.
    """
    strong = [k for k in order if len(buckets[k]) >= 2]
    if not strong:
        return
    for key in list(order):
        if len(buckets[key]) >= 2:
            continue
        best, best_d = None, max_distance + 1
        for cand in strong:
            d = _edit_distance(key, cand)
            if d < best_d:
                best, best_d = cand, d
        # Require the difference to be small in absolute terms and small
        # relative to the key, so genuinely distinct short keys never merge.
        if best and best_d <= max_distance and best_d * 3 < max(len(key), len(best)):
            buckets[best].extend(buckets[key])
            del buckets[key]
            order.remove(key)


def _row_identity(row, schema):
    key = schema.get("row_key")
    if key:
        f = next(f for f in schema["fields"] if f["name"] == key)
        return normalize(row.get(key), f)
    return None


def reconcile(passes, schema):
    """Merge N passes into one record list with a status per field.

    Rows are matched across passes by `row_key`, never by position: a single
    dropped row would otherwise shift every later row and turn one miss into a
    whole-document mismatch.
    """
    names = field_names(schema)
    fields_by_name = {f["name"]: f for f in schema["fields"]}
    key = schema.get("row_key")

    if key:
        order = []
        buckets = {}
        for rows in passes:
            for row in rows:
                ident = _row_identity(row, schema)
                if not ident:
                    continue
                if ident not in buckets:
                    buckets[ident] = []
                    order.append(ident)
                buckets[ident].append(row)
        _merge_misread_keys(order, buckets)
        grouped = [buckets[i] for i in order]
    else:
        depth = max((len(p) for p in passes), default=0)
        grouped = [[p[i] for p in passes if i < len(p)] for i in range(depth)]

    n_passes = len(passes)
    records = []
    for variants in grouped:
        cells = {}
        for name in names:
            field = fields_by_name[name]
            seen = [normalize(v.get(name), field) for v in variants]
            raw_by_norm = {}
            for v in variants:
                raw_by_norm.setdefault(normalize(v.get(name), field), v.get(name, ""))

            counts = Counter(seen)
            top, hits = counts.most_common(1)[0]
            n = len(seen)
            if n == 0:
                status = "missing"
            elif hits == n and n >= 2:
                status = "ok"
            elif hits > n / 2:
                status = "majority"
            else:
                status = "conflict"
            # A single observation is not a consensus, whatever it says.
            if n == 1:
                status = "majority"
            cells[name] = {
                "value": raw_by_norm.get(top, ""),
                "status": status,
                "candidates": sorted(set(seen)) if status != "ok" else [],
                "votes": f"{hits}/{n}",
            }
        # A row that only some passes saw at all is its own kind of silence:
        # the values that were read may agree perfectly while a pass simply
        # missed the line. Record the observation count so that is visible.
        records.append({"cells": cells, "seen": len(variants), "passes": n_passes})
    return records


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _coerce_number(text):
    cleaned = re.sub(r"[^\d.\-]", "", str(text or ""))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def validate(records, schema):
    """Apply schema rules independently of the vote.

    Necessary because all passes can share a mistake; unanimity is evidence,
    not proof. A field that every pass agreed on still fails here if it does not
    match its declared shape.
    """
    fields_by_name = {f["name"]: f for f in schema["fields"]}
    problems = []

    for idx, record in enumerate(records):
        if record["seen"] < record["passes"]:
            problems.append(
                f"행 {idx + 1}: {record['passes']}회 중 {record['seen']}회만 관측됨"
            )
        for name, cell in record["cells"].items():
            field = fields_by_name.get(name, {})
            value = str(cell.get("value", "")).strip()

            if not value:
                if cell["status"] == "ok":
                    cell["status"] = "missing"
                continue

            pattern = field.get("pattern")
            if pattern and not re.match(pattern, value):
                cell["status"] = "invalid"
                cell["reason"] = f"'{pattern}' 형식과 다름"
                problems.append(f"행 {idx + 1} {name}: 형식 불일치 ({value})")
                continue

            if field.get("type") in ("integer", "number"):
                num = _coerce_number(value)
                if num is None:
                    cell["status"] = "invalid"
                    cell["reason"] = "숫자가 아님"
                    problems.append(f"행 {idx + 1} {name}: 숫자가 아님 ({value})")
                    continue
                if field.get("type") == "integer" and num != int(num):
                    cell["status"] = "invalid"
                    cell["reason"] = "정수가 아님"
                    problems.append(f"행 {idx + 1} {name}: 정수가 아님 ({value})")
                    continue
                if "min" in field and num < field["min"]:
                    cell["status"] = "invalid"
                    cell["reason"] = f"{field['min']} 미만"
                    problems.append(f"행 {idx + 1} {name}: 범위 밖 ({value})")
                if "max" in field and num > field["max"]:
                    cell["status"] = "invalid"
                    cell["reason"] = f"{field['max']} 초과"
                    problems.append(f"행 {idx + 1} {name}: 범위 밖 ({value})")

    for check in schema.get("checks", []):
        if check.get("type") == "sum":
            col = check["of"]
            total = sum(
                _coerce_number(r["cells"][col]["value"]) or 0
                for r in records
                if col in r["cells"]
            )
            expected = check.get("equals")
            if isinstance(expected, (int, float)) and abs(total - expected) > 0.01:
                problems.append(
                    f"합계 불일치: {col} 합 {total:,.0f} != 기대값 {expected:,.0f}"
                )
    return problems


def worst_status(record):
    """The status a reviewer should judge the whole row by."""
    if record["seen"] < record["passes"]:
        return "majority"
    present = [c["status"] for c in record["cells"].values()]
    for s in reversed(_STATUS_ORDER):
        if s in present:
            return s
    return "ok"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def extract_image(reader, img, schema, passes=DEFAULT_PASSES, on_progress=None,
                  rows_per_tile=dv.DEFAULT_ROWS_PER_TILE, max_calls=MAX_MODEL_CALLS,
                  probe=True):
    """Run the full pipeline over one image.

    `reader` is any callable (pil_image, prompt, max_tokens=None, think=False)
    -> Reply, so this works both from the CLI and from inside the server's
    worker.

    Reasoning is requested for the probe and never for a tile read. With
    thinking on, a tile reply spends its whole budget inside the thought
    channel -- the model reads the rows there, runs out of tokens before
    closing the channel, and returns an empty answer. So the thinking happens
    once, up front, and the tiles then do mechanical transcription against
    what it found.
    """
    started = time.time()

    def progress(**kw):
        if on_progress:
            on_progress(kw)

    calls = 0
    structure = None
    if probe:
        # Look at the whole image once before reading any of it. A tile is 12
        # rows with no header, so the layout has to be established here or each
        # tile guesses at it independently.
        progress(phase="probe")
        structure = probe_structure(reader, img, schema)
        calls += 1
        stop = probe_rejects(structure, schema)
        if stop:
            return {
                "records": [], "discarded": [], "problems": [],
                "fields": field_names(schema),
                "schema_name": schema.get("name", "schema"),
                "empty_reason": stop,
                "aborted": True,
                "samples": [structure["raw"][:400]] if structure.get("raw") else [],
                "analysis": structure,
                "stats": {
                    "rows": 0, "tiles": 0, "passes": passes, "model_calls": calls,
                    "rereads": 0, "lines_detected": 0, "fallback_tiling": False,
                    "truncated": False,
                    "wall_seconds": round(time.time() - started, 1),
                },
            }

    prompt = build_prompt(schema, structure)
    lines, tiles, fallback = dv.plan_document(img, rows_per_tile=rows_per_tile)

    # The probe already spent a call, so the tile budget is what is left.
    budget = max(1, max_calls - calls)
    truncated = False
    if len(tiles) * passes > budget:
        tiles = tiles[: max(1, budget // passes)]
        truncated = True

    pass_results = []
    samples = []  # kept so a zero-row result can show what the model did say
    for p in range(passes):
        rows = []
        for t_index, box in enumerate(tiles):
            jittered = dv.jitter_box(box, img.height, p)
            tile_img = dv.crop(img, jittered)
            progress(phase="tile", pass_index=p + 1, passes=passes,
                     tile=t_index + 1, tiles=len(tiles))
            reply = reader(tile_img, prompt)
            if p == 0 and len(samples) < 2:
                samples.append(reply.text.strip()[:400])
            rows.extend(parse_rows(reply.text, schema))
            calls += 1
        pass_results.append(rows)

    records = reconcile(pass_results, schema)

    # Targeted re-read: a row the passes could not agree on gets shown alone and
    # magnified, which puts many more pixels per character into the encoder.
    rereads = 0
    if lines and schema.get("row_key"):
        key = schema["row_key"]
        line_by_row = _map_records_to_lines(records, lines, key)
        for idx, record in enumerate(records):
            if not any(c["status"] == "conflict" for c in record["cells"].values()):
                continue
            line = line_by_row.get(idx)
            if line is None or calls >= max_calls:
                continue
            progress(phase="reread", row=idx + 1)
            zoomed = dv.crop_row(img, line)
            again = parse_rows(reader(zoomed, prompt).text, schema)
            calls += 1
            rereads += 1
            _apply_reread(record, again, schema)

    problems = validate(records, schema)
    records, discarded = _split_phantoms(records, schema, passes)
    if discarded:
        problems.append(
            f"판독 실패로 제외한 행 {len(discarded)}개 (타일 경계에서 깨진 텍스트로 보임)"
        )

    return {
        "records": records,
        "discarded": discarded,
        # With no rows there is nothing to have agreed on, so the caller must
        # not present it as a clean result. Hand back what the model actually
        # replied so the reason is visible instead of guessed at.
        "empty_reason": None if records else (
            "이 이미지에서 스키마와 맞는 행을 찾지 못했습니다."
        ),
        "samples": samples if not records else [],
        "aborted": False,
        "analysis": structure,
        "problems": problems,
        "fields": field_names(schema),
        "schema_name": schema.get("name", "schema"),
        "stats": {
            "rows": len(records),
            "tiles": len(tiles),
            "passes": passes,
            "model_calls": calls,
            "rereads": rereads,
            "lines_detected": len(lines),
            "fallback_tiling": fallback,
            "truncated": truncated,
            "wall_seconds": round(time.time() - started, 1),
        },
    }


def _split_phantoms(records, schema, passes):
    """Separate rows that are almost certainly garbled duplicates.

    Tiles overlap, so a line sitting on a boundary can be read as mangled text
    in one pass and produce a row key that matches nothing. Two signals have to
    agree before a row is set aside -- only one pass ever saw it, and its key
    fails the schema's own format -- and even then it is returned to the caller
    rather than dropped, because silently deleting a row would be the same
    failure mode this whole pipeline exists to prevent.
    """
    key = schema.get("row_key")
    if not key or passes < 3:
        return records, []
    field = next((f for f in schema["fields"] if f["name"] == key), None)
    pattern = field.get("pattern") if field else None
    if not pattern:
        return records, []

    kept, phantoms = [], []
    for rec in records:
        value = str(rec["cells"][key]["value"]).strip()
        if rec["seen"] <= 1 and not re.match(pattern, value):
            phantoms.append(rec)
        else:
            kept.append(rec)
    return kept, phantoms


def _map_records_to_lines(records, lines, key):
    """Best-effort record index -> detected line box.

    Rows come back in reading order, so when the counts line up the mapping is
    positional. When they do not, re-reads are skipped rather than guessed at.
    """
    if len(records) == len(lines):
        return dict(enumerate(lines))
    return {}


def _apply_reread(record, again, schema):
    """Let a magnified single-row read break a tie."""
    if not again:
        return
    fields_by_name = {f["name"]: f for f in schema["fields"]}
    fresh = again[0]
    for name, cell in record["cells"].items():
        if cell["status"] != "conflict":
            continue
        field = fields_by_name[name]
        value = normalize(fresh.get(name), field)
        if not value:
            cell["status"] = "unresolved"
            cell["value"] = ""
            continue
        if value in cell["candidates"]:
            cell["value"] = fresh.get(name, "")
            cell["status"] = "resolved"
        else:
            # The re-read invented a third answer; nothing here is trustworthy.
            cell["status"] = "unresolved"
            cell["value"] = ""
            cell["candidates"] = sorted(set(cell["candidates"]) | {value})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


THOUGHT_RE = re.compile(r"<\|channel>[^\n]*\n?(.*?)(?:<channel\|>|$)", re.S)


def split_thought(text):
    """Separate the model's reasoning channel from its answer.

    Returns Reply(answer, thinking). Used by readers that get the whole reply
    as one string; the server streams and uses ChannelSplitter instead.
    """
    thoughts = [m.group(1).strip() for m in THOUGHT_RE.finditer(text)]
    return Reply(THOUGHT_RE.sub("", text).strip(), "\n".join(t for t in thoughts if t))


def make_model_reader(model_name, max_tokens=700):
    """Load the model and return a reader callable. CLI use only -- inside the
    server the Engine already owns the model and supplies its own reader."""
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    from context_guard import PREFILL_STEP_SIZE

    model, processor = load(model_name)
    config = load_config(model_name)

    def read(pil_image, prompt, tokens=None, think=False):
        formatted = apply_chat_template(
            processor, config, [{"role": "user", "content": prompt}], num_images=1,
            # mlx-vlm forces this to False unless it is passed. Off by default
            # here: see the note on the reader contract in extract_image.
            enable_thinking=think,
        )
        out = []
        for r in stream_generate(
            model, processor, formatted, image=pil_image,
            max_tokens=tokens or max_tokens, temperature=0.0,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            out.append(r.text)
        return split_thought("".join(out))

    return read


def format_table(result):
    names = result["fields"]
    marks = {"ok": " ", "majority": "?", "resolved": "~", "conflict": "!",
             "unresolved": "!", "invalid": "X", "missing": "-"}
    widths = [max(len(n), 12) for n in names]
    out = ["  ".join(n.upper().ljust(w) for n, w in zip(names, widths))]
    out.append("  ".join("-" * w for w in widths))
    for rec in result["records"]:
        cells = []
        for n, w in zip(names, widths):
            c = rec["cells"][n]
            cells.append(f"{marks.get(c['status'],' ')}{c['value']}".ljust(w))
        row = "  ".join(cells)
        if rec["seen"] < rec["passes"]:
            row += f"   ({rec['passes']}회 중 {rec['seen']}회 관측)"
        out.append(row)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Tiled multi-pass document extraction")
    ap.add_argument("image")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    ap.add_argument("--rows-per-tile", type=int, default=dv.DEFAULT_ROWS_PER_TILE)
    ap.add_argument("--model", default="mlx-community/diffusiongemma-26B-A4B-it-4bit")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the structure pre-pass (for comparing against it)")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    img = dv.load_image(args.image)
    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    reader = make_model_reader(args.model)

    def progress(info):
        if info.get("phase") == "probe":
            print("  구조 파악 중…", file=sys.stderr, flush=True)
        elif info.get("phase") == "tile":
            print(f"  pass {info['pass_index']}/{info['passes']} "
                  f"tile {info['tile']}/{info['tiles']}", file=sys.stderr, flush=True)
        elif info.get("phase") == "reread":
            print(f"  re-reading row {info['row']}", file=sys.stderr, flush=True)

    result = extract_image(reader, img, schema, passes=args.passes,
                           on_progress=progress, rows_per_tile=args.rows_per_tile,
                           probe=not args.no_probe)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    a = result.get("analysis")
    if a:
        print()
        if a.get("doc_type"):
            print(f"파악: {a['doc_type']}")
        if a.get("columns"):
            print(f"열  : {', '.join(a['columns'])}")
        if a.get("notes"):
            print(f"주의: {a['notes']}")

    if result.get("aborted"):
        print()
        print(f"중단: {result['empty_reason']}")
        print(f"모델 호출 {result['stats']['model_calls']}회, "
              f"{result['stats']['wall_seconds']}s")
        print("표가 아닌 이미지는 서버의 '이미지 읽기' 모드를 쓰세요.")
        return

    print()
    print(format_table(result))
    print()
    s = result["stats"]
    print(f"rows={s['rows']} tiles={s['tiles']} passes={s['passes']} "
          f"calls={s['model_calls']} rereads={s['rereads']} {s['wall_seconds']}s")
    flagged = sum(1 for r in result["records"] if worst_status(r) != "ok")
    print(f"검토 필요 행: {flagged}/{s['rows']}   (? 다수결  ~ 재판독  ! 불일치  X 형식오류)")
    for p in result["problems"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
