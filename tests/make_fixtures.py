#!/usr/bin/env python3
"""Generate synthetic ledger pages with known ground truth.

The values are random and unguessable on purpose: a model that reproduces
INV-52445 cannot have inferred it from context, so a correct answer proves the
pixels were actually read.

    .venv/bin/python tests/make_fixtures.py
"""

import json
import pathlib
import random

from PIL import Image, ImageDraw, ImageFont

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


SCHEMA_COLUMNS = ("code", "sku", "qty", "amount")


def make_ledger(rows, width=1920, height=1080, seed=0, layout=None, header=False):
    """Render `rows` records; returns (image, records).

    `layout` is the left-to-right column order actually drawn. It defaults to
    the schema's own order; passing a different order (optionally with extra
    columns the schema does not want) produces a page whose layout has to be
    discovered rather than assumed.
    """
    rnd = random.Random(seed)
    records = [
        {
            "code": f"INV-{rnd.randint(10000, 99999)}",
            "sku": "".join(rnd.choice("ABCDEFGHJKLMNP") for _ in range(4)),
            "qty": str(rnd.randint(2, 97)),
            "amount": f"{rnd.randint(1000, 99999):,}",
            "date": f"2026-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}",
            "memo": rnd.choice(("정상", "보류", "확인필요", "완료")),
        }
        for _ in range(rows)
    ]

    layout = layout or SCHEMA_COLUMNS
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    pad = height * 0.04
    n_lines = rows + (1 if header else 0)
    line_h = (height - pad * 2) / (n_lines + 1)
    font = _font(max(6, int(line_h * 0.62)))
    col_w = (width - pad * 2) / (len(layout) + 0.2)

    row_index = 0
    if header:
        y = pad + line_h
        for ci, key in enumerate(layout):
            draw.text((pad + ci * col_w, y), key.upper(), font=font, fill="black")
        row_index = 1

    for i, rec in enumerate(records):
        y = pad + line_h * (i + 1 + row_index)
        for ci, key in enumerate(layout):
            draw.text((pad + ci * col_w, y), rec[key], font=font, fill="black")
    return img, records


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def emit(name, img, records, **meta):
        img.save(FIXTURES / f"{name}.png")
        (FIXTURES / f"{name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2)
        )
        manifest[name] = {"rows": len(records), "size": img.size, **meta}
        print(f"  {name}.png {img.size}, {len(records)} rows  {meta or ''}")

    for rows, seed in ((20, 42), (45, 7)):
        img, records = make_ledger(rows, seed=seed)
        emit(f"ledger_{rows}", img, records)

    # Same data, but the columns are shuffled and two the schema does not want
    # are mixed in. Assuming schema order here silently produces wrong values
    # in every row, so this is what the structure pre-pass has to earn.
    layout = ("date", "sku", "amount", "code", "memo", "qty")
    img, records = make_ledger(24, seed=13, layout=layout, header=True)
    emit("ledger_shuffled", img, records, layout=list(layout), header=True)

    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
