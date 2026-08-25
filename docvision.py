"""Image splitting for document extraction.

Everything here is pure image geometry with no model involved, so it can be
tested on its own.

Why split at all: the model resizes every image to 224x224 regardless of the
source, so accuracy is set by how much text is packed into one image, not by
resolution. Measured on synthetic ledgers (field accuracy):

    10 rows 100%   20 rows 85%   30 rows 91%   45 rows 66%

Splitting a 45-row page into three 15-row tiles moved it from 62% to 95%, and
was slightly *faster* than the single call, so there is no cost to trade off.
"""

from PIL import Image
import numpy as np

# Beyond this the extra pixels are thrown away by the 224x224 resize anyway,
# and large sources just cost memory.
MAX_SOURCE_SIDE = 4000

# Rows per tile. 10 rows measured 100% and 20 rows 85%, so stay nearer 10.
DEFAULT_ROWS_PER_TILE = 12

# Repeat a line at the tile boundary so a row split across two tiles is still
# read whole by at least one of them.
DEFAULT_OVERLAP_ROWS = 1

# A detected band thinner than this fraction of the image height is treated as
# speckle rather than a line of text.
_MIN_LINE_FRAC = 0.004


def load_image(path, max_side=MAX_SOURCE_SIDE):
    """Open an image as RGB, downscaling only if it is enormous."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    return img


def _ink_profile(img):
    """Per-row ink density in 0..1, orientation-agnostic.

    Returns (profile, threshold). Light-on-dark pages are inverted first so
    "ink" always means "differs from the page background".
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    # The background is whichever tone dominates; measure distance from it.
    background = np.median(gray)
    ink = np.abs(gray - background)
    profile = ink.mean(axis=1)
    if profile.max() <= 1e-6:
        return profile, None
    # Otsu on the row profile separates text rows from gaps without needing a
    # tuned constant, which matters across scans, screenshots and photos.
    return profile, _otsu(profile)


def _otsu(values):
    """Otsu threshold over a 1-D signal."""
    hist, edges = np.histogram(values, bins=64, range=(values.min(), values.max()))
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    centers = (edges[:-1] + edges[1:]) / 2
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(values.mean())
    cum_mean = np.cumsum(hist * centers)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros_like(cum_mean), where=weight_bg > 0)
    total_mean = cum_mean[-1]
    mean_fg = np.divide(
        total_mean - cum_mean, weight_fg, out=np.zeros_like(cum_mean), where=weight_fg > 0
    )
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -1
    return float(centers[int(np.argmax(variance))])


def detect_text_lines(img, min_line_frac=_MIN_LINE_FRAC):
    """Find horizontal bands that contain text.

    Returns [(top, bottom), ...] in pixels, top-to-bottom. Empty when the image
    has no discernible line structure (a photo, a chart), which the caller
    should treat as "fall back to fixed slicing" rather than as an error.
    """
    profile, threshold = _ink_profile(img)
    if threshold is None:
        return []

    is_text = profile > threshold
    min_h = max(2, int(img.height * min_line_frac))

    lines = []
    start = None
    for y, on in enumerate(is_text):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_h:
                lines.append((start, y))
            start = None
    if start is not None and len(is_text) - start >= min_h:
        lines.append((start, len(is_text)))

    # A page of solid ink (a photo) produces one band covering everything;
    # that is not line structure and splitting on it would be arbitrary.
    if len(lines) == 1 and (lines[0][1] - lines[0][0]) > img.height * 0.8:
        return []
    return lines


def plan_tiles(
    lines,
    image_height,
    rows_per_tile=DEFAULT_ROWS_PER_TILE,
    overlap_rows=DEFAULT_OVERLAP_ROWS,
):
    """Group detected lines into tile boxes as (top, bottom) pixel pairs.

    Cuts land in the gaps between lines, never through one. Consecutive tiles
    repeat `overlap_rows` lines so a row on a boundary is read intact somewhere.
    """
    if not lines:
        return []
    if rows_per_tile <= 0:
        raise ValueError("rows_per_tile must be positive")
    overlap_rows = max(0, min(overlap_rows, rows_per_tile - 1))

    # Spread the rows evenly instead of filling each tile to the brim: 20 rows
    # at 12 per tile would otherwise be 12 + 9, and the fuller tile is the less
    # accurate one. Balancing makes it 10 + 10.
    tile_count = max(1, -(-len(lines) // max(1, rows_per_tile - overlap_rows)))
    if tile_count > 1:
        rows_per_tile = min(rows_per_tile, -(-len(lines) // tile_count) + overlap_rows)

    tiles = []
    start = 0
    step = max(1, rows_per_tile - overlap_rows)
    while start < len(lines):
        group = lines[start : start + rows_per_tile]
        first_top = group[0][0]
        last_bottom = group[-1][1]

        # Extend into the surrounding gaps so ascenders and descenders survive.
        prev_bottom = lines[start - 1][1] if start > 0 else 0
        after = start + len(group)
        next_top = lines[after][0] if after < len(lines) else image_height
        top = max(0, (first_top + prev_bottom) // 2 if start > 0 else max(0, first_top - 8))
        bottom = min(
            image_height,
            (last_bottom + next_top) // 2 if after < len(lines) else min(image_height, last_bottom + 8),
        )
        tiles.append((top, bottom))

        if after >= len(lines):
            break
        start += step
    return tiles


def fallback_tiles(image_height, count, overlap_frac=0.06):
    """Even slices with overlap, for images where no lines were detected."""
    count = max(1, count)
    if count == 1:
        return [(0, image_height)]
    band = image_height / count
    pad = int(band * overlap_frac)
    out = []
    for i in range(count):
        top = max(0, int(i * band) - pad)
        bottom = min(image_height, int((i + 1) * band) + pad)
        out.append((top, bottom))
    return out


def jitter_box(box, image_height, pass_index):
    """Shift a tile box slightly for pass `pass_index`.

    This is what makes multi-pass consensus meaningful. Generation runs at
    temperature 0, so an identical crop returns an identical answer and extra
    passes would be pure cost. Moving the crop changes what the 224x224 resize
    lands on, which changes *which* characters get misread -- and that is
    exactly the independence a majority vote needs.
    """
    top, bottom = box
    if pass_index == 0:
        return (top, bottom)
    height = bottom - top
    # A few percent is enough to change the resampling grid without dropping
    # any text out of the tile.
    shift = int(height * (0.03 * pass_index))
    grow = int(height * (0.02 * pass_index))
    new_top = max(0, top - shift)
    new_bottom = min(image_height, bottom + grow)
    if new_bottom <= new_top:
        return (top, bottom)
    return (new_top, new_bottom)


def crop(img, box, upscale=1.0):
    """Crop a full-width horizontal band, optionally magnified."""
    top, bottom = box
    top = max(0, min(top, img.height - 1))
    bottom = max(top + 1, min(bottom, img.height))
    out = img.crop((0, top, img.width, bottom))
    if upscale and upscale != 1.0:
        out = out.resize(
            (max(1, int(out.width * upscale)), max(1, int(out.height * upscale))),
            Image.LANCZOS,
        )
    return out


def crop_row(img, line, pad_frac=0.6, upscale=2.0):
    """Crop a single text line, padded and magnified, for a targeted re-read.

    Used when the passes disagree on a row: showing the model that row alone
    puts far more pixels per character into the 224x224 the encoder sees.
    """
    top, bottom = line
    pad = int((bottom - top) * pad_frac)
    return crop(img, (top - pad, bottom + pad), upscale=upscale)


def plan_document(img, rows_per_tile=DEFAULT_ROWS_PER_TILE, overlap_rows=DEFAULT_OVERLAP_ROWS):
    """Full plan for one image: detected lines plus the tile boxes to send.

    Returns (lines, tiles, used_fallback).
    """
    lines = detect_text_lines(img)
    if lines:
        return lines, plan_tiles(lines, img.height, rows_per_tile, overlap_rows), False
    # No line structure: slice by height so a dense photo of a table still gets
    # broken up rather than sent whole.
    count = max(1, round(img.height / max(1, img.width) * 2))
    return [], fallback_tiles(img.height, count), True
