"""White-on-blue card blob finder.

Extracts rectangular card-like regions from a BGR table screenshot by masking
bright card faces against blue felt, then splitting wide multi-card strips on
the thin blue gaps between cards.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

# White cards on blue felt (match crop_cards_from_image thresholds).
CARD_WHITE_MIN = 140
FELT_B_DOMINANCE = 15

# Single card is portrait: width/height ~ 0.6–0.75.
DEFAULT_MIN_ASPECT = 0.42
DEFAULT_MAX_ASPECT = 0.82
MAX_SINGLE_CARD_RATIO = 0.85
# Overlapping hole pairs are often ~0.75–1.1 wide before seam-split.
PAIR_SPLIT_ASPECT = 0.72

# Contours often miss the bottom edge; pad slightly.
BOTTOM_PAD_PX = 8
BOTTOM_PAD_FRAC = 0.04


def find_card_blobs(
    image_bgr: np.ndarray,
    *,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    min_aspect: float = DEFAULT_MIN_ASPECT,
    max_aspect: Optional[float] = None,
    min_width: int = 15,
    min_height: int = 25,
) -> List[Box]:
    """Find card-like bounding boxes in a BGR image.

    Returns ``(x1, y1, x2, y2)`` boxes sorted left-to-right by center x.
    Wide multi-card strips are split on blue seams before return.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be an HxWx3 BGR array")

    h, w = image_bgr.shape[:2]
    # Full browser windows dwarf the cropped table shots this was tuned on.
    if min_area is None:
        min_area = max(500.0, (w * h) / 2200.0)
    if max_area is None:
        max_area = (w * h) / 2.0
    # Absolute floors scale with frame — kills seam-split UI crumbs.
    min_width = max(min_width, int(0.028 * w))
    min_height = max(min_height, int(0.055 * h))

    # Keep wide multi-card strips here; seam-split below, then aspect-filter.
    collect_max_aspect = None

    regions = _contours_from_edges(
        image_bgr,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=collect_max_aspect,
        min_width=min_width,
        min_height=min_height,
    )
    regions.extend(
        _contours_from_white_mask(
            image_bgr,
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=collect_max_aspect,
            min_width=min_width,
            min_height=min_height,
        )
    )

    # Hero holes: bottom band + mid-side seats (BTN/CO style, not only south).
    seat_bands = [
        (0.0, 0.58, 1.0, 1.0),   # bottom
        (0.55, 0.10, 1.0, 0.58),  # mid-right
        (0.0, 0.10, 0.45, 0.58),  # mid-left
    ]
    for x0f, y0f, x1f, y1f in seat_bands:
        x0, y0 = int(w * x0f), int(h * y0f)
        x1, y1 = int(w * x1f), int(h * y1f)
        if y1 - y0 < 40 or x1 - x0 < 40:
            continue
        band = image_bgr[y0:y1, x0:x1, :]
        band_min = max(280.0, min_area * 0.40)
        extra = _contours_from_white_mask(
            band,
            min_area=band_min,
            max_area=max_area * 0.25,
            min_aspect=min_aspect,
            max_aspect=collect_max_aspect,
            min_width=min_width,
            min_height=min_height,
        )
        for bx1, by1, bx2, by2 in extra:
            regions.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))

    split: List[Box] = []
    for box in regions:
        bw = box[2] - box[0]
        bh = max(1, box[3] - box[1])
        ratio = bw / float(bh)
        if ratio > MAX_SINGLE_CARD_RATIO:
            # Overlapped hole pair (~1.0–1.45): bisect. Wider board strips: gap-split.
            if ratio <= 1.45:
                mid = (box[0] + box[2]) // 2
                split.append((box[0], box[1], mid, box[3]))
                split.append((mid, box[1], box[2], box[3]))
            else:
                split.extend(_split_wide_by_gaps(image_bgr, box))
        elif ratio > PAIR_SPLIT_ASPECT:
            # Slightly wide single / soft overlap: try seams, else bisect.
            parts = _split_wide_by_gaps(image_bgr, box)
            if len(parts) >= 2:
                split.extend(parts)
            else:
                mid = (box[0] + box[2]) // 2
                split.append((box[0], box[1], mid, box[3]))
                split.append((mid, box[1], box[2], box[3]))
        else:
            split.append(box)

    split = _merge_duplicates(split)
    split = [_extend_bottom(box, h) for box in split]
    aspect_hi = max_aspect if max_aspect is not None else DEFAULT_MAX_ASPECT
    out: List[Box] = []
    for box in split:
        bw = box[2] - box[0]
        bh = max(1, box[3] - box[1])
        ratio = bw / float(bh)
        if ratio < min_aspect or ratio > aspect_hi:
            continue
        if bw < min_width or bh < min_height:
            continue
        out.append(box)
    out.sort(key=lambda box: (box[0] + box[2]) // 2)
    return out


def filter_by_aspect_ratio(
    boxes: Sequence[Box],
    *,
    min_aspect: float = DEFAULT_MIN_ASPECT,
    max_aspect: float = DEFAULT_MAX_ASPECT,
) -> List[Box]:
    """Keep boxes whose width/height falls in ``[min_aspect, max_aspect]``."""
    out: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bh <= 0:
            continue
        ratio = bw / float(bh)
        if min_aspect <= ratio <= max_aspect:
            out.append((x1, y1, x2, y2))
    return out


def filter_by_min_area(boxes: Sequence[Box], min_area: float) -> List[Box]:
    """Keep boxes with pixel area ``>= min_area``."""
    out: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        if (x2 - x1) * (y2 - y1) >= min_area:
            out.append((x1, y1, x2, y2))
    return out


def _card_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    b, g, r = cv2.split(image_bgr)
    bright = np.minimum(np.minimum(r, g), b)
    card_like = (bright >= CARD_WHITE_MIN) | (
        (blurred > 100) & (b < np.maximum(r, g) + FELT_B_DOMINANCE)
    )
    mask = np.uint8(np.where(card_like, 255, 0))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _contours_from_edges(
    image_bgr: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: Optional[float],
    min_width: int,
    min_height: int,
) -> List[Box]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    mask = _card_mask(image_bgr)
    edges = cv2.bitwise_and(edges, cv2.dilate(mask, np.ones((3, 3), np.uint8)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _boxes_from_contours(
        contours,
        img_w=w,
        img_h=h,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_width=min_width,
        min_height=min_height,
    )


def _contours_from_white_mask(
    image_bgr: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: Optional[float],
    min_width: int,
    min_height: int,
) -> List[Box]:
    h, w = image_bgr.shape[:2]
    mask = _card_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _boxes_from_contours(
        contours,
        img_w=w,
        img_h=h,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_width=min_width,
        min_height=min_height,
    )


def _boxes_from_contours(
    contours,
    *,
    img_w: int,
    img_h: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: Optional[float],
    min_width: int,
    min_height: int,
) -> List[Box]:
    regions: List[Box] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw < min_width or rh < min_height:
            continue
        ratio = rw / float(rh)
        if ratio < min_aspect:
            continue
        if max_aspect is not None and ratio > max_aspect:
            continue
        pad = 2
        bottom_extra = max(BOTTOM_PAD_PX, int(rh * BOTTOM_PAD_FRAC))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + rw + pad)
        y2 = min(img_h, y + rh + pad + bottom_extra)
        regions.append((x1, y1, x2, y2))
    return regions


def _split_wide_by_gaps(image_bgr: np.ndarray, box: Box) -> List[Box]:
    """Split a wide strip into single cards using low white-fraction columns (felt gaps).

    Re-splits leftovers that are still wider than a single card (e.g. turn/river
    when a chip bridges two faces and the first pass only finds some seams).
    """
    return _split_wide_by_gaps_once(image_bgr, box, depth=0)


def _split_wide_by_gaps_once(
    image_bgr: np.ndarray, box: Box, *, depth: int
) -> List[Box]:
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return [box]
    ch, cw = crop.shape[:2]
    if cw < 20 or ch < 20:
        return [box]

    ratio = cw / float(ch)
    if ratio <= MAX_SINGLE_CARD_RATIO:
        return [box]

    b, g, r = cv2.split(crop)
    bright = np.minimum(np.minimum(r, g), b)
    col_white = np.mean(bright >= CARD_WHITE_MIN, axis=0)
    # Light smooth — heavy blur erases thin felt gaps between tight cards.
    k = max(3, (cw // 55) | 1)
    col_s = cv2.GaussianBlur(col_white.reshape(1, -1), (k, 1), 0).flatten()

    est_w = max(20.0, 0.62 * ch)
    n_est = max(2, int(round(cw / est_w)))
    # Prefer counting valleys when they are clear.
    margin = max(3, int(est_w * 0.20))
    valleys: List[Tuple[int, float]] = []
    for j in range(margin, cw - margin):
        if col_s[j] > 0.40:
            continue
        if col_s[j] <= col_s[j - 1] and col_s[j] <= col_s[j + 1]:
            valleys.append((j, float(col_s[j])))
    valleys.sort(key=lambda t: t[1])

    splits: List[int] = []
    min_sep = est_w * 0.45
    for j, _ in valleys:
        if any(abs(j - s) < min_sep for s in splits):
            continue
        splits.append(j)
        if len(splits) >= max(n_est - 1, 8):
            break
    splits.sort()

    if not splits:
        splits = [int((i + 1) * cw / n_est) for i in range(n_est - 1)]

    bounds = [0] + splits + [cw]
    pieces: List[Box] = []
    min_piece = max(12, int(est_w * 0.30))
    for i in range(len(bounds) - 1):
        a, b_ = bounds[i], bounds[i + 1]
        if b_ - a < min_piece:
            continue
        pieces.append((x1 + a, y1, x1 + b_, y2))

    if not pieces:
        return [box]

    # Recurse on any leftover multi-card strips (chip under Kh often merges Kh+9s).
    if depth >= 3:
        return pieces
    out: List[Box] = []
    for piece in pieces:
        pw = piece[2] - piece[0]
        ph = max(1, piece[3] - piece[1])
        if pw / float(ph) > MAX_SINGLE_CARD_RATIO:
            out.extend(_split_wide_by_gaps_once(image_bgr, piece, depth=depth + 1))
        else:
            out.append(piece)
    return out if out else [box]


def _extend_bottom(box: Box, img_height: int) -> Box:
    x1, y1, x2, y2 = box
    bh = y2 - y1
    extra = max(BOTTOM_PAD_PX, int(bh * BOTTOM_PAD_FRAC))
    return (x1, y1, x2, min(img_height, y2 + extra))


def _merge_duplicates(boxes: Sequence[Box], contain_ratio: float = 0.7) -> List[Box]:
    """Drop a box only if it is mostly contained inside another."""
    if not boxes:
        return []
    ordered = sorted(
        boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
    )
    kept: List[Box] = []
    for b in ordered:
        bx1, by1, bx2, by2 = b
        b_area = (bx2 - bx1) * (by2 - by1)
        duplicate = False
        for kx1, ky1, kx2, ky2 in kept:
            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if b_area > 0 and inter / b_area >= contain_ratio:
                duplicate = True
                break
        if not duplicate:
            kept.append(b)
    return kept
