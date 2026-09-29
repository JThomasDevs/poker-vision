"""
Interactive tool: load a screenshot, auto-detect card-like regions (edge/contour),
show each region on the image, then a popup asks for card name (e.g. Ah) or 'skip'.
Image window stays open; no terminal input.

Usage:
  python -m src.training.crop_cards_from_image path/to/screenshot.png

If auto-detect finds no/few regions, falls back to click mode (click 2 corners per card).
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.training.card_names import CARD_NAMES


def valid_card_name(s: str) -> bool:
    s = (s or "").strip().upper()
    if len(s) == 2:
        return s[0] in "23456789TJQKA" and s[1] in "HDCS"
    if len(s) == 3 and s[:2] == "10":
        return s[2] in "HDCS"
    return False


def canonical_card_name(s: str) -> str:
    """Normalize to folder form: rank uppercase, suit lowercase. Accept 10 or T for ten (e.g. 10h -> Th)."""
    s = s.strip().upper()
    if len(s) == 3 and s[:2] == "10":
        return "T" + s[2].lower()
    return s[0] + s[1].lower()


# Single card is portrait: width/height ~ 0.6-0.75. Reject wider so side-by-side cards get split.
MAX_SINGLE_CARD_RATIO = 0.85
# Minimum width for one card (ratio of height); used to reject seams that fall inside a card.
MIN_CARD_WIDTH_RATIO = 0.5
# Extra pixels (and %) to extend bottom so we don't crop off card edges (contours often miss bottom).
BOTTOM_PAD_PX = 8
BOTTOM_PAD_FRAC = 0.04

# White cards on blue felt: color thresholds for boundary detection
CARD_WHITE_MIN = 140
FELT_BLUE_B_MIN = 80
FELT_B_DOMINANCE = 15
# Felt must be visibly blue, not black (card text is white->black; card edge is white->blue)
FELT_MIN_MEAN_BRIGHTNESS = 55
# Blue must dominate over R+G so gray (white+black mix) never counts as felt
FELT_BLUE_SATURATION = 20  # B >= (R+G)/2 + this
# Nudge region boundaries outward so we don't crop into the card edge
EDGE_NUDGE_PX = 2
# Straight edges on card text are not card edges: require blue/white over most of column height
MIN_EDGE_HEIGHT_RATIO = 0.5


def _is_blue_felt_bgr(b: float, g: float, r: float) -> bool:
    """True if BGR looks like blue felt."""
    return b >= FELT_BLUE_B_MIN and b >= r + FELT_B_DOMINANCE and b >= g + FELT_B_DOMINANCE


def _is_blue_felt_not_black_bgr(b: float, g: float, r: float) -> bool:
    """True if BGR is blue felt, not black/dark (card text) and not gray (white+black mix)."""
    mean_bright = (b + g + r) / 3.0
    if mean_bright < FELT_MIN_MEAN_BRIGHTNESS:
        return False
    if b < (r + g) / 2.0 + FELT_BLUE_SATURATION:
        return False
    return _is_blue_felt_bgr(b, g, r)


def _is_white_card_bgr(b: float, g: float, r: float) -> bool:
    """True if BGR looks like white/light card."""
    return min(r, g, b) >= CARD_WHITE_MIN


def _column_white_blue_fractions(crop: np.ndarray, col: int, band: int = 2) -> Tuple[float, float]:
    """Fraction of column height that is white (card) and blue (felt). Used to reject short strokes (card text)."""
    ch, cw = crop.shape[:2]
    lo, hi = max(0, col - band), min(cw, col + band + 1)
    strip = crop[:, lo:hi, :]
    n_rows = strip.shape[0]
    if n_rows == 0:
        return 0.0, 0.0
    white_count = blue_count = 0
    for i in range(n_rows):
        b, g, r = float(np.mean(strip[i, :, 0])), float(np.mean(strip[i, :, 1])), float(np.mean(strip[i, :, 2]))
        if _is_white_card_bgr(b, g, r):
            white_count += 1
        if _is_blue_felt_not_black_bgr(b, g, r):
            blue_count += 1
    return white_count / n_rows, blue_count / n_rows


def _strip_blue_fraction_height(crop: np.ndarray, x_lo: int, x_hi: int) -> float:
    """Fraction of rows in crop[:, x_lo:x_hi] that are blue (by row mean). Full-height edge check."""
    if x_hi <= x_lo or crop.size == 0:
        return 0.0
    strip = crop[:, x_lo:x_hi, :]
    n_rows = strip.shape[0]
    if n_rows == 0:
        return 0.0
    blue_count = 0
    for i in range(n_rows):
        b, g, r = float(np.mean(strip[i, :, 0])), float(np.mean(strip[i, :, 1])), float(np.mean(strip[i, :, 2]))
        if _is_blue_felt_not_black_bgr(b, g, r):
            blue_count += 1
    return blue_count / n_rows


def _strip_white_fraction_height(crop: np.ndarray, x_lo: int, x_hi: int) -> float:
    """Fraction of rows in crop[:, x_lo:x_hi] that are white (by row mean). Full-height edge check."""
    if x_hi <= x_lo or crop.size == 0:
        return 0.0
    strip = crop[:, x_lo:x_hi, :]
    n_rows = strip.shape[0]
    if n_rows == 0:
        return 0.0
    white_count = 0
    for i in range(n_rows):
        b, g, r = float(np.mean(strip[i, :, 0])), float(np.mean(strip[i, :, 1])), float(np.mean(strip[i, :, 2]))
        if _is_white_card_bgr(b, g, r):
            white_count += 1
    return white_count / n_rows


def _strip_blue_fraction_width(crop: np.ndarray, y_lo: int, y_hi: int) -> float:
    """Fraction of columns in crop[y_lo:y_hi, :, :] that are blue (by column mean). For horizontal edges."""
    if y_hi <= y_lo or crop.size == 0:
        return 0.0
    strip = crop[y_lo:y_hi, :, :]
    n_cols = strip.shape[1]
    if n_cols == 0:
        return 0.0
    blue_count = 0
    for c in range(n_cols):
        b, g, r = float(np.mean(strip[:, c, 0])), float(np.mean(strip[:, c, 1])), float(np.mean(strip[:, c, 2]))
        if _is_blue_felt_not_black_bgr(b, g, r):
            blue_count += 1
    return blue_count / n_cols


def _region_is_mostly_blue(img: np.ndarray, box: Tuple[int, int, int, int], blue_ratio: float = 0.55) -> bool:
    """True if the region is mostly blue felt (no card). Used to drop empty regions."""
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or len(crop.shape) != 3:
        return True
    pixels = crop.reshape(-1, 3)
    blue_count = sum(
        1 for i in range(pixels.shape[0])
        if _is_blue_felt_not_black_bgr(float(pixels[i, 0]), float(pixels[i, 1]), float(pixels[i, 2]))
    )
    return (blue_count / pixels.shape[0]) >= blue_ratio


def _filter_candidates_by_min_segment_width(
    strip_x1: int, strip_x2: int, candidates: List[int], min_w: int
) -> List[int]:
    """Keep only candidates that would not create a segment narrower than min_w (to the left or right)."""
    if not candidates or min_w <= 0:
        return candidates
    cand = sorted(set(c for c in candidates if strip_x1 < c < strip_x2))
    out = []
    for i, c in enumerate(cand):
        left_w = c - (strip_x1 if i == 0 else cand[i - 1])
        right_w = (strip_x2 if i == len(cand) - 1 else cand[i + 1]) - c
        if left_w >= min_w and right_w >= min_w:
            out.append(c)
    return out


def find_card_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find rectangular regions that look like cards. Uses white-on-blue to focus edges on card boundaries."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        bright = np.minimum(np.minimum(r, g), b)
        card_like = (bright >= CARD_WHITE_MIN) | ((blurred > 100) & (b < np.maximum(r, g) + FELT_B_DOMINANCE))
        card_mask = np.uint8(np.where(card_like, 255, 0))
        edges = cv2.bitwise_and(edges, cv2.dilate(card_mask, np.ones((3, 3))))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) / 400
    max_area = (w * h) / 2
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw < 15 or rh < 25:
            continue
        ratio = rw / rh
        if ratio < 0.25:
            continue
        pad = 2
        bottom_extra = max(BOTTOM_PAD_PX, int(rh * BOTTOM_PAD_FRAC))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + rw + pad)
        y2 = min(h, y + rh + pad + bottom_extra)
        # If wider than a single card, split by vertical seams (thin lines between cards)
        if ratio > MAX_SINGLE_CARD_RATIO:
            sub = _split_wide_region_by_seams(img, (x1, y1, x2, y2))
            regions.extend(sub)
        else:
            regions.append((x1, y1, x2, y2))

    regions = _merge_duplicates_only(regions)
    # Ensure every region gets bottom padding so we don't crop off card bottom
    regions = [_extend_bottom(r, h) for r in regions]
    regions.sort(key=lambda r: (r[0] + r[2]) // 2)
    return regions


def _extend_bottom(box: Tuple[int, int, int, int], img_height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bh = y2 - y1
    extra = max(BOTTOM_PAD_PX, int(bh * BOTTOM_PAD_FRAC))
    y2_new = min(img_height, y2 + extra)
    return (x1, y1, x2, y2_new)


def _split_wide_region_by_seams(img: np.ndarray, box: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """Split a wide box into single-card boxes by finding vertical seams (blue felt gaps between white cards)."""
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return [box]
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = np.float32(gray)
    col_mean = np.mean(gray, axis=0).astype(np.float32)
    col_std = np.std(gray, axis=0).astype(np.float32) + 1e-6
    band = 2
    col_is_blue = np.zeros(cw, dtype=bool)
    col_is_white = np.zeros(cw, dtype=bool)
    for j in range(cw):
        fw, fb = _column_white_blue_fractions(crop, j, band)
        col_is_blue[j] = fb >= MIN_EDGE_HEIGHT_RATIO
        col_is_white[j] = fw >= MIN_EDGE_HEIGHT_RATIO
    side = 8
    dark_boundary = np.zeros(cw, dtype=np.float32)
    for j in range(side, cw - side):
        center = np.mean(col_mean[max(0, j - band) : j + band + 1])
        left = np.mean(col_mean[max(0, j - side - band) : j - side + band + 1])
        right = np.mean(col_mean[j + side - band : min(cw, j + side + band + 1)])
        diff = (left + right) / 2.0 - center
        dark_boundary[j] = np.clip(diff / 80.0, 0, 1)
    gap_light = col_mean / 255.0 - 0.3 * (col_std / 80.0)
    gap_light = np.clip(gap_light, 0, 1)
    gap_score = np.maximum(gap_light, dark_boundary)
    # Vertical edge strength (gradient at boundary)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)
    col_strength = np.sum(sobelx, axis=0).astype(np.float32)
    combined = col_strength * (0.5 + 0.5 * gap_score)
    k = min(9, cw // 4) | 1
    combined = cv2.GaussianBlur(combined.reshape(1, -1), (k, 1), 0).flatten()
    margin = max(5, cw // 25)
    peaks = []
    for j in range(margin, cw - margin):
        if combined[j] <= 0:
            continue
        is_peak = True
        for d in range(1, margin + 1):
            if j - d >= 0 and combined[j] <= combined[j - d]:
                is_peak = False
                break
            if j + d < cw and combined[j] <= combined[j + d]:
                is_peak = False
                break
        if is_peak:
            peaks.append((j, float(combined[j])))
    peaks.sort(key=lambda p: -p[1])
    est_card_w = cw / max(2, round(cw / (0.65 * ch)))
    min_gap = est_card_w * 0.85  # stricter: seams must be ~1 card width apart so we don't split inside a card
    split_cols = []
    for j, _ in peaks:
        if any(abs(j - s) < min_gap for s in split_cols):
            continue
        if gap_score[j] < 0.25:
            continue
        if not col_is_blue[j]:
            continue
        # Right edge of region = white->blue (end of card), not blue->white (start of next). Require white to the left of j.
        if j > 0 and not col_is_white[j - 1]:
            continue
        split_cols.append(j)
        if len(split_cols) >= 8:
            break
    split_cols.sort()
    if not split_cols:
        n_est = max(2, round(cw / (0.65 * ch)))
        scanned = _find_n_regions_by_boundary_scan(img, (x1, y1, x2, y2), n_est)
        if len(scanned) >= 2:
            return scanned
        n = n_est
        split_cols = [int((i + 1) * cw / (n + 1)) for i in range(n - 1)]
    bounds = [0] + split_cols + [cw]
    out = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 15:
            continue
        out.append((x1 + a, y1, x1 + b, y2))
    return out if out else [box]


def _merge_duplicates_only(boxes: List[Tuple[int, int, int, int]], contain_ratio: float = 0.7) -> List[Tuple[int, int, int, int]]:
    """Drop a box only if it is mostly contained inside another (duplicate), not just overlapping."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        bx1, by1, bx2, by2 = b
        b_area = (bx2 - bx1) * (by2 - by1)
        is_duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k
            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            # Only treat as duplicate if most of b is inside k (b is the smaller one when we're comparing to kept)
            if b_area > 0 and inter / b_area >= contain_ratio:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(b)
    return kept


def _ask_num_cards_popup() -> Optional[int]:
    """Ask how many cards in the image. Returns 2-10 or None if cancel."""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = simpledialog.askstring(
        "How many cards?",
        "How many cards in this image? (2–10)\nCancel = auto-detect.",
        parent=root,
    )
    root.destroy()
    if result is None or not result.strip():
        return None
    try:
        n = int(result.strip())
        if 2 <= n <= 10:
            return n
    except ValueError:
        pass
    return None


def _split_image_into_strips(img: np.ndarray, num_cards: int) -> List[Tuple[int, int, int, int]]:
    """Split image into N equal vertical strips (full height). Fallback when detection finds nothing."""
    h, w = img.shape[:2]
    if num_cards < 1 or w < num_cards * 20:
        return []
    strip_w = w // num_cards
    margin = strip_w // 8
    regions = []
    for i in range(num_cards):
        x1 = i * strip_w + (margin // 2)
        x2 = (i + 1) * strip_w - (margin // 2)
        x1 = max(0, x1)
        x2 = min(w, x2)
        if x2 - x1 < 20:
            continue
        regions.append((x1, 0, x2, h))
    return regions


def _refine_card1_left_edge(img: np.ndarray, box: Tuple[int, int, int, int], search_width: int = 25) -> Tuple[int, int, int, int]:
    """Refine the left edge: blue (felt) -> white (card). Crop extends right so we see the card; require clear blue left and white right."""
    x1, y1, x2, y2 = box
    img_w = img.shape[1]
    left = max(0, x1 - search_width)
    right_crop = min(img_w, x1 + 35)
    crop = img[y1:y2, left : right_crop]
    if crop.size == 0 or len(crop.shape) != 3:
        return box
    cw = crop.shape[1]
    for j in range(cw - 6, 2, -1):
        x_lo, x_hi = max(0, j - 4), j
        if _strip_blue_fraction_height(crop, x_lo, x_hi) < MIN_EDGE_HEIGHT_RATIO:
            continue
        left_strip = crop[:, x_lo:x_hi, :].reshape(-1, 3).mean(axis=0)
        right_strip = crop[:, j : min(cw, j + 6), :].reshape(-1, 3).mean(axis=0)
        b_l, g_l, r_l = left_strip[0], left_strip[1], left_strip[2]
        b_r, g_r, r_r = right_strip[0], right_strip[1], right_strip[2]
        if (r_l + g_l + b_l) / 3.0 < 150 and _is_blue_felt_not_black_bgr(b_l, g_l, r_l) and _is_white_card_bgr(b_r, g_r, r_r):
            return (max(0, left + j - EDGE_NUDGE_PX), y1, x2, y2)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(np.float32(gray), cv2.CV_64F, 1, 0, ksize=3)
    col_strength = np.sum(np.abs(sobelx), axis=0)
    col_strength = np.convolve(col_strength, np.ones(3) / 3.0, mode="same")
    thresh = np.max(col_strength) * 0.4
    for j in range(len(col_strength) - 2, 0, -1):
        if col_strength[j] >= thresh and col_strength[j] >= col_strength[j - 1] and col_strength[j] >= col_strength[j + 1]:
            x1 = left + j
            break
    return (x1, y1, x2, y2)


def _refine_region_left_edge(
    img: np.ndarray, box: Tuple[int, int, int, int], search_width: int = 28, min_left: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """Refine the left edge of any region (card 2..N): find blue->white. min_left avoids making previous region a gap."""
    x1, y1, x2, y2 = box
    img_w = img.shape[1]
    left = max(0, x1 - search_width)
    right_crop = min(img_w, x1 + 25)
    crop = img[y1:y2, left : right_crop]
    if crop.size == 0 or len(crop.shape) != 3:
        return box
    cw = crop.shape[1]
    for j in range(cw - 6, 2, -1):
        refined_x1 = max(0, left + j - EDGE_NUDGE_PX)
        if min_left is not None and refined_x1 < min_left:
            continue
        x_lo, x_hi = max(0, j - 4), j
        if _strip_blue_fraction_height(crop, x_lo, x_hi) < MIN_EDGE_HEIGHT_RATIO:
            continue
        left_strip = crop[:, x_lo:x_hi, :].reshape(-1, 3).mean(axis=0)
        right_strip = crop[:, j : min(cw, j + 6), :].reshape(-1, 3).mean(axis=0)
        b_l, g_l, r_l = left_strip[0], left_strip[1], left_strip[2]
        b_r, g_r, r_r = right_strip[0], right_strip[1], right_strip[2]
        if (r_l + g_l + b_l) / 3.0 < 150 and _is_blue_felt_not_black_bgr(b_l, g_l, r_l) and _is_white_card_bgr(b_r, g_r, r_r):
            return (refined_x1, y1, x2, y2)
    return box


def _refine_region_top_edge(img: np.ndarray, box: Tuple[int, int, int, int], search_height: int = 35) -> Tuple[int, int, int, int]:
    """Refine the top edge: find blue (felt) -> white (card) transition when scanning top to bottom."""
    x1, y1, x2, y2 = box
    img_h = img.shape[0]
    top = max(0, y1 - search_height)
    bottom_crop = min(img_h, y1 + 20)
    crop = img[top : bottom_crop, x1:x2]
    if crop.size == 0 or len(crop.shape) != 3:
        return box
    rh, rw = crop.shape[0], crop.shape[1]
    for i in range(1, rh - 4):
        y_lo, y_hi = max(0, i - 4), i
        if _strip_blue_fraction_width(crop, y_lo, y_hi) < MIN_EDGE_HEIGHT_RATIO:
            continue
        above = crop[y_lo:y_hi, :, :].reshape(-1, 3).mean(axis=0)
        below = crop[i : min(rh, i + 5), :, :].reshape(-1, 3).mean(axis=0)
        b_a, g_a, r_a = above[0], above[1], above[2]
        b_b, g_b, r_b = below[0], below[1], below[2]
        if _is_blue_felt_not_black_bgr(b_a, g_a, r_a) and _is_white_card_bgr(b_b, g_b, r_b):
            return (x1, max(0, top + i - EDGE_NUDGE_PX), x2, y2)
    return box


def _refine_region_bottom_edge(img: np.ndarray, box: Tuple[int, int, int, int], search_height: int = 45) -> Tuple[int, int, int, int]:
    """Refine the bottom edge: find white (card) -> blue (felt) transition when scanning top to bottom."""
    x1, y1, x2, y2 = box
    img_h = img.shape[0]
    top_crop = max(0, y2 - 25)
    bottom = min(img_h, y2 + search_height)
    crop = img[top_crop : bottom, x1:x2]
    if crop.size == 0 or len(crop.shape) != 3:
        return box
    rh, rw = crop.shape[0], crop.shape[1]
    for i in range(4, rh - 2):
        y_lo, y_hi = i, min(rh, i + 5)
        if _strip_blue_fraction_width(crop, y_lo, y_hi) < MIN_EDGE_HEIGHT_RATIO:
            continue
        above = crop[max(0, i - 4) : i, :, :].reshape(-1, 3).mean(axis=0)
        below = crop[y_lo:y_hi, :, :].reshape(-1, 3).mean(axis=0)
        b_a, g_a, r_a = above[0], above[1], above[2]
        b_b, g_b, r_b = below[0], below[1], below[2]
        if _is_white_card_bgr(b_a, g_a, r_a) and _is_blue_felt_not_black_bgr(b_b, g_b, r_b):
            return (x1, y1, x2, min(img_h, top_crop + i + EDGE_NUDGE_PX))
    return box


def _refine_region1_right_edge(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    search_width: int = 40,
    search_right: int = 70,
    min_right: Optional[int] = None,
    max_right: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Refine the right edge to the first white->blue (end of card). min_right avoids too skinny; max_right avoids stealing next region."""
    x1, y1, x2, y2 = box
    img_w = img.shape[1]
    right_limit = min(img_w, x2 + search_right)
    if max_right is not None:
        right_limit = min(right_limit, max_right + 1)
    left = x1 if max_right is None else max(x1, x2 - search_width)
    crop = img[y1:y2, left : right_limit]
    if crop.size == 0 or len(crop.shape) != 3:
        return box
    cw = crop.shape[1]
    for j in range(2, cw - 3):
        refined_right = min(img_w, left + j + EDGE_NUDGE_PX)
        if min_right is not None and refined_right < min_right:
            continue
        if max_right is not None and refined_right > max_right:
            break
        x_lo, x_hi = j, min(cw, j + 4)
        if _strip_blue_fraction_height(crop, x_lo, x_hi) < MIN_EDGE_HEIGHT_RATIO:
            continue
        left_strip = crop[:, max(0, j - 4) : j, :].reshape(-1, 3).mean(axis=0)
        b_l, g_l, r_l = left_strip[0], left_strip[1], left_strip[2]
        if _is_white_card_bgr(b_l, g_l, r_l):
            return (x1, y1, refined_right, y2)
    return box


def _pick_boundaries_for_roughly_equal_regions(
    strip_x1: int, strip_x2: int, candidate_x: List[int], n: int
) -> List[int]:
    """Pick n-1 split positions from candidates so the n regions have roughly equal width. Returns sorted list of n-1 x positions."""
    if n < 2 or not candidate_x:
        return []
    candidates = sorted(set(x for x in candidate_x if strip_x1 < x < strip_x2))
    if len(candidates) < n - 1:
        return []
    total_w = strip_x2 - strip_x1
    target_w = total_w / n
    min_first_segment = max(15, int(0.5 * target_w))
    chosen: List[int] = []
    last_idx = -1
    for slot in range(n - 1):
        ideal = strip_x1 + (slot + 1) * target_w
        need_after = (n - 2) - slot
        hi = len(candidates) - need_after if need_after > 0 else len(candidates)
        lo = last_idx + 1
        if lo >= hi:
            return []
        if slot == 0:
            lo = next((i for i in range(lo, hi) if candidates[i] >= strip_x1 + min_first_segment), hi)
        if lo >= hi:
            return []
        best_idx = min(range(lo, hi), key=lambda idx: abs(candidates[idx] - ideal))
        chosen.append(candidates[best_idx])
        last_idx = best_idx
    return chosen


def _regions_from_boundaries(
    box: Tuple[int, int, int, int], split_x: List[int]
) -> List[Tuple[int, int, int, int]]:
    """Build N regions from strip box and N-1 split x positions."""
    x1, y1, x2, y2 = box
    if not split_x:
        return [(x1, y1, x2, y2)]
    bounds = [x1] + split_x + [x2]
    return [(bounds[i], y1, bounds[i + 1], y2) for i in range(len(bounds) - 1)]


def _find_n_regions_by_boundary_scan(
    img: np.ndarray, box: Tuple[int, int, int, int], n: int, *, drop_mostly_blue: bool = True
) -> List[Tuple[int, int, int, int]]:
    """Find N card regions by scanning the full strip for blue<->white transitions. Picks boundaries so regions are roughly equal size.
    When drop_mostly_blue is False (e.g. user provided N), keep all N regions so we don't drop leftmost/rightmost and undercount."""
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or len(crop.shape) != 3 or n < 1:
        return []
    ch, cw = crop.shape[:2]
    band = 2
    col_white = np.zeros(cw, dtype=bool)
    col_blue = np.zeros(cw, dtype=bool)
    for j in range(cw):
        fw, fb = _column_white_blue_fractions(crop, j, band)
        col_white[j] = fw >= MIN_EDGE_HEIGHT_RATIO
        col_blue[j] = fb >= MIN_EDGE_HEIGHT_RATIO
    # Require a run of blue/white so we don't split on in-card strokes (white->black has 0 blue to the right)
    min_run = 2
    run_len = 8
    transitions = []
    for j in range(1, cw):
        if col_white[j - 1] and not col_white[j]:
            if np.sum(col_white[max(0, j - run_len) : j]) >= min_run and np.sum(col_blue[j : min(cw, j + run_len)]) >= min_run:
                transitions.append((j, "white_to_blue"))
        elif not col_white[j - 1] and col_white[j]:
            if np.sum(col_blue[max(0, j - run_len) : j]) >= min_run and np.sum(col_white[j : min(cw, j + run_len)]) >= min_run:
                transitions.append((j, "blue_to_white"))
    if not transitions:
        if n >= 1 and (x2 - x1) >= n * 15:
            split_x = [x1 + (i + 1) * (x2 - x1) // n for i in range(n - 1)]
            return _regions_from_boundaries((x1, y1, x2, y2), split_x)
        return []
    # Candidate card right edges (white_to_blue) in image coordinates
    candidate_right_edges = [x1 + t[0] for t in transitions if t[1] == "white_to_blue"]
    min_card_w = 15
    total_w = x2 - x1
    min_segment_w = max(min_card_w, int(0.4 * total_w / n))
    candidate_right_edges = _filter_candidates_by_min_segment_width(x1, x2, candidate_right_edges, min_segment_w)
    if len(candidate_right_edges) >= n - 1:
        split_x = sorted(candidate_right_edges)[: n - 1]
        if len(split_x) == n - 1:
            out = _regions_from_boundaries((x1, y1, x2, y2), split_x)
            if all((r[2] - r[0]) >= min_card_w for r in out):
                if drop_mostly_blue:
                    out_filtered = [r for r in out if not _region_is_mostly_blue(img, r)]
                    if len(out_filtered) >= n:
                        return out_filtered
                if out:
                    return out
    # Fallback: pair blue_to_white with white_to_blue as before
    out = []
    i = 0
    if transitions[0][1] == "white_to_blue":
        i = 1
    while i + 1 < len(transitions) and len(out) < n:
        left_type, right_type = transitions[i][1], transitions[i + 1][1]
        left_x = x1 + transitions[i][0]
        right_x = x1 + transitions[i + 1][0]
        if left_type == "blue_to_white" and right_type == "white_to_blue" and (right_x - left_x) >= min_card_w:
            out.append((left_x, y1, right_x, y2))
        i += 2
    if len(out) == n:
        if drop_mostly_blue:
            out_filtered = [r for r in out if not _region_is_mostly_blue(img, r)]
            if len(out_filtered) >= n:
                out = out_filtered
        widths = [r[2] - r[0] for r in out]
        target_w = (x2 - x1) / n
        if min(widths) >= 0.5 * target_w and max(widths) <= 1.5 * target_w:
            return out
        candidate_right_edges = [r[2] for r in out[:-1]]
        candidate_right_edges = _filter_candidates_by_min_segment_width(x1, x2, candidate_right_edges, min_segment_w)
        if len(candidate_right_edges) >= n - 1:
            split_x = sorted(candidate_right_edges)[: n - 1]
            if len(split_x) == n - 1:
                out = _regions_from_boundaries((x1, y1, x2, y2), split_x)
                if drop_mostly_blue:
                    out_f = [r for r in out if not _region_is_mostly_blue(img, r)]
                    return out_f if len(out_f) >= n else out
                return out
    if drop_mostly_blue:
        out_f = [r for r in out if not _region_is_mostly_blue(img, r)]
        if len(out_f) >= n:
            return out_f
    if not out and n >= 1 and (x2 - x1) >= n * 15:
        split_x = [x1 + (i + 1) * (x2 - x1) // n for i in range(n - 1)]
        return _regions_from_boundaries((x1, y1, x2, y2), split_x)
    return out


def _tile_n_regions_from_first(img: np.ndarray, regions: List[Tuple[int, int, int, int]], n: int, img_width: int) -> List[Tuple[int, int, int, int]]:
    """Build N regions: prefer boundary scan (each card independent); fallback to tile from region 1 width."""
    if not regions or n < 1:
        return []
    regions = sorted(regions, key=lambda r: (r[0] + r[2]) // 2)
    strip = (min(r[0] for r in regions), min(r[1] for r in regions), max(r[2] for r in regions), max(r[3] for r in regions))
    scanned = _find_n_regions_by_boundary_scan(img, strip, n)
    if len(scanned) >= n:
        region1 = _refine_card1_left_edge(img, scanned[0])
        region1 = _refine_region1_right_edge(img, region1)
        strip_w = strip[2] - strip[0]
        target_w = strip_w / n
        if (region1[2] - region1[0]) < 0.6 * target_w:
            region1 = (region1[0], region1[1], scanned[0][2], region1[3])
        result = [region1] + scanned[1:n]
        result_filtered = [r for r in result if not _region_is_mostly_blue(img, r)]
        return result_filtered if len(result_filtered) >= n else result
    img_h = img.shape[0]
    full_scanned = _find_n_regions_by_boundary_scan(img, (0, 0, img_width, img_h), n)
    if len(full_scanned) >= n:
        region1 = _refine_card1_left_edge(img, full_scanned[0])
        region1 = _refine_region1_right_edge(img, region1)
        strip_w = img_width
        target_w = strip_w / n
        if (region1[2] - region1[0]) < 0.6 * target_w:
            region1 = (region1[0], region1[1], full_scanned[0][2], region1[3])
        result = [region1] + full_scanned[1:n]
        result_filtered = [r for r in result if not _region_is_mostly_blue(img, r)]
        return result_filtered if len(result_filtered) >= n else result
    first_box = regions[0]
    sub = _split_wide_region_by_seams(img, first_box)
    sub = sorted(sub, key=lambda r: r[0])
    min_width = int((first_box[3] - first_box[1]) * MIN_CARD_WIDTH_RATIO)
    region1 = sub[0]
    for i in range(1, len(sub)):
        if (region1[2] - region1[0]) >= min_width:
            break
        # Merge with next sub-region to the right (seam was inside first card)
        region1 = (region1[0], min(region1[1], sub[i][1]), sub[i][2], max(region1[3], sub[i][3]))
    region1 = _refine_card1_left_edge(img, region1)
    region1 = _refine_region1_right_edge(img, region1)
    x1, y1, x2, y2 = region1
    w = x2 - x1
    if w < 15:
        return regions[:n] if len(regions) >= n else regions
    out = []
    for i in range(n):
        start_x = x1 + i * w
        end_x = min(x1 + (i + 1) * w, img_width)
        if start_x >= img_width:
            break
        out.append((start_x, y1, end_x, y2))
    return out


def _ask_card_name_popup(region_index: int, total: int) -> Optional[str]:
    """Popup for card name; returns None if user cancels/quits."""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = simpledialog.askstring(
        "Card name",
        f"Region {region_index + 1}/{total}\nCard name (e.g. 7h, Td) or 'skip':",
        parent=root,
    )
    root.destroy()
    return result


MIN_CARD_WIDTH = 15

# Max width for a region to be considered a "gap" (blue strip between cards) and merged away
GAP_MAX_WIDTH_RATIO = 0.4
GAP_MAX_WIDTH_PX = 35


def _region_is_mostly_white(img: np.ndarray, box: Tuple[int, int, int, int], white_ratio: float = 0.4) -> bool:
    """True if the region has at least this fraction of white/card pixels."""
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or len(crop.shape) != 3:
        return False
    pixels = crop.reshape(-1, 3)
    white_count = sum(
        1 for i in range(pixels.shape[0])
        if _is_white_card_bgr(float(pixels[i, 0]), float(pixels[i, 1]), float(pixels[i, 2]))
    )
    return (white_count / pixels.shape[0]) >= white_ratio


def _merge_gap_regions(img: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Merge (1) narrow mostly-blue gaps into neighbors; (2) consecutive narrow mostly-white regions (half-cards)."""
    if len(regions) <= 1:
        return regions
    widths = [r[2] - r[0] for r in regions]
    median_w = float(np.median(widths))
    gap_threshold = min(max(median_w * GAP_MAX_WIDTH_RATIO, 15), GAP_MAX_WIDTH_PX)
    out: List[Tuple[int, int, int, int]] = []
    i = 0
    while i < len(regions):
        r = regions[i]
        w = r[2] - r[0]
        is_gap = w < gap_threshold and _region_is_mostly_blue(img, r, blue_ratio=0.5)
        if is_gap and out:
            out[-1] = (out[-1][0], out[-1][1], r[2], out[-1][3])
            i += 1
            continue
        if is_gap and i + 1 < len(regions):
            next_r = regions[i + 1]
            out.append((r[0], next_r[1], next_r[2], next_r[3]))
            i += 2
            continue
        if is_gap:
            i += 1
            continue
        out.append(r)
        i += 1
    if len(out) <= 1:
        return out if out else regions
    widths = [r[2] - r[0] for r in out]
    median_w = float(np.median(widths))
    narrow_threshold = median_w * 0.55
    merged: List[Tuple[int, int, int, int]] = []
    j = 0
    while j < len(out):
        r = out[j]
        w = r[2] - r[0]
        if w < narrow_threshold and _region_is_mostly_white(img, r) and j + 1 < len(out):
            next_r = out[j + 1]
            nw = next_r[2] - next_r[0]
            if nw < narrow_threshold and _region_is_mostly_white(img, next_r):
                merged.append((r[0], min(r[1], next_r[1]), next_r[2], max(r[3], next_r[3])))
                j += 2
                continue
        merged.append(r)
        j += 1
    return merged if merged else out


def _regions_to_x_positions(regions: List[Tuple[int, int, int, int]]) -> Tuple[List[int], int, int]:
    """From left-to-right regions, get (x_positions, y1, y2). x_positions has len N+1 for N regions."""
    if not regions:
        return [], 0, 0
    xs = [regions[0][0]]
    for r in regions:
        xs.append(r[2])
    y1, y2 = regions[0][1], regions[0][3]
    return xs, y1, y2


def _x_positions_to_regions(x_positions: List[int], y1: int, y2: int) -> List[Tuple[int, int, int, int]]:
    """Build regions from vertical split positions."""
    return [(x_positions[i], y1, x_positions[i + 1], y2) for i in range(len(x_positions) - 1)]


def adjust_regions_interactive(img: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Optional[List[Tuple[int, int, int, int]]]:
    """Draw splits; user drags vertical lines to adjust. Press D/Enter when done, Q to cancel."""
    if not regions:
        return None
    h, w = img.shape[:2]
    x_positions, y1, y2 = _regions_to_x_positions(regions)
    n = len(x_positions) - 1
    dragging: Optional[int] = None
    line_hit_radius = 12
    win = "Adjust splits: drag lines, D/Enter=done, Q=cancel"

    def on_mouse(event, mx, my, _flags, _param):
        nonlocal dragging, x_positions
        if event == cv2.EVENT_LBUTTONDOWN:
            best = None
            best_d = line_hit_radius + 1
            for i in range(len(x_positions)):
                d = abs(mx - x_positions[i])
                if d < best_d:
                    best_d = d
                    best = i
            if best is not None:
                dragging = best
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = None
        elif event == cv2.EVENT_MOUSEMOVE and dragging is not None:
            lo = x_positions[dragging - 1] + MIN_CARD_WIDTH if dragging > 0 else 0
            hi = x_positions[dragging + 1] - MIN_CARD_WIDTH if dragging < len(x_positions) - 1 else w - 1
            x_positions[dragging] = max(lo, min(hi, mx))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Drag lines to adjust. A=add region (right). D/Enter=done, Q=cancel.")

    while True:
        n = len(x_positions) - 1
        display = img.copy()
        for i, x in enumerate(x_positions):
            cv2.line(display, (x, y1), (x, y2), (0, 255, 0), 2)
        for i in range(n):
            cx = (x_positions[i] + x_positions[i + 1]) // 2
            cy = (y1 + y2) // 2
            cv2.putText(display, str(i + 1), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Hint
        cv2.putText(display, "A=add region right | D/Enter=done | Q=cancel", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyWindow(win)
            return None
        if key == ord("d") or key == 13:  # Enter
            cv2.destroyWindow(win)
            return _x_positions_to_regions(x_positions, y1, y2)
        if key == ord("a") or key == ord("A"):
            # Add a region to the right: new split after the last, same width as last region
            last_w = x_positions[-1] - x_positions[-2]
            new_x = x_positions[-1] + last_w
            new_x = min(w - 1, max(x_positions[-1] + MIN_CARD_WIDTH, new_x))
            x_positions.append(new_x)

    return None


def run_auto_mode(img: np.ndarray, image_path: Path, cards_root: Path, window: str) -> int:
    """Ask how many cards, find and refine regions, preview/adjust all splits in one step, then name each."""
    num_cards = _ask_num_cards_popup()
    h, w = img.shape[:2]
    full_box = (0, 0, w, h)
    if num_cards is not None:
        regions = _find_n_regions_by_boundary_scan(img, full_box, num_cards, drop_mostly_blue=False)
        if len(regions) != num_cards:
            regions = find_card_regions(img)
            if regions:
                regions = _tile_n_regions_from_first(img, regions, num_cards, w)
            if not regions or len(regions) != num_cards:
                regions = _find_n_regions_by_boundary_scan(img, full_box, num_cards)
                if len(regions) >= num_cards:
                    regions = regions[:num_cards]
                elif len(regions) < num_cards:
                    regions = _split_image_into_strips(img, num_cards)
            if not regions:
                regions = _split_image_into_strips(img, num_cards)
    else:
        probe = _find_n_regions_by_boundary_scan(img, full_box, 10)
        if len(probe) >= 2:
            regions = probe
            n_est = len(regions)
        else:
            regions = find_card_regions(img)
            n_est = max(2, min(10, round(w / (h * 0.65))))
            if len(regions) == 1:
                r = regions[0]
                box_w, box_h = r[2] - r[0], r[3] - r[1]
                split = _find_n_regions_by_boundary_scan(img, r, n_est)
                if len(split) >= 2:
                    regions = split
                else:
                    split_full = _find_n_regions_by_boundary_scan(img, full_box, n_est)
                    if len(split_full) >= 2:
                        regions = split_full
                    elif box_w >= 1.5 * box_h:
                        sub = _split_wide_region_by_seams(img, r)
                        if len(sub) >= 2:
                            regions = sub
        if len(regions) != n_est and len(regions) >= 1 and n_est >= 1:
            y1 = min(r[1] for r in regions)
            y2 = max(r[3] for r in regions)
            strip = (0, y1, w, y2)
            x1, y1, x2, y2 = strip
            strip_w = x2 - x1
            if strip_w >= n_est * 15:
                split_x = [x1 + (i + 1) * (x2 - x1) // n_est for i in range(n_est - 1)]
                regions = _regions_from_boundaries((x1, y1, x2, y2), split_x)
    if not regions:
        print("No card-like regions found. Use click mode instead.")
        return -1  # signal fallback to click mode

    if num_cards is not None and len(regions) != num_cards and num_cards >= 1:
        h, w = img.shape[:2]
        strip = (0, 0, w, h)
        if len(regions) > 0:
            y1 = min(r[1] for r in regions)
            y2 = max(r[3] for r in regions)
            x2_union = max(r[2] for r in regions)
            strip = (0, y1, x2_union, y2)
        x1, y1, x2, y2 = strip
        strip_w = x2 - x1
        min_strip_w = num_cards * 15
        if strip_w < min_strip_w:
            x2 = min(x1 + min_strip_w, w)
            strip = (x1, y1, x2, y2)
            strip_w = x2 - x1
        if strip_w >= min_strip_w:
            split_x = [x1 + (i + 1) * (x2 - x1) // num_cards for i in range(num_cards - 1)]
            regions = _regions_from_boundaries((x1, y1, x2, y2), split_x)

    regions[0] = _refine_card1_left_edge(img, regions[0], search_width=50)
    for i in range(1, len(regions)):
        min_left = regions[i - 1][2]
        regions[i] = _refine_region_left_edge(img, regions[i], min_left=min_left)

    for i in range(len(regions)):
        min_right = regions[i][0] + MIN_CARD_WIDTH
        max_right = (regions[i + 1][0] - 1) if i + 1 < len(regions) else None
        regions[i] = _refine_region1_right_edge(img, regions[i], min_right=min_right, max_right=max_right)

    for i in range(len(regions)):
        regions[i] = _refine_region_top_edge(img, regions[i])
        regions[i] = _refine_region_bottom_edge(img, regions[i])

    regions = _merge_gap_regions(img, regions)

    if num_cards is None:
        regions = [r for r in regions if not _region_is_mostly_blue(img, r)]
    if not regions:
        print("No card-like regions found after filtering.")
        return -1

    regions = adjust_regions_interactive(img, regions)
    if regions is None:
        print("Cancelled.")
        return 0

    print(f"Found {len(regions)} region(s). Popup will ask for each. Enter card name (e.g. 7h) or 'skip'.")
    saved = 0
    for i, (x1, y1, x2, y2) in enumerate(regions):
        display = img.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"Region {i+1}/{len(regions)}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, display)
        cv2.waitKey(100)

        name = _ask_card_name_popup(i, len(regions))
        if name is None:
            break
        name = name.strip()
        if name.lower() == "skip" or not name:
            continue
        if not valid_card_name(name):
            print(f"Invalid '{name}'. Use rank+suit: 2h, Td, As (10=T).")
            continue
        crop = img[y1:y2, x1:x2].copy()
        out_dir = cards_root / canonical_card_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))) + 1
        out_path = out_dir / f"{image_path.stem}_{n:03d}.png"
        cv2.imwrite(str(out_path), crop)
        saved += 1
        print(f"Saved {out_path}")
    return saved


def run_click_mode(img: np.ndarray, image_path: Path, cards_root: Path) -> int:
    """Original click-two-corners mode; use popup for card name so window stays open."""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()

    points = []
    display = img.copy()
    saved_count = 0
    window = "Crop cards: click 2 corners per card. q=quit c=clear"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal points, display
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        if len(points) == 1:
            display = img.copy()
            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
        elif len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("Click top-left then bottom-right for each card. Popup will ask for card name.")

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            points = []
            display = img.copy()
            continue
        if len(points) < 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            points = []
            display = img.copy()
            continue

        name = simpledialog.askstring("Card name", "Card name (e.g. Ah, Td) or 'skip':", parent=root)
        if name is None:
            break
        name = name.strip()
        if name.lower() == "skip" or not name:
            points = []
            display = img.copy()
            continue
        if not valid_card_name(name):
            print(f"Invalid '{name}'. Use rank+suit: 2h, Td, As.")
            points = []
            display = img.copy()
            continue

        crop = img[y1:y2, x1:x2].copy()
        out_dir = cards_root / canonical_card_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))) + 1
        out_path = out_dir / f"{image_path.stem}_{n:03d}.png"
        cv2.imwrite(str(out_path), crop)
        saved_count += 1
        print(f"Saved {out_path}")
        points = []
        display = img.copy()

    root.destroy()
    return saved_count


def main(image_path: str) -> None:
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    cards_root = Path(__file__).resolve().parents[2] / "data" / "cards" / "by_card"
    cards_root.mkdir(parents=True, exist_ok=True)
    for name in CARD_NAMES:
        (cards_root / name).mkdir(parents=True, exist_ok=True)

    window = "Crop cards: auto-detect regions, enter card name in popup (or skip)"
    cv2.namedWindow(window)
    saved = run_auto_mode(img, image_path, cards_root, window)
    if saved == -1:
        saved = run_click_mode(img, image_path, cards_root)
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} card crops.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.crop_cards_from_image <screenshot.png>")
        sys.exit(1)
    main(sys.argv[1])
