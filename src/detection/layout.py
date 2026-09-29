"""Table layout helpers for card blob boxes.

Takes ``(x1, y1, x2, y2)`` boxes from ``find_card_blobs`` and partitions them
into community vs hole regions, picks the hero pair, and crops for classification.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

Box = Tuple[int, int, int, int]
ImageShape = Union[Tuple[int, ...], List[int]]

# Match blobs.py / crop_cards_from_image white-card threshold.
CARD_WHITE_MIN = 140

# Y-gap between box centers (as a fraction of median card height) that starts a new band.
_Y_CLUSTER_GAP_FRAC = 0.75
_Y_MIN_GAP_IMG_FRAC = 0.06
_UPPER_BAND_FRAC = 0.55

# Community row is near mid-felt; hero holes sit at bottom or side seats.
_COMMUNITY_Y_LO = 0.15
_COMMUNITY_Y_HI = 0.58
_HOLE_Y_LO = 0.58
# Center board strip (fractional) — extreme side seats are hole candidates.
_BOARD_X_LO = 0.25
_BOARD_X_HI = 0.75
_MIN_FACE_UP = 0.35
# Avatar-covered hole faces often sit ~0.05–0.28 full-box white.
_MIN_FACE_UP_HOLE_SEAT = 0.08
_MIN_FACE_UP_AVATAR = 0.04
_MAX_COMMUNITY = 5
_MAX_HOLES = 2


def filter_card_like_boxes(
    boxes: Sequence[Box],
    image_bgr: Optional[np.ndarray],
    *,
    min_face_up: float = _MIN_FACE_UP,
    min_aspect: float = 0.42,
    max_aspect: float = 0.82,
) -> List[Box]:
    """Drop UI chrome: require portrait aspect, face-up white, table-scale size.

    Hole-seat boxes (bottom band / side seats) use a lower white threshold and
    UL-index brightness so avatar overlays do not erase hero cards.
    """
    if not boxes:
        return []
    img_h = img_w = None
    if image_bgr is not None and image_bgr.size > 0:
        img_h, img_w = image_bgr.shape[:2]

    scored: List[Tuple[Box, float, float]] = []
    for box in boxes:
        b = _as_box(box)
        bw = b[2] - b[0]
        bh = max(1, b[3] - b[1])
        aspect = bw / float(bh)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if img_h and not (0.05 * img_h <= bh <= 0.20 * img_h):
            continue
        if img_w and not (0.025 * img_w <= bw <= 0.12 * img_w):
            continue
        if image_bgr is None:
            white = 0.6
            need = min_face_up
        else:
            white = _face_up_score_robust(image_bgr, b)
            need = min_face_up
            if _is_hole_seat_zone(b, img_h, img_w):
                need = min(_MIN_FACE_UP_HOLE_SEAT, min_face_up)
                cy_n = _center_y(b) / float(img_h)
                # Bottom face-down backs are dim; real south holes stay brighter.
                if cy_n >= _HOLE_Y_LO:
                    need = max(need, 0.16)
                else:
                    crop = image_bgr[b[1] : b[3], b[0] : b[2]]
                    if crop.size and avatar_contaminated(crop):
                        need = min(need, _MIN_FACE_UP_AVATAR)
        if white < need:
            continue
        scored.append((b, white, float(bw * bh)))
    if not scored:
        return []
    # Keep the dominant size cluster (real cards), drop crumb outliers.
    areas = np.array([a for _, _, a in scored], dtype=np.float64)
    med = float(np.median(areas))
    lo, hi = med * 0.45, med * 2.2
    kept = [b for b, _, a in scored if lo <= a <= hi]
    return _sort_lr(kept)


def select_community_row(
    boxes: Sequence[Box],
    image_shape: Optional[ImageShape],
    *,
    max_cards: int = _MAX_COMMUNITY,
) -> List[Box]:
    """Pick one horizontal board row (≤5) nearest mid-felt with best packing.

    Always chooses a *centered* contiguous pack of 3–5 cards so side-seat hole
    pairs that share the board's Y band are not swallowed into community.
    """
    if not boxes:
        return []
    img_h = int(image_shape[0]) if image_shape is not None and len(image_shape) >= 1 else None
    img_w = int(image_shape[1]) if image_shape is not None and len(image_shape) >= 2 else None

    cands = [_as_box(b) for b in boxes]
    if img_h:
        mid = [
            b
            for b in cands
            if _COMMUNITY_Y_LO <= (_center_y(b) / img_h) <= _COMMUNITY_Y_HI
        ]
        if mid:
            cands = mid
    if not cands:
        return []

    heights = [max(1, b[3] - b[1]) for b in cands]
    med_h = float(np.median(heights))
    y_tol = max(med_h * 0.45, 12.0)

    # Build Y-clusters; score each.
    ordered = sorted(cands, key=_center_y)
    clusters: List[List[Box]] = []
    cur = [ordered[0]]
    for b in ordered[1:]:
        if abs(_center_y(b) - _center_y(cur[-1])) <= y_tol:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)

    target_y = (img_h * 0.40) if img_h else None
    target_x = (img_w * 0.50) if img_w else None
    best_row: List[Box] = []
    best_score = -1e18
    for cl in clusters:
        row = _best_centered_pack(_sort_lr(cl), max_cards, img_w)
        if not row:
            continue
        n = len(row)
        # Board is flop/turn/river (3–5). Never treat a 1–2 card hole pair as board.
        if n < 3:
            continue
        n_score = 1.0 if n <= 5 else 0.4
        cy = float(np.mean([_center_y(b) for b in row]))
        cx = float(np.mean([_center_x(b) for b in row]))
        y_score = 1.0 - abs(cy - target_y) / img_h if target_y else 0.5
        x_score = 1.0 - abs(cx - target_x) / (img_w * 0.5) if target_x else 0.5
        # Reject packs that sit clearly in a side seat.
        if img_w and not (_BOARD_X_LO <= (cx / img_w) <= _BOARD_X_HI):
            continue
        area = float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in row]))
        score = 2.0 * n_score + 1.2 * max(0.0, y_score) + 1.0 * max(0.0, x_score)
        score += 0.00001 * area
        if score > best_score:
            best_score = score
            best_row = row
    return best_row[:max_cards]


def split_community_and_holes(
    boxes: Sequence[Box],
    image_shape: Optional[ImageShape],
) -> Tuple[List[Box], List[Box]]:
    """Split card boxes into community (board row) and holes.

    Community is only a centered mid-felt row of 3–5 cards. Side-seat and
    bottom-band cards stay hole candidates even when the board already has 3+.
    """
    if not boxes:
        return [], []

    all_boxes = [_as_box(b) for b in boxes]
    community = select_community_row(all_boxes, image_shape)
    img_h = int(image_shape[0]) if image_shape is not None and len(image_shape) >= 1 else None
    img_w = int(image_shape[1]) if image_shape is not None and len(image_shape) >= 2 else None
    # Peel lateral seat cards that still landed in the board pack.
    if img_h and img_w and community:
        kept_c = [b for b in community if not _is_hole_seat_zone(b, img_h, img_w)]
        if len(kept_c) >= 3:
            community = kept_c
        elif len(kept_c) < 3 and any(
            _is_hole_seat_zone(b, img_h, img_w) for b in community
        ):
            # Mis-packed side seats as "board" — treat all as hole candidates.
            community = []
    comm_set = set(community)
    holes = _sort_lr([b for b in all_boxes if b not in comm_set])
    return community, holes


def select_hero_hole_pair(
    hole_boxes: Sequence[Box],
    image_bgr: Optional[np.ndarray] = None,
) -> List[Box]:
    """Pick the hero's two hole cards from candidate hole boxes.

    Prefers face-up (bright / white, UL-robust) cards when ``image_bgr`` is
    given, then a nearby pair at a side/bottom hole seat. Returns at most two
    boxes sorted left-to-right.
    """
    if not hole_boxes:
        return []

    boxes = [_as_box(b) for b in hole_boxes]
    img_h = img_w = None
    if image_bgr is not None and image_bgr.size > 0:
        img_h, img_w = image_bgr.shape[:2]

    def _white(box: Box) -> float:
        if image_bgr is None:
            return 0.5
        return _face_up_score_robust(image_bgr, box)

    def _keep(box: Box) -> bool:
        if image_bgr is None:
            return True
        white = _white(box)
        need = (
            _MIN_FACE_UP_HOLE_SEAT
            if _is_hole_seat_zone(box, img_h, img_w)
            else _MIN_FACE_UP
        )
        if image_bgr is not None and img_h and img_w:
            cy_n = _center_y(box) / float(img_h)
            if _is_hole_seat_zone(box, img_h, img_w) and cy_n >= _HOLE_Y_LO:
                need = max(need, 0.16)
            else:
                ch, cw = image_bgr.shape[:2]
                x1, y1, x2, y2 = box
                crop = image_bgr[max(0, y1) : min(ch, y2), max(0, x1) : min(cw, x2)]
                if crop.size and avatar_contaminated(crop):
                    need = min(need, _MIN_FACE_UP_AVATAR)
        return white >= need

    boxes = [b for b in boxes if _keep(b)]
    if not boxes:
        return []
    if len(boxes) <= 2:
        return _sort_lr(boxes)[:_MAX_HOLES]

    # Score individual boxes, then prefer a spatially close face-up pair.
    scored: List[Tuple[float, Box]] = []
    for box in boxes:
        white = _white(box)
        cy = _center_y(box)
        cx = _center_x(box)
        cy_n = (cy / img_h) if img_h else 0.5
        cx_n = (cx / img_w) if img_w else 0.5
        if cy_n >= _HOLE_Y_LO:
            seat = 0.55  # bottom — de-prioritize vs mid-right face-up
        elif cx_n >= 0.68:
            seat = 1.0  # mid-right BTN/CO (common Stake hero)
        elif cx_n <= 0.32:
            seat = 0.80
        else:
            seat = 0.25
        # Avatar overlay on the face is a strong hero-hole signal.
        avatar_bonus = 0.0
        if image_bgr is not None:
            ch, cw = image_bgr.shape[:2]
            x1, y1, x2, y2 = box
            crop = image_bgr[max(0, y1) : min(ch, y2), max(0, x1) : min(cw, x2)]
            if crop.size and avatar_contaminated(crop):
                avatar_bonus = 0.20
        score = 0.58 * white + 0.28 * seat + 0.10 * cy_n + avatar_bonus
        scored.append((score, box))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Prefer a mid-right / bottom face-up pair that sits together.
    if img_w and img_h and len(scored) >= 2:
        def _dist(a: Box, b: Box) -> float:
            dx = (_center_x(a) - _center_x(b)) / img_w
            dy = (_center_y(a) - _center_y(b)) / img_h
            return (dx * dx + dy * dy) ** 0.5

        # First: best-scoring mid-right pair (Stake BTN/CO hero).
        right = [
            (s, b)
            for s, b in scored
            if (_center_x(b) / img_w) >= 0.68
        ]
        best_pair: Optional[List[Box]] = None
        best_pair_score = -1e18
        for i in range(len(right)):
            for j in range(i + 1, len(right)):
                sa, a = right[i]
                sb, b = right[j]
                if _dist(a, b) > 0.12:
                    continue
                pair_score = sa + sb
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = [a, b]
        if best_pair is not None:
            return _sort_lr(best_pair)

        best = scored[0][1]
        rest = [b for _, b in scored[1:]]
        neighbor = min(rest, key=lambda b: _dist(best, b))
        if _dist(best, neighbor) <= 0.14:
            return _sort_lr([best, neighbor])
    # Fallback: top two by score.
    chosen = [b for _, b in scored[:_MAX_HOLES]]
    return _sort_lr(chosen)


def crops_from_boxes(image_bgr: np.ndarray, boxes: Sequence[Box]) -> List[np.ndarray]:
    """Crop ``image_bgr`` at each box; skips empty / out-of-bounds results."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    out: List[np.ndarray] = []
    for box in boxes:
        x1, y1, x2, y2 = _as_box(box)
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(image_bgr[y1:y2, x1:x2].copy())
    return out


def ul_crop(card_bgr: np.ndarray, frac: float = 0.45) -> np.ndarray:
    """Upper-left index region of a card crop (rank + suit pip on stacked holes)."""
    if card_bgr is None or card_bgr.size == 0:
        return card_bgr
    f = float(frac)
    if f <= 0:
        raise ValueError("frac must be > 0")
    f = min(f, 1.0)
    ch, cw = card_bgr.shape[:2]
    th = max(1, int(round(ch * f)))
    tw = max(1, int(round(cw * f)))
    return card_bgr[:th, :tw].copy()


def hole_index_crop(
    card_bgr: np.ndarray,
    *,
    frac_h: float = 0.38,
    frac_w: float = 0.48,
) -> np.ndarray:
    """Tight UL index for hole cards (avatar often covers center pips)."""
    if card_bgr is None or card_bgr.size == 0:
        return card_bgr
    ch, cw = card_bgr.shape[:2]
    th = max(1, int(round(ch * min(max(frac_h, 0.15), 1.0))))
    tw = max(1, int(round(cw * min(max(frac_w, 0.15), 1.0))))
    return card_bgr[:th, :tw].copy()


def avatar_contaminated(card_bgr: np.ndarray) -> bool:
    """True when a circular avatar / red UI sits on the card face.

    Hero seats overlay a round portrait on the hole pair. The center pips and
    lower face go dark / skin / saturated red while the UL index stays white.
    """
    if card_bgr is None or card_bgr.size == 0 or card_bgr.ndim != 3:
        return False
    ch, cw = card_bgr.shape[:2]
    if ch < 24 or cw < 18:
        return False

    b, g, r = card_bgr[:, :, 0], card_bgr[:, :, 1], card_bgr[:, :, 2]
    bright = np.minimum(np.minimum(r, g), b)
    mid_y0, mid_y1 = int(ch * 0.28), int(ch * 0.82)
    mid_x0, mid_x1 = int(cw * 0.18), int(cw * 0.92)
    center = bright[mid_y0:mid_y1, mid_x0:mid_x1]
    upper = bright[: max(1, int(ch * 0.38)), : max(1, int(cw * 0.55))]
    if center.size == 0 or upper.size == 0:
        return False
    white_c = float(np.mean(center >= CARD_WHITE_MIN))
    white_u = float(np.mean(upper >= CARD_WHITE_MIN))

    center_bgr = card_bgr[mid_y0:mid_y1, mid_x0:mid_x1]
    cb, cg, cr = center_bgr[:, :, 0], center_bgr[:, :, 1], center_bgr[:, :, 2]
    red_frac = float(
        np.mean(
            (cr.astype(np.int16) - cg > 35)
            & (cr.astype(np.int16) - cb > 35)
            & (cr > 110)
        )
    )

    # Hough circle covering a large share of the crop (avatar frame).
    circle_hit = False
    try:
        import cv2

        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
        min_r = max(8, int(0.18 * min(ch, cw)))
        max_r = max(min_r + 1, int(0.62 * min(ch, cw)))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12, min(ch, cw) // 3),
            param1=70,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for cx, cy, rad in circles[0]:
                # Avatar sits mid-card, not in the UL index.
                if cy > ch * 0.22 and rad >= 0.20 * min(ch, cw):
                    circle_hit = True
                    break
    except Exception:
        circle_hit = False

    if circle_hit and white_c < 0.45:
        return True
    if white_u >= 0.04 and white_c <= white_u * 0.55 and red_frac >= 0.04:
        return True
    if white_c < 0.12 and red_frac >= 0.08:
        return True
    return False


def mask_circular_avatar(
    card_bgr: np.ndarray,
    *,
    fill_bgr: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Paint detected avatar circles with card-white (or ``fill_bgr``)."""
    if card_bgr is None or card_bgr.size == 0:
        return card_bgr
    out = card_bgr.copy()
    ch, cw = out.shape[:2]
    try:
        import cv2
    except ImportError:
        return out

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    min_r = max(8, int(0.18 * min(ch, cw)))
    max_r = max(min_r + 1, int(0.65 * min(ch, cw)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min(ch, cw) // 3),
        param1=70,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return out

    if fill_bgr is None:
        ul = out[: max(1, ch // 4), : max(1, cw // 4)]
        fill_bgr = tuple(int(x) for x in np.median(ul.reshape(-1, 3), axis=0))
    for cx, cy, rad in np.round(circles[0]).astype(int):
        if cy < ch * 0.18:
            continue
        cv2.circle(out, (int(cx), int(cy)), int(rad), fill_bgr, thickness=-1)
    return out


def _is_hole_seat_zone(
    box: Box,
    img_h: Optional[int],
    img_w: Optional[int],
) -> bool:
    """True for bottom hero band or far left/right seat strips (not board)."""
    if not img_h or not img_w:
        return False
    cx = _center_x(box) / float(img_w)
    cy = _center_y(box) / float(img_h)
    if cy >= _HOLE_Y_LO:
        return True
    # Far side seats that can share the board's Y band (BTN/CO mid-right).
    if cy <= _COMMUNITY_Y_HI and (cx <= 0.24 or cx >= 0.68):
        return True
    return False


def _face_up_score_robust(image_bgr: np.ndarray, box: Box) -> float:
    """Full-box white, or UL-index white when an avatar darkens a white face.

    Does not promote dark face-down backs or light-blue UI chrome (Call buttons)
    that only have a scrap of bright pixels.
    """
    full = _face_up_score(image_bgr, box)
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return full
    crop = image_bgr[y1:y2, x1:x2]
    if crop.ndim == 3 and crop.shape[2] >= 3:
        b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        b_mean = float(np.mean(b))
        r_mean = float(np.mean(r))
        # Face-down backs / blue Call chrome: blue-dominant, not a white face.
        if b_mean > r_mean + 25 and full < 0.22:
            return full * 0.5
        if b_mean > r_mean + 40 and full < 0.35:
            return min(full, 0.10)
    bh = y2 - y1
    bw = x2 - x1
    uy2 = y1 + max(1, int(round(bh * 0.40)))
    ux2 = x1 + max(1, int(round(bw * 0.55)))
    ul = _face_up_score(image_bgr, (x1, y1, ux2, uy2))
    # Only boost with UL when the card already looks partly white (avatar case).
    if full >= 0.12:
        return max(full, ul)
    return full


def _best_centered_pack(
    row: Sequence[Box],
    max_cards: int,
    img_w: Optional[int],
) -> List[Box]:
    """Best contiguous L→R pack of 3..max_cards nearest horizontal center.

    Unlike taking the whole cluster when ``len <= max_cards``, this leaves
    side-seat cards out of the board when a tighter centered flop exists.
    """
    row = _sort_lr(row)
    if len(row) < 3:
        return []
    target = (img_w * 0.5) if img_w else None
    best: List[Box] = []
    best_score = -1e18
    upper = min(max_cards, len(row))
    for n in range(3, upper + 1):
        for i in range(0, len(row) - n + 1):
            pack = list(row[i : i + n])
            cx = float(np.mean([_center_x(b) for b in pack]))
            span = pack[-1][2] - pack[0][0]
            if target is not None:
                x_score = 1.0 - abs(cx - target) / max(1.0, img_w * 0.5)
            else:
                x_score = 0.5
            # Prefer more cards when equally centered (river > flop), else center.
            score = 2.0 * max(0.0, x_score) + 0.35 * n - 0.0005 * span
            if score > best_score:
                best_score = score
                best = pack
    return best


def _tightest_x_pack(row: Sequence[Box], max_cards: int, img_w: Optional[int]) -> List[Box]:
    """Among L→R boxes, take ``max_cards`` with smallest x-span (prefer center)."""
    row = _sort_lr(row)
    if len(row) <= max_cards:
        return list(row)
    best: List[Box] = list(row[:max_cards])
    best_span = best[-1][2] - best[0][0]
    best_center_dist = 0.0
    target = (img_w * 0.5) if img_w else None
    for i in range(0, len(row) - max_cards + 1):
        pack = list(row[i : i + max_cards])
        span = pack[-1][2] - pack[0][0]
        if target is not None:
            mid = (_center_x(pack[0]) + _center_x(pack[-1])) * 0.5
            cdist = abs(mid - target)
        else:
            cdist = 0.0
        if span < best_span - 1e-6 or (
            abs(span - best_span) < 1e-6 and cdist < best_center_dist
        ):
            best = pack
            best_span = span
            best_center_dist = cdist
    return best


def _as_box(box: Sequence[int]) -> Box:
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    return (x1, y1, x2, y2)


def _center_x(box: Box) -> float:
    return (box[0] + box[2]) * 0.5


def _center_y(box: Box) -> float:
    return (box[1] + box[3]) * 0.5


def _sort_lr(boxes: Sequence[Box]) -> List[Box]:
    return sorted(boxes, key=lambda b: (_center_x(b), _center_y(b)))


def _face_up_score(image_bgr: Optional[np.ndarray], box: Box) -> float:
    """Fraction of bright (white-card) pixels in the box; 0 if empty."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image_bgr[y1:y2, x1:x2]
    if crop.ndim != 3 or crop.shape[2] < 3:
        gray = crop if crop.ndim == 2 else crop[:, :, 0]
        return float(np.mean(gray >= CARD_WHITE_MIN))
    b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    bright = np.minimum(np.minimum(r, g), b)
    return float(np.mean(bright >= CARD_WHITE_MIN))
