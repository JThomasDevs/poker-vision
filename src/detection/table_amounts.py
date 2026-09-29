"""Read pot / to-call / hero stack from Stake-style table screenshots.

Amounts are in **BB** (big blinds), optionally labeled SC. ROI fractions are
tunable; OCR uses pytesseract when installed. Parsers are pure regex and work
without OCR for tests.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Common Windows install locations when `tesseract` is not on PATH.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
)
_tesseract_cmd_resolved: Optional[str] = None

# Debug OCR dump throttle (seconds between full raw dumps).
_DEBUG_OCR = os.environ.get("POKER_VISION_OCR_DEBUG", "").strip() not in ("", "0", "false")
_last_ocr_debug_at = 0.0
_OCR_DEBUG_INTERVAL_S = 3.0

# Whitelist: digits, BB/SC/Pot/Call labels — no currency symbols.
_TESS_WHITELIST = "0123456789.,:BbSsCcAaLlOoTtPpRrEeIiNnKkFfGgHhUuVvWwXxYyZz "


def _ensure_tesseract_cmd() -> None:
    """Point pytesseract at an installed binary if PATH lookup fails."""
    global _tesseract_cmd_resolved
    if _tesseract_cmd_resolved is not None:
        return
    import pytesseract

    existing = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if existing and os.path.isfile(str(existing)):
        _tesseract_cmd_resolved = str(existing)
        return
    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        _tesseract_cmd_resolved = which
        return
    for candidate in _TESSERACT_CANDIDATES:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            _tesseract_cmd_resolved = candidate
            return
    _tesseract_cmd_resolved = ""


# ---------------------------------------------------------------------------
# Tunable ROIs as (x0, y0, x1, y1) fractions of frame width / height.
# Calibrated on full Chrome window captures (tabs + bookmarks above felt).
# Stake: pot pill is FIXED above the center logo — never jackpot / stakes line.
# Hero stack sits under the face-up hole cards (often mid-right or bottom).
# ---------------------------------------------------------------------------
ROI_POT: Tuple[float, float, float, float] = (0.38, 0.28, 0.62, 0.40)
ROI_POT_ALT: Tuple[float, float, float, float] = (0.36, 0.26, 0.64, 0.42)
# Facing-bet chip / Call amount near mid-felt — NOT the "NLHE 0.01/0.02" line.
ROI_TO_CALL: Tuple[float, float, float, float] = (0.38, 0.44, 0.62, 0.54)
ROI_TO_CALL_ALT: Tuple[float, float, float, float] = (0.42, 0.46, 0.58, 0.56)
# Action bar (Check / Call / Fold) — lower-right chrome.
ROI_ACTION: Tuple[float, float, float, float] = (0.55, 0.82, 0.98, 0.98)
# Classic bottom-center hero seat (Telecaster-style when hero is south).
ROI_HERO_STACK: Tuple[float, float, float, float] = (0.40, 0.76, 0.60, 0.92)
ROI_HERO_STACK_ALT: Tuple[float, float, float, float] = (0.38, 0.78, 0.62, 0.95)
# Mid-right face-up hole seat (common when hero is BTN / CO on 9-max).
ROI_HERO_STACK_RIGHT: Tuple[float, float, float, float] = (0.55, 0.25, 0.95, 0.55)

# Full-band fallbacks when primary ROIs miss (browser chrome / DPI).
# Pot band stays above mid-felt — excludes stakes line and jackpot chrome.
_BAND_POT = (0.30, 0.24, 0.70, 0.44)
_BAND_LOWER = (0.20, 0.68, 0.80, 0.95)


@dataclass
class TableAmounts:
    """Chip amounts in BB (or SC treated as the same unit). None = unreadable."""

    pot: Optional[float] = None
    to_call: Optional[float] = None
    hero_stack: Optional[float] = None

    def format_status(self) -> str:
        def _f(v: Optional[float]) -> str:
            if v is None:
                return "?"
            if abs(v - round(v)) < 1e-6:
                return str(int(round(v)))
            return f"{v:g}"

        return f"pot={_f(self.pot)} to_call={_f(self.to_call)} stack={_f(self.hero_stack)}"


# Number with optional thousands commas / decimals.
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
# Prefer amounts that carry an explicit BB|SC unit (Stake UI).
_AMOUNT_BB_RE = re.compile(
    rf"{_NUM}\s*(?:BB|SC|bb|sc)\b",
    re.IGNORECASE,
)
# Bare number fallback (no unit) — used only when no BB/SC match exists.
_AMOUNT_BARE_RE = re.compile(
    rf"(?<![.\d]){_NUM}(?![.\d])",
    re.IGNORECASE,
)
# Stake logo often OCR's as a lone digit glued before the real amount: "6 8 BB", "6126BB".
_LOGO_DIGIT_SPLIT = re.compile(
    r"(?P<logo>\d)\s+(?P<amt>\d+(?:\.\d+)?)\s*(?:BB|SC)\b",
    re.IGNORECASE,
)
_LOGO_GLUED = re.compile(
    r"(?<!\d)(?P<logo>[5689])(?P<amt>\d{2,3}(?:\.\d+)?)\s*(?:BB|SC)\b",
    re.IGNORECASE,
)
_POT_HINT = re.compile(r"pot", re.IGNORECASE)
_CALL_HINT = re.compile(r"(?:to\s*call|call|raise|bet)", re.IGNORECASE)
_CHECK_HINT = re.compile(r"\bcheck(?:ed|s)?\b", re.IGNORECASE)
# Table chrome that must never become pot / to_call / stack.
_TABLE_NOISE = re.compile(
    r"(?:nlhe|nl\s*hold|bad\s*beat|jackpot|0\.\d{1,3}\s*/\s*0\.\d{1,3}"
    r"|tigard|hold'?em)",
    re.IGNORECASE,
)
# OCR often drops one B: "209.5B" / "5B" still means BB.
_AMOUNT_B_LOOSE = re.compile(
    rf"{_NUM}\s*B\b",
    re.IGNORECASE,
)


def _normalize_ocr_text(text: str) -> str:
    """Strip Stake icon / currency noise; keep letters for Pot/BB/Call."""
    if not text:
        return ""
    cleaned = str(text).replace("\u00a0", " ")
    # Common OCR stand-ins for the green [S] logo
    cleaned = cleaned.replace("@", " ")
    cleaned = re.sub(r"[\[\]\|\(\)\{\}\$€£¥♠♥♦♣]", " ", cleaned)
    # Lone capital S (logo) — not part of SC when followed by C
    cleaned = re.sub(r"(?<![A-Za-z])S(?![Cc])", " ", cleaned)
    cleaned = re.sub(r"[^\w.\s,:%\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_float(raw: str) -> Optional[float]:
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    if value < 0 or value > 1_000_000:
        return None
    return value


def _is_table_noise(text: str) -> bool:
    """True for stakes line / jackpot / table-name chrome (not pot or stack)."""
    if not text:
        return False
    return bool(_TABLE_NOISE.search(_normalize_ocr_text(text)))


def parse_amount_bb(text: str, *, prefer_unit: bool = True) -> Optional[float]:
    """Extract a chip amount in BB from OCR / UI text.

    Prefers ``N BB`` / ``N SC`` over bare numbers. Handles Stake logo garbage
    that becomes a leading digit (``Pot: 6 8 BB`` → 8, ``6126 BB`` → 126).
    """
    if not text or not str(text).strip():
        return None
    cleaned = _normalize_ocr_text(text)

    # Explicit logo + amount patterns first
    m = _LOGO_DIGIT_SPLIT.search(cleaned)
    if m:
        got = _parse_float(m.group("amt"))
        if got is not None:
            return got
    m = _LOGO_GLUED.search(cleaned)
    if m:
        got = _parse_float(m.group("amt"))
        if got is not None:
            return got

    unit_matches = list(_AMOUNT_BB_RE.finditer(cleaned))
    if unit_matches:
        # When multiple ``N BB`` appear, take the last (often after Pot:/Call:)
        return _parse_float(unit_matches[-1].group("num"))

    # Single trailing B (OCR dropped one): "209.5B"
    m = _AMOUNT_B_LOOSE.search(cleaned)
    if m:
        got = _parse_float(m.group("num"))
        if got is not None:
            return got

    if prefer_unit:
        # Still try bare number as weak fallback
        m = _AMOUNT_BARE_RE.search(cleaned)
        if m:
            return _parse_float(m.group("num"))
        return None

    m = _AMOUNT_BARE_RE.search(cleaned)
    if not m:
        return None
    return _parse_float(m.group("num"))


def parse_pot(text: str) -> Optional[float]:
    """Amount next to a 'Pot' label only — never bare stakes/jackpot numbers."""
    if not text:
        return None
    cleaned = _normalize_ocr_text(text)
    m = re.search(
        r"pot\s*[:\-]?\s*(?P<body>.+)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    body = m.group("body")
    # Stop at newline / unrelated chrome after the pot amount
    body = re.split(r"[\n\r]|nlhe|jackpot", body, maxsplit=1, flags=re.IGNORECASE)[0]
    got = parse_amount_bb(body)
    if got is not None and got > 50_000:
        return None
    return got


def parse_to_call(text: str) -> Optional[float]:
    """Facing bet in BB from Call/Bet UI only.

    Bare ``N BB`` without call language is rejected (often pot or seat bets).
    Check / checked → 0. Stakes line / jackpot → None.
    """
    if not text:
        return None
    if _is_table_noise(text) and not _CALL_HINT.search(text):
        return None
    cleaned = _normalize_ocr_text(text)
    if _CHECK_HINT.search(cleaned) and not _CALL_HINT.search(cleaned):
        return 0.0
    m = re.search(
        r"(?:to\s*call|call|raise|bet)\s*[:\-]?\s*(?P<body>.+)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        got = parse_amount_bb(m.group("body"))
        if got is not None:
            return got
    # No Call/Bet/Raise label → do not invent a facing amount from bare chips
    return None


def parse_hero_stack(text: str) -> Optional[float]:
    """Stack under the nameplate — usually bare '99 BB' / '222.5 BB'.

    When a crop also contains a seat bet chip (``1 BB``), prefer the larger
    stack amount rather than the last OCR match.
    """
    if not text or _is_table_noise(text):
        return None
    cleaned = _normalize_ocr_text(text)

    m = _LOGO_DIGIT_SPLIT.search(cleaned)
    if m:
        got = _parse_float(m.group("amt"))
        if got is not None and got <= 20_000:
            return got
    m = _LOGO_GLUED.search(cleaned)
    if m:
        got = _parse_float(m.group("amt"))
        if got is not None and got <= 20_000:
            return got

    values: List[float] = []
    for m in _AMOUNT_BB_RE.finditer(cleaned):
        got = _parse_float(m.group("num"))
        if got is not None and got <= 20_000:
            values.append(got)
    for m in _AMOUNT_B_LOOSE.finditer(cleaned):
        got = _parse_float(m.group("num"))
        if got is not None and got <= 20_000:
            values.append(got)
    if values:
        # Drop tiny bet chips when a real stack is also present
        big = [v for v in values if v >= 5.0]
        return max(big) if big else max(values)

    got = parse_amount_bb(cleaned)
    if got is not None and got > 20_000:
        return None
    return got


def _roi_pixels(
    frame: np.ndarray,
    frac: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x0 = max(0, int(frac[0] * w))
    y0 = max(0, int(frac[1] * h))
    x1 = min(w, int(frac[2] * w))
    y1 = min(h, int(frac[3] * h))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, max(1, w), max(1, h)
    return x0, y0, x1, y1


def crop_roi(frame: np.ndarray, frac: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = _roi_pixels(frame, frac)
    return frame[y0:y1, x0:x1].copy()


def _suppress_green_logo(crop_bgr: np.ndarray) -> np.ndarray:
    """Paint over the green Stake [S] circle so OCR does not read it as a digit."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3:
        return crop_bgr
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Stake SC icon is saturated green
    mask = cv2.inRange(hsv, (35, 60, 60), (95, 255, 255))
    if int(np.count_nonzero(mask)) < 8:
        return crop_bgr
    out = crop_bgr.copy()
    # Fill with local dark felt so text contrast stays high
    out[mask > 0] = (25, 35, 30)
    kernel = np.ones((3, 3), np.uint8)
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    out[mask_d > 0] = (25, 35, 30)
    return out


def _binarize_variants(gray: np.ndarray) -> List[np.ndarray]:
    """Several threshold variants; caller OCRs each and picks the best parse."""
    variants: List[np.ndarray] = []
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 140 else (2 if max(h, w) < 240 else 2)
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # Light text on dark → invert for Tesseract's dark-on-light assumption
    if float(np.mean(up)) < 120:
        inv = 255 - up
    else:
        inv = up
    blur = cv2.GaussianBlur(inv, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(inv)
    # Adaptive (helps uneven felt lighting)
    try:
        adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
        variants.append(adapt)
    except Exception:
        pass
    # Mild morph clean on Otsu
    kernel = np.ones((2, 2), np.uint8)
    variants.append(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel))
    return variants


def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale + Otsu for light text on dark Stake chrome (primary variant)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((8, 8), dtype=np.uint8)
    masked = _suppress_green_logo(crop_bgr)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY) if masked.ndim == 3 else masked
    return _binarize_variants(gray)[0]


def _ocr_image(img: np.ndarray, *, psm: int = 7) -> str:
    """Run pytesseract if available; else empty string."""
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        _ensure_tesseract_cmd()
        config = f"--psm {psm} -c tessedit_char_whitelist={_TESS_WHITELIST}"
        return pytesseract.image_to_string(img, config=config) or ""
    except Exception:
        return ""


def ocr_roi(
    frame: np.ndarray,
    frac: Tuple[float, float, float, float],
    *,
    psms: Sequence[int] = (7, 6, 11),
) -> str:
    """OCR one ROI; try preprocess variants × PSM modes; return best raw text."""
    crop = crop_roi(frame, frac)
    if crop.size == 0:
        return ""
    masked = _suppress_green_logo(crop)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY) if masked.ndim == 3 else masked
    best_text = ""
    best_score = -1
    for prep in _binarize_variants(gray):
        for psm in psms:
            text = _ocr_image(prep, psm=psm).strip()
            if not text:
                continue
            score = _ocr_text_score(text)
            if score > best_score:
                best_score = score
                best_text = text
    return best_text


def _ocr_text_score(text: str) -> int:
    """Higher = more likely real Stake amount text (BB / Pot / digits)."""
    t = text.lower()
    score = 0
    if re.search(r"\bb\s*b\b|\bbb\b|\bsc\b", t):
        score += 5
    if "pot" in t:
        score += 4
    if re.search(r"\d", t):
        score += 2
    if re.search(r"\d+\.\d+", t):
        score += 1
    # Penalize long browser-chrome garbage
    if len(text) > 80:
        score -= 3
    return score


def _ocr_and_parse(
    frame: np.ndarray,
    rois: Sequence[Tuple[float, float, float, float]],
    parser: Callable[[str], Optional[float]],
) -> Tuple[Optional[float], str]:
    """Try multiple ROIs; return (parsed, raw_text) for the first successful parse."""
    texts: List[str] = []
    for frac in rois:
        raw = ocr_roi(frame, frac)
        texts.append(raw)
        got = parser(raw)
        if got is not None:
            return got, raw
    return None, " | ".join(t for t in texts if t)


def _band_search_pot(frame: np.ndarray) -> Tuple[Optional[float], str]:
    """Regex-scan a fixed upper-center band for ``Pot: … BB`` when ROI miss."""
    raw = ocr_roi(frame, _BAND_POT, psms=(6, 11))
    if not raw:
        return None, ""
    if _is_table_noise(raw) and not _POT_HINT.search(raw):
        return None, raw
    m = re.search(
        r"pot\s*[:\-]?\s*.{0,24}?(\d+(?:[.,]\d+)?)\s*(?:BB|SC|B)\b",
        _normalize_ocr_text(raw),
        re.IGNORECASE,
    )
    if m:
        return _parse_float(m.group(1)), raw
    return parse_pot(raw), raw


def _action_says_check(frame: np.ndarray) -> bool:
    raw = ocr_roi(frame, ROI_ACTION, psms=(6, 11))
    return bool(_CHECK_HINT.search(raw or "")) and not bool(
        re.search(r"\bcall\b", raw or "", re.IGNORECASE)
    )


def _face_up_hole_rois(
    frame: np.ndarray,
) -> List[Tuple[float, float, float, float]]:
    """Build stack ROIs under bright face-up card blobs (hero hole cards).

    Only keeps compact card-like pairs away from the center board strip and
    bottom-center Telecaster seat, so we OCR the stack under hero holes.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mask = (gray > 155).astype(np.uint8) * 255
    mask[: int(0.12 * h), :] = 0
    mask[int(0.82 * h) :, :] = 0  # drop bottom nameplates / UI
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(400, (w * h) // 3500) or bh < 22 or bw < 14:
            continue
        if bh > 0.22 * h or bw > 0.16 * w:
            continue
        aspect = bw / float(max(1, bh))
        # Single hole card ~0.55–0.75; overlapped pair a bit wider
        if aspect < 0.40 or aspect > 1.15:
            continue
        boxes.append((x, y, x + bw, y + bh))
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
    rois: List[Tuple[float, float, float, float]] = []
    used = [False] * len(boxes)
    for i, a in enumerate(boxes):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        ax = (a[0] + a[2]) / 2.0
        ay = (a[1] + a[3]) / 2.0
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            bx = (b[0] + b[2]) / 2.0
            by = (b[1] + b[3]) / 2.0
            if abs(bx - ax) < 0.10 * w and abs(by - ay) < 0.08 * h:
                group.append(b)
                used[j] = True
        # Hero holes are a pair (or one wide overlapped blob)
        if len(group) > 3:
            continue
        xs0 = min(g[0] for g in group)
        ys0 = min(g[1] for g in group)
        xs1 = max(g[2] for g in group)
        ys1 = max(g[3] for g in group)
        cx = ((xs0 + xs1) / 2.0) / w
        cy = ((ys0 + ys1) / 2.0) / h
        # Skip center board strip
        if 0.30 <= cx <= 0.70 and 0.30 <= cy <= 0.55:
            continue
        # Skip bottom-center seat (Telecaster) — not hero when holes are side
        if 0.35 <= cx <= 0.65 and cy >= 0.68:
            continue
        x0 = max(0.0, cx - 0.16)
        x1 = min(1.0, cx + 0.16)
        y0 = max(0.0, cy - 0.06)
        y1 = min(1.0, cy + 0.28)
        rois.append((x0, y0, x1, y1))
    return rois


def _hero_stack_roi_list(
    frame: np.ndarray,
    hero_stack_roi: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """Prefer mid-right hole seat, then detected holes, then bottom."""
    rois: List[Tuple[float, float, float, float]] = []
    # Fixed mid-right first — calibrated to read "209.5 BB" under rhyzome
    rois.append(ROI_HERO_STACK_RIGHT)
    rois.extend(_face_up_hole_rois(frame))
    rois.append(hero_stack_roi)
    rois.append(ROI_HERO_STACK_ALT)
    uniq: List[Tuple[float, float, float, float]] = []
    for r in rois:
        if any(
            abs(r[0] - u[0]) < 0.02
            and abs(r[1] - u[1]) < 0.02
            and abs(r[2] - u[2]) < 0.02
            and abs(r[3] - u[3]) < 0.02
            for u in uniq
        ):
            continue
        uniq.append(r)
    return uniq


def _pick_hero_stack(
    frame: np.ndarray,
    rois: Sequence[Tuple[float, float, float, float]],
) -> Tuple[Optional[float], str]:
    """OCR stack candidates; prefer hole-adjacent parses with real stacks."""
    best: Optional[float] = None
    best_raw = ""
    best_score = -1
    for i, frac in enumerate(rois):
        raw = ocr_roi(frame, frac)
        got = parse_hero_stack(raw)
        if got is None:
            continue
        score = 0.0
        score += max(0, 6 - i)
        if re.search(r"\d+\.\d+", raw or ""):
            score += 5.0  # decimals like 209.5
        if got >= 20.0:
            score += 4.0  # real stack, not a 1 BB chip
        elif got < 5.0:
            score -= 6.0
        if re.search(r"\bbb\b|\bB\b", raw or "", re.IGNORECASE):
            score += 2.0
        cx = (frac[0] + frac[2]) / 2.0
        cy = (frac[1] + frac[3]) / 2.0
        if 0.35 <= cx <= 0.65 and cy >= 0.70:
            score -= 8.0  # bottom-center Telecaster
        if score > best_score:
            best_score = score
            best = got
            best_raw = raw
    return best, best_raw


def read_table_amounts(
    frame: np.ndarray,
    *,
    pot_roi: Tuple[float, float, float, float] = ROI_POT,
    to_call_roi: Tuple[float, float, float, float] = ROI_TO_CALL,
    hero_stack_roi: Tuple[float, float, float, float] = ROI_HERO_STACK,
    to_call_alt_roi: Tuple[float, float, float, float] = ROI_TO_CALL_ALT,
    debug: bool = False,
) -> TableAmounts:
    """OCR the ROIs and parse BB amounts. Missing OCR → Nones."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return TableAmounts()

    global _last_ocr_debug_at
    do_debug = debug or _DEBUG_OCR

    pot, pot_raw = _ocr_and_parse(
        frame, (pot_roi, ROI_POT_ALT), parse_pot
    )
    if pot is None:
        pot, pot_raw = _band_search_pot(frame)

    # Facing bet: Call/Bet language only; Check / unclear ⇒ 0 (never pot)
    call, call_raw = _ocr_and_parse(
        frame, (to_call_roi, to_call_alt_roi, ROI_ACTION), parse_to_call
    )
    action_check = _action_says_check(frame)
    if action_check:
        call = 0.0
        call_raw = (call_raw or "") + " [check]"
    elif call is not None and _POT_HINT.search(call_raw or "") and not _CALL_HINT.search(
        call_raw or ""
    ):
        # Ambiguous crop that looks like the pot pill — not a facing bet
        call = None
        call_raw = (call_raw or "") + " [rejected-pot]"

    # Mid-table chip equal to pot without Call language is pot echo, not to_call
    if (
        call is not None
        and pot is not None
        and abs(call - pot) < 1e-6
        and not _CALL_HINT.search(call_raw or "")
    ):
        call = None
        call_raw = (call_raw or "") + " [eq-pot]"

    # Unclear facing amount → 0 (do not invent from pot / stakes)
    if call is None:
        call = 0.0
        if not call_raw:
            call_raw = "[unclear→0]"

    stack_rois = _hero_stack_roi_list(frame, hero_stack_roi)
    stack, stack_raw = _pick_hero_stack(frame, stack_rois)
    if stack is None:
        band = ocr_roi(frame, _BAND_LOWER, psms=(6, 11))
        stack_raw = band
        stack = parse_hero_stack(band)

    if do_debug:
        now = time.monotonic()
        if now - _last_ocr_debug_at >= _OCR_DEBUG_INTERVAL_S:
            _last_ocr_debug_at = now
            print(
                f"[ocr] pot raw={pot_raw!r} -> {pot} | "
                f"to_call raw={call_raw!r} -> {call} | "
                f"stack raw={stack_raw!r} -> {stack}"
            )

    return TableAmounts(pot=pot, to_call=call, hero_stack=stack)


def pot_odds(pot: Optional[float], to_call: Optional[float]) -> Optional[float]:
    """to_call / (pot + to_call). None if not facing a bet."""
    if to_call is None or to_call <= 0:
        return None
    pot_v = float(pot) if pot is not None and pot > 0 else 0.0
    denom = pot_v + to_call
    if denom <= 0:
        return None
    return to_call / denom


def merge_amounts(prev: Optional[TableAmounts], new: TableAmounts) -> TableAmounts:
    """Keep last good BB values when a fresh OCR pass returns None for a field."""
    if prev is None:
        return TableAmounts(
            pot=new.pot, to_call=new.to_call, hero_stack=new.hero_stack
        )
    return TableAmounts(
        pot=new.pot if new.pot is not None else prev.pot,
        to_call=new.to_call if new.to_call is not None else prev.to_call,
        hero_stack=new.hero_stack if new.hero_stack is not None else prev.hero_stack,
    )


def frame_fingerprint(frame: np.ndarray, *, size: int = 32) -> bytes:
    """Cheap downscaled grayscale digest for 'frame unchanged' skips."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return b""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    # Quantize to cut camera/JPEG noise so identical tables match
    q = (small.astype(np.uint16) // 8).astype(np.uint8)
    return q.tobytes()


@dataclass
class AmountsThrottle:
    """Throttle expensive Tesseract OCR; cache last good pot/to_call/stack.

    ``read_table_amounts`` runs many ROI × preprocess × PSM passes per call.
    Calling it every capture frame is the main live-loop stall. This wrapper:
      - skips OCR when disabled (`enabled=False` / ``--no-ocr``)
      - skips when the frame fingerprint is unchanged
      - otherwise runs at most once per ``interval_s`` (default 1.5s)
      - merges None fields with the previous good read
    """

    interval_s: float = 1.5
    enabled: bool = True
    amounts: TableAmounts = field(default_factory=TableAmounts)
    last_ocr_at: float = 0.0
    last_fp: bytes = b""
    last_ran: bool = False

    def get(
        self,
        frame: np.ndarray,
        *,
        force: bool = False,
        debug: bool = False,
    ) -> TableAmounts:
        """Return cached or freshly OCR'd amounts."""
        if not self.enabled or frame is None or getattr(frame, "size", 0) == 0:
            self.last_ran = False
            return self.amounts

        fp = frame_fingerprint(frame)
        now = time.monotonic()
        unchanged = fp == self.last_fp and fp != b""
        due = (now - self.last_ocr_at) >= max(0.0, float(self.interval_s))

        if not force and unchanged:
            self.last_ran = False
            return self.amounts
        if not force and self.last_ocr_at > 0.0 and not due:
            self.last_ran = False
            return self.amounts

        fresh = read_table_amounts(frame, debug=debug)
        self.amounts = merge_amounts(self.amounts, fresh)
        self.last_ocr_at = now
        self.last_fp = fp
        self.last_ran = True
        return self.amounts


def draw_amount_rois(
    frame: np.ndarray,
    amounts: Optional[TableAmounts] = None,
    rois: Optional[Sequence[Tuple[str, Tuple[float, float, float, float]]]] = None,
) -> np.ndarray:
    """Debug helper: draw ROI rectangles (and optional parsed labels)."""
    out = frame.copy()
    labeled = rois or (
        ("pot", ROI_POT),
        ("to_call", ROI_TO_CALL),
        ("stack", ROI_HERO_STACK_RIGHT),
        ("stack_bot", ROI_HERO_STACK),
    )
    colors = {"pot": (0, 255, 255), "to_call": (0, 165, 255), "stack": (255, 200, 0)}
    for name, frac in labeled:
        x0, y0, x1, y1 = _roi_pixels(out, frac)
        color = colors.get(name, (180, 180, 180))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = name
        if amounts is not None:
            val = getattr(amounts, "hero_stack" if name == "stack" else name, None)
            if name == "to_call":
                val = amounts.to_call
            label = f"{name}={val if val is not None else '?'}"
        cv2.putText(out, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out
