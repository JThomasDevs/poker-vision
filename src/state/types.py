"""Shared card-state contracts: slots, visibility, observations, probs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.classification.model import RANKS, SUITS

Box = Tuple[int, int, int, int]

SLOT_HERO = ("hero_0", "hero_1")
SLOT_BOARD = ("board_0", "board_1", "board_2", "board_3", "board_4")

VISIBLE = "VISIBLE"
TRANSITIONING = "TRANSITIONING"
UNKNOWN = "UNKNOWN"

UNKNOWN_LABEL = "??"

# EMA / association (frozen)
EMA_ALPHA = 0.35
LOW_MARGIN_WEIGHT = 0.15
MARGIN_WEAK = 0.12
IOU_MATCH = 0.30
BOX_STABLE_IOU = 0.85
MAX_MISSED_FRAMES = 8

# Acceptance defaults
BOARD_MIN_TOP = 0.45
BOARD_MIN_MARGIN = 0.18
BOARD_MIN_TEMPORAL = 0.55
BOARD_MIN_STABLE_FRAMES = 1

HOLE_MIN_TOP = 0.28
HOLE_MIN_MARGIN = 0.10
HOLE_MIN_TEMPORAL = 0.65
# Moderate hole path: two confirming frames (was a rigid 3 for all holes).
HOLE_MIN_STABLE_FRAMES = 2

# Strong hole path: accept on first quality frame when clearly peaked.
HOLE_STRONG_TOP = 0.55
HOLE_STRONG_MARGIN = 0.25
HOLE_STRONG_STABLE_FRAMES = 1


def normalize_probs(probs: Sequence[float] | np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Return a float64 probability vector that sums to 1 (uniform if empty/zero)."""
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if n is not None:
        if arr.size == 0:
            arr = np.ones(n, dtype=np.float64)
        elif arr.size != n:
            raise ValueError(f"expected length {n}, got {arr.size}")
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        arr = np.ones(arr.size, dtype=np.float64)
        total = float(arr.size)
    return arr / total


def rank_margin(rank_probs: np.ndarray) -> float:
    p = normalize_probs(rank_probs, n=13)
    if p.size < 2:
        return float(p[0]) if p.size else 0.0
    top2 = np.partition(p, -2)[-2:]
    return float(top2[-1] - top2[-2])


def suit_margin(suit_probs: np.ndarray) -> float:
    p = normalize_probs(suit_probs, n=4)
    if p.size < 2:
        return float(p[0]) if p.size else 0.0
    top2 = np.partition(p, -2)[-2:]
    return float(top2[-1] - top2[-2])


def top_from_probs(
    rank_probs: np.ndarray,
    suit_probs: np.ndarray,
    ranks: Sequence[str] = RANKS,
    suits: Sequence[str] = SUITS,
) -> Tuple[str, str, float]:
    rp = normalize_probs(rank_probs, n=len(ranks))
    sp = normalize_probs(suit_probs, n=len(suits))
    ri = int(rp.argmax())
    si = int(sp.argmax())
    conf = float(rp[ri] * sp[si])
    return ranks[ri], suits[si], conf


@dataclass
class CardObservation:
    """One tracked card slot with banked probability distributions."""

    slot_id: str
    bbox: Box
    rank_probs: np.ndarray
    suit_probs: np.ndarray
    visibility: str = UNKNOWN
    label: str = UNKNOWN_LABEL
    confidence: float = 0.0
    rank_margin: float = 0.0
    suit_margin: float = 0.0
    stable_frames: int = 0
    first_seen: int = 0
    last_seen: int = 0
    box_stable: bool = False
    temporal_agreement: float = 0.0
    hole_mode: bool = False
    recent_labels: List[str] = field(default_factory=list)
    accept_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.rank_probs = normalize_probs(self.rank_probs, n=13)
        self.suit_probs = normalize_probs(self.suit_probs, n=4)

    @property
    def rank(self) -> str:
        if self.label == UNKNOWN_LABEL or len(self.label) < 2:
            return "?"
        return self.label[:-1]

    @property
    def suit(self) -> str:
        if self.label == UNKNOWN_LABEL or len(self.label) < 2:
            return "?"
        return self.label[-1]

    def refresh_top(self, ranks: Sequence[str] = RANKS, suits: Sequence[str] = SUITS) -> None:
        r, s, conf = top_from_probs(self.rank_probs, self.suit_probs, ranks, suits)
        self.confidence = conf
        self.rank_margin = rank_margin(self.rank_probs)
        self.suit_margin = suit_margin(self.suit_probs)
        if self.visibility == VISIBLE:
            self.label = f"{r}{s}"
        elif self.visibility == TRANSITIONING:
            self.label = UNKNOWN_LABEL
        else:
            self.label = UNKNOWN_LABEL


@dataclass
class RawDet:
    """Single-frame classifier output before tracking."""

    bbox: Box
    rank_probs: np.ndarray
    suit_probs: np.ndarray
    hole_mode: bool = False
    slot_hint: Optional[str] = None

    def __post_init__(self) -> None:
        self.rank_probs = normalize_probs(self.rank_probs, n=13)
        self.suit_probs = normalize_probs(self.suit_probs, n=4)
