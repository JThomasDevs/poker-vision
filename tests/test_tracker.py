"""Tests for CardTracker IoU association and EMA banking."""

from __future__ import annotations

import numpy as np
import pytest

from src.classification.model import RANKS, SUITS
from src.state.tracker import CardTracker, box_iou
from src.state.types import (
    MAX_MISSED_FRAMES,
    UNKNOWN,
    UNKNOWN_LABEL,
    RawDet,
    top_from_probs,
)


def _peaked_rank(rank: str, conf: float = 0.92) -> np.ndarray:
    idx = RANKS.index(rank)
    p = np.full(13, (1.0 - conf) / 12.0, dtype=np.float64)
    p[idx] = conf
    return p


def _peaked_suit(suit: str, conf: float = 0.95) -> np.ndarray:
    idx = SUITS.index(suit)
    p = np.full(4, (1.0 - conf) / 3.0, dtype=np.float64)
    p[idx] = conf
    return p


def _two_rank(primary: str, p1: float, secondary: str, p2: float) -> np.ndarray:
    """Two-peak rank vector; remaining mass uniform on others."""
    i1, i2 = RANKS.index(primary), RANKS.index(secondary)
    rest = max(0.0, 1.0 - p1 - p2)
    p = np.full(13, rest / 11.0, dtype=np.float64)
    p[i1] = p1
    p[i2] = p2
    return p


def test_box_iou_identical():
    b = (10, 20, 50, 80)
    assert box_iou(b, b) == pytest.approx(1.0)


def test_oscillation_banked_top_stays_9s():
    """Strong 9s bank then a weak 6s/9s flip barely moves EMA → still 9s."""
    tracker = CardTracker()
    box = (100, 100, 140, 160)
    strong_9s = RawDet(
        bbox=box,
        rank_probs=_peaked_rank("9", 0.90),
        suit_probs=_peaked_suit("s", 0.95),
        slot_hint="board_0",
    )
    # Frames 0-3: strong 9s
    for _ in range(4):
        out = tracker.update([strong_9s])
        assert len(out) == 1
        r, s, _ = top_from_probs(out[0].rank_probs, out[0].suit_probs)
        assert f"{r}{s}" == "9s"

    # Frame 4: oscillating 6s/.54 vs 9s/.43 (margin 0.11 < MARGIN_WEAK)
    osc = RawDet(
        bbox=box,
        rank_probs=_two_rank("6", 0.54, "9", 0.43),
        suit_probs=_peaked_suit("s", 0.90),
        slot_hint="board_0",
    )
    out = tracker.update([osc])
    assert len(out) == 1
    r, s, _ = top_from_probs(out[0].rank_probs, out[0].suit_probs)
    assert f"{r}{s}" == "9s"
    # Banked 9 mass should still dominate 6
    assert float(out[0].rank_probs[RANKS.index("9")]) > float(
        out[0].rank_probs[RANKS.index("6")]
    )


def test_track_through_small_box_jitter():
    tracker = CardTracker()
    base = (200, 200, 240, 260)
    det0 = RawDet(
        bbox=base,
        rank_probs=_peaked_rank("A", 0.88),
        suit_probs=_peaked_suit("h", 0.9),
        slot_hint="hero_0",
        hole_mode=True,
    )
    out0 = tracker.update([det0])
    assert len(out0) == 1
    slot = out0[0].slot_id

    # Small jitter well above IOU_MATCH and near BOX_STABLE_IOU
    jittered = (202, 201, 242, 261)
    assert box_iou(base, jittered) >= 0.85
    det1 = RawDet(
        bbox=jittered,
        rank_probs=_peaked_rank("A", 0.87),
        suit_probs=_peaked_suit("h", 0.91),
        slot_hint="hero_0",
        hole_mode=True,
    )
    out1 = tracker.update([det1])
    assert len(out1) == 1
    assert out1[0].slot_id == slot
    r, s, _ = top_from_probs(out1[0].rank_probs, out1[0].suit_probs)
    assert f"{r}{s}" == "Ah"


def test_miss_drop_after_max_missed_frames():
    tracker = CardTracker()
    det = RawDet(
        bbox=(0, 0, 40, 60),
        rank_probs=_peaked_rank("K", 0.9),
        suit_probs=_peaked_suit("c", 0.9),
        slot_hint="board_1",
    )
    assert len(tracker.update([det])) == 1

    # Miss for MAX_MISSED_FRAMES - 1: track still held internally (not emitted)
    for _ in range(MAX_MISSED_FRAMES - 1):
        assert tracker.update([]) == []

    # One more miss drops the track; a new det at same box is a fresh track
    assert tracker.update([]) == []

    again = tracker.update([det])
    assert len(again) == 1
    # Fresh track: first_seen resets (not carrying long history)
    assert again[0].stable_frames <= 1


def test_unknown_on_garbage_flat_probs():
    tracker = CardTracker()
    flat = RawDet(
        bbox=(50, 50, 90, 110),
        rank_probs=np.ones(13, dtype=np.float64),
        suit_probs=np.ones(4, dtype=np.float64),
        slot_hint="board_2",
    )
    out = tracker.update([flat])
    assert len(out) == 1
    assert out[0].visibility == UNKNOWN
    assert out[0].label == UNKNOWN_LABEL
