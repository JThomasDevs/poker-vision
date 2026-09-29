"""Tests for src.state.accept.accept_table."""

from __future__ import annotations

import numpy as np

from src.classification.model import RANKS, SUITS
from src.state.accept import accept_table
from src.state.types import (
    BOARD_MIN_STABLE_FRAMES,
    HOLE_MIN_STABLE_FRAMES,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
)


def _peaked(rank: str, suit: str, rank_p: float = 0.95, suit_p: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    rp = np.full(13, (1.0 - rank_p) / 12.0, dtype=np.float64)
    sp = np.full(4, (1.0 - suit_p) / 3.0, dtype=np.float64)
    rp[RANKS.index(rank)] = rank_p
    sp[SUITS.index(suit)] = suit_p
    return rp, sp


def _obs(
    slot_id: str,
    rank: str,
    suit: str,
    *,
    confidence: float,
    rank_margin: float,
    stable_frames: int,
    temporal_agreement: float,
    box_stable: bool = True,
    hole_mode: bool = False,
    visibility: str = UNKNOWN,
) -> CardObservation:
    rp, sp = _peaked(rank, suit)
    return CardObservation(
        slot_id=slot_id,
        bbox=(0, 0, 10, 20),
        rank_probs=rp,
        suit_probs=sp,
        visibility=visibility,
        label=UNKNOWN_LABEL,
        confidence=confidence,
        rank_margin=rank_margin,
        suit_margin=0.9,
        stable_frames=stable_frames,
        box_stable=box_stable,
        temporal_agreement=temporal_agreement,
        hole_mode=hole_mode,
        recent_labels=[f"{rank}{suit}"] * max(stable_frames, 1),
    )


def test_board_accepts_clear_high_conf_card():
    board = _obs(
        "board_0",
        "A",
        "s",
        confidence=0.90,
        rank_margin=0.50,
        stable_frames=BOARD_MIN_STABLE_FRAMES + 1,
        temporal_agreement=0.80,
    )
    out = accept_table([board])
    assert len(out) == 1
    assert out[0].visibility == VISIBLE
    assert out[0].label == "As"


def test_hole_rejects_single_frame_flip():
    """Low stable_frames / low temporal → not VISIBLE for hole cards."""
    hole = _obs(
        "hero_0",
        "K",
        "h",
        confidence=0.85,
        rank_margin=0.40,
        stable_frames=1,  # below HOLE_MIN_STABLE_FRAMES (3)
        temporal_agreement=0.20,  # below HOLE_MIN_TEMPORAL (0.65)
        hole_mode=True,
    )
    assert hole.stable_frames < HOLE_MIN_STABLE_FRAMES
    out = accept_table([hole])
    assert out[0].visibility == UNKNOWN
    assert out[0].label == UNKNOWN_LABEL


def test_duplicate_9s_demotes_weaker():
    """Hero + board both claiming 9s → keep higher confidence, demote loser."""
    hero = _obs(
        "hero_0",
        "9",
        "s",
        confidence=0.70,
        rank_margin=0.40,
        stable_frames=HOLE_MIN_STABLE_FRAMES + 1,
        temporal_agreement=0.90,
        hole_mode=True,
    )
    board = _obs(
        "board_1",
        "9",
        "s",
        confidence=0.92,
        rank_margin=0.50,
        stable_frames=BOARD_MIN_STABLE_FRAMES + 2,
        temporal_agreement=0.90,
    )
    out = accept_table([hero, board])
    by_slot = {o.slot_id: o for o in out}
    assert by_slot["board_1"].visibility == VISIBLE
    assert by_slot["board_1"].label == "9s"
    assert by_slot["hero_0"].visibility == UNKNOWN
    assert by_slot["hero_0"].label == UNKNOWN_LABEL
