"""Tests for src.state.types contracts."""

from __future__ import annotations

import numpy as np

from src.state.types import (
    SLOT_BOARD,
    SLOT_HERO,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
    normalize_probs,
    rank_margin,
)


def test_slot_constants():
    assert SLOT_HERO == ("hero_0", "hero_1")
    assert len(SLOT_BOARD) == 5
    assert SLOT_BOARD[0] == "board_0"


def test_normalize_probs_sums_to_one():
    p = normalize_probs([1.0, 2.0, 3.0], n=3)
    assert p.shape == (3,)
    assert abs(float(p.sum()) - 1.0) < 1e-9


def test_normalize_probs_zero_vector_uniform():
    p = normalize_probs([0.0, 0.0, 0.0], n=3)
    assert np.allclose(p, np.ones(3) / 3.0)


def test_card_observation_unknown_label():
    obs = CardObservation(
        slot_id="hero_0",
        bbox=(0, 0, 10, 20),
        rank_probs=np.ones(13),
        suit_probs=np.ones(4),
        visibility=UNKNOWN,
    )
    obs.refresh_top()
    assert obs.label == UNKNOWN_LABEL
    assert obs.rank == "?"
    assert obs.suit == "?"


def test_card_observation_visible_label():
    rp = np.zeros(13)
    rp[7] = 1.0  # "9"
    sp = np.zeros(4)
    sp[3] = 1.0  # "s"
    obs = CardObservation(
        slot_id="board_0",
        bbox=(1, 2, 3, 4),
        rank_probs=rp,
        suit_probs=sp,
        visibility=VISIBLE,
    )
    obs.refresh_top()
    assert obs.label == "9s"
    assert obs.confidence > 0.99
    assert rank_margin(rp) > 0.9
