"""Unit tests for tracked FastCardsPipeline path (no GPU / live table)."""

from __future__ import annotations

import numpy as np

from src.classification.model import RANKS, SUITS
from src.detection.pipeline import FastCardsPipeline, FastCardsResult, LabeledCard
from src.state.table import TableState, build_table_state
from src.state.tracker import CardTracker
from src.state.types import (
    BOARD_MIN_STABLE_FRAMES,
    HOLE_MIN_STABLE_FRAMES,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
    RawDet,
)


def _peaked(rank: str, suit: str, rank_p: float = 0.92, suit_p: float = 0.95):
    rp = np.full(13, (1.0 - rank_p) / 12.0, dtype=np.float64)
    sp = np.full(4, (1.0 - suit_p) / 3.0, dtype=np.float64)
    rp[RANKS.index(rank)] = rank_p
    sp[SUITS.index(suit)] = suit_p
    return rp, sp


def _det(
    slot: str,
    rank: str,
    suit: str,
    bbox,
    *,
    hole_mode: bool = False,
) -> RawDet:
    rp, sp = _peaked(rank, suit)
    return RawDet(
        bbox=bbox,
        rank_probs=rp,
        suit_probs=sp,
        hole_mode=hole_mode,
        slot_hint=slot,
    )


def _pipe_seam() -> FastCardsPipeline:
    """Construct pipeline without loading the CNN checkpoint."""
    pipe = object.__new__(FastCardsPipeline)
    pipe.tracker = CardTracker()
    pipe.use_tracker = True
    return pipe


def test_fast_cards_result_table_default_none():
    r = FastCardsResult()
    assert r.table is None
    assert r.community == []
    assert r.holes == []


def test_build_table_state_valid_when_heroes_visible_no_board():
    rp, sp = _peaked("A", "s")
    hero = [
        CardObservation(
            slot_id="hero_0",
            bbox=(0, 0, 10, 20),
            rank_probs=rp,
            suit_probs=sp,
            visibility=VISIBLE,
            label="As",
            confidence=0.9,
            hole_mode=True,
        ),
        CardObservation(
            slot_id="hero_1",
            bbox=(20, 0, 30, 20),
            rank_probs=rp,
            suit_probs=sp,
            visibility=VISIBLE,
            label="Kh",
            confidence=0.88,
            hole_mode=True,
        ),
    ]
    # Fix second label probs for Kh
    hero[1].rank_probs, hero[1].suit_probs = _peaked("K", "h")
    table = build_table_state(hero, board_detected=False)
    assert table.state_valid is True
    assert len(table.hero) == 2
    assert table.street == "preflop"
    assert table.vision_confidence > 0.5
    assert 0.0 <= table.uncertainty <= 1.0


def test_build_table_state_invalid_unresolved_board():
    rp, sp = _peaked("A", "s")
    obs = [
        CardObservation(
            slot_id="hero_0",
            bbox=(0, 100, 10, 120),
            rank_probs=rp,
            suit_probs=sp,
            visibility=VISIBLE,
            label="As",
            confidence=0.9,
            hole_mode=True,
        ),
        CardObservation(
            slot_id="hero_1",
            bbox=(20, 100, 30, 120),
            rank_probs=_peaked("K", "h")[0],
            suit_probs=_peaked("K", "h")[1],
            visibility=VISIBLE,
            label="Kh",
            confidence=0.88,
            hole_mode=True,
        ),
        CardObservation(
            slot_id="board_0",
            bbox=(50, 10, 70, 40),
            rank_probs=rp,
            suit_probs=sp,
            visibility=UNKNOWN,
            label=UNKNOWN_LABEL,
            confidence=0.0,
        ),
    ]
    table = build_table_state(obs, board_detected=True)
    assert table.state_valid is False


def test_apply_tracker_dets_emits_visible_after_stability():
    """Synthetic RawDet stream → VISIBLE hero+board without CNN."""
    pipe = _pipe_seam()
    board_box = (100, 40, 140, 100)
    h0 = (80, 200, 120, 260)
    h1 = (130, 200, 170, 260)

    frames = max(BOARD_MIN_STABLE_FRAMES, HOLE_MIN_STABLE_FRAMES) + 1
    table = None
    community: list[LabeledCard] = []
    holes: list[LabeledCard] = []
    for _ in range(frames):
        dets = [
            _det("board_0", "9", "s", board_box),
            _det("hero_0", "A", "h", h0, hole_mode=True),
            _det("hero_1", "K", "d", h1, hole_mode=True),
        ]
        table, community, holes = pipe.apply_tracker_dets(
            dets, board_detected=True
        )

    assert table is not None
    assert isinstance(table, TableState)
    assert table.state_valid is True
    assert [c.label for c in holes] == ["Ah", "Kd"]
    assert [c.label for c in community] == ["9s"]
    assert all(o.visibility == VISIBLE for o in table.hero)
    assert all(o.visibility == VISIBLE for o in table.board)


def test_apply_tracker_dets_omits_unknown_from_equity_lists():
    pipe = _pipe_seam()
    # Single weak frame → acceptance keeps UNKNOWN; lists empty
    flat_r = np.ones(13, dtype=np.float64) / 13.0
    flat_s = np.ones(4, dtype=np.float64) / 4.0
    dets = [
        RawDet(
            bbox=(10, 10, 40, 50),
            rank_probs=flat_r,
            suit_probs=flat_s,
            hole_mode=True,
            slot_hint="hero_0",
        ),
        RawDet(
            bbox=(50, 10, 80, 50),
            rank_probs=flat_r,
            suit_probs=flat_s,
            hole_mode=True,
            slot_hint="hero_1",
        ),
    ]
    table, community, holes = pipe.apply_tracker_dets(dets, board_detected=False)
    assert table.state_valid is False
    assert community == []
    assert holes == []
