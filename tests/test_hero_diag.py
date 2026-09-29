"""Unit tests for hero hole-path diagnostics (no live table)."""

from __future__ import annotations

import numpy as np

from src.classification.model import RANKS, SUITS
from src.state.accept import (
    HOLE_REASON_ACCEPTED_MODERATE,
    HOLE_REASON_ACCEPTED_STRONG,
    HOLE_REASON_NEED_STABLE_2,
    HOLE_REASON_NOT_BOX_STABLE,
    HOLE_REASON_WEAK_CONF,
    HOLE_REASON_WEAK_MARGIN,
    accept_table,
    hole_accept_reason,
)
from src.state.hero_diag import HeroSlotDiag, HeroTickDiag
from src.state.types import (
    HOLE_MIN_STABLE_FRAMES,
    HOLE_STRONG_MARGIN,
    HOLE_STRONG_TOP,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
)


def _peaked(rank: str, suit: str, rank_p: float = 0.95, suit_p: float = 0.95):
    rp = np.full(13, (1.0 - rank_p) / 12.0, dtype=np.float64)
    sp = np.full(4, (1.0 - suit_p) / 3.0, dtype=np.float64)
    rp[RANKS.index(rank)] = rank_p
    sp[SUITS.index(suit)] = suit_p
    return rp, sp


def _obs(
    *,
    confidence: float,
    rank_margin: float,
    stable_frames: int,
    temporal_agreement: float = 0.80,
    box_stable: bool = True,
    rank: str = "A",
    suit: str = "h",
) -> CardObservation:
    rp, sp = _peaked(rank, suit)
    return CardObservation(
        slot_id="hero_0",
        bbox=(10, 20, 30, 60),
        rank_probs=rp,
        suit_probs=sp,
        visibility=UNKNOWN,
        label=UNKNOWN_LABEL,
        confidence=confidence,
        rank_margin=rank_margin,
        suit_margin=0.9,
        stable_frames=stable_frames,
        box_stable=box_stable,
        temporal_agreement=temporal_agreement,
        hole_mode=True,
    )


def test_hero_tick_diag_format_includes_stage_and_slots():
    diag = HeroTickDiag(
        hero_blobs=2,
        filtered_blobs=7,
        raw_blobs=9,
        candidate_boxes=[(100, 200, 140, 280), (150, 200, 190, 280)],
        slots=[
            HeroSlotDiag(
                slot_id="hero_0",
                box=(100, 200, 140, 280),
                box_stable=True,
                path="cheap",
                cheap_label="Ah",
                cheap_conf=0.62,
                cheap_margin=0.31,
                banked_label="Ah",
                banked_conf=0.62,
                banked_margin=0.31,
                reason=HOLE_REASON_ACCEPTED_STRONG,
            )
        ],
    )
    line = diag.format_line()
    assert line.startswith("[holes]")
    assert "hero=2/filt=7/raw=9" in line
    assert "120,240" in line  # cx,cy of first box
    assert "cheap=Ah@0.62/m0.31" in line
    assert "reason=accepted_strong" in line
    assert "path=cheap" in line


def test_hero_tick_diag_no_blobs_stage():
    diag = HeroTickDiag(hero_blobs=0, filtered_blobs=0, raw_blobs=0)
    line = diag.format_line()
    assert "reason=no_blobs" in line


def test_hero_tick_diag_filter_white_low_stage():
    diag = HeroTickDiag(hero_blobs=0, filtered_blobs=2, raw_blobs=8)
    line = diag.format_line()
    assert "reason=filter_white_low" in line


def test_hero_tick_diag_one_blob_stage():
    diag = HeroTickDiag(hero_blobs=1, filtered_blobs=4, raw_blobs=5)
    line = diag.format_line()
    assert "reason=one_blob" in line


def test_hole_accept_reasons_cover_gates():
    assert (
        hole_accept_reason(
            _obs(
                confidence=max(HOLE_STRONG_TOP, 0.85),
                rank_margin=max(HOLE_STRONG_MARGIN, 0.40),
                stable_frames=1,
            )
        )
        == HOLE_REASON_ACCEPTED_STRONG
    )
    assert (
        hole_accept_reason(
            _obs(
                confidence=0.35,
                rank_margin=0.15,
                stable_frames=HOLE_MIN_STABLE_FRAMES,
            )
        )
        == HOLE_REASON_ACCEPTED_MODERATE
    )
    assert (
        hole_accept_reason(
            _obs(confidence=0.35, rank_margin=0.15, stable_frames=1)
        )
        == HOLE_REASON_NEED_STABLE_2
    )
    assert (
        hole_accept_reason(
            _obs(
                confidence=0.35,
                rank_margin=0.15,
                stable_frames=2,
                box_stable=False,
            )
        )
        == HOLE_REASON_NOT_BOX_STABLE
    )
    assert (
        hole_accept_reason(
            _obs(confidence=0.10, rank_margin=0.15, stable_frames=5)
        )
        == HOLE_REASON_WEAK_CONF
    )
    assert (
        hole_accept_reason(
            _obs(confidence=0.35, rank_margin=0.02, stable_frames=5)
        )
        == HOLE_REASON_WEAK_MARGIN
    )


def test_accept_table_sets_accept_reason_on_holes():
    hole = _obs(
        confidence=max(HOLE_STRONG_TOP, 0.85),
        rank_margin=max(HOLE_STRONG_MARGIN, 0.40),
        stable_frames=1,
    )
    out = accept_table([hole])
    assert out[0].visibility == VISIBLE
    assert out[0].accept_reason == HOLE_REASON_ACCEPTED_STRONG

    weak = _obs(confidence=0.10, rank_margin=0.02, stable_frames=5)
    out2 = accept_table([weak])
    assert out2[0].visibility == UNKNOWN
    assert out2[0].accept_reason == HOLE_REASON_WEAK_CONF
