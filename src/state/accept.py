"""Acceptance gates and deck uniqueness for table card observations."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .types import (
    BOARD_MIN_MARGIN,
    BOARD_MIN_STABLE_FRAMES,
    BOARD_MIN_TEMPORAL,
    BOARD_MIN_TOP,
    HOLE_MIN_MARGIN,
    HOLE_MIN_STABLE_FRAMES,
    HOLE_MIN_TEMPORAL,
    HOLE_MIN_TOP,
    SLOT_BOARD,
    SLOT_HERO,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
    top_from_probs,
)


def _thresholds_for(
    obs: CardObservation,
    hole_slots: Sequence[str],
    board_slots: Sequence[str],
) -> Tuple[float, float, float, int]:
    """Return (min_top, min_margin, min_temporal, min_stable_frames)."""
    if obs.slot_id in hole_slots:
        return HOLE_MIN_TOP, HOLE_MIN_MARGIN, HOLE_MIN_TEMPORAL, HOLE_MIN_STABLE_FRAMES
    if obs.slot_id in board_slots:
        return BOARD_MIN_TOP, BOARD_MIN_MARGIN, BOARD_MIN_TEMPORAL, BOARD_MIN_STABLE_FRAMES
    if obs.hole_mode:
        return HOLE_MIN_TOP, HOLE_MIN_MARGIN, HOLE_MIN_TEMPORAL, HOLE_MIN_STABLE_FRAMES
    return BOARD_MIN_TOP, BOARD_MIN_MARGIN, BOARD_MIN_TEMPORAL, BOARD_MIN_STABLE_FRAMES


def _passes_local_gate(
    obs: CardObservation,
    min_top: float,
    min_margin: float,
    min_temporal: float,
    min_stable: int,
) -> bool:
    return (
        obs.confidence >= min_top
        and obs.rank_margin >= min_margin
        and obs.stable_frames >= min_stable
        and obs.temporal_agreement >= min_temporal
        and obs.box_stable
    )


def _demote(obs: CardObservation) -> None:
    obs.visibility = UNKNOWN
    obs.label = UNKNOWN_LABEL


def _promote(obs: CardObservation) -> None:
    rank, suit, _conf = top_from_probs(obs.rank_probs, obs.suit_probs)
    obs.visibility = VISIBLE
    obs.label = f"{rank}{suit}"


def accept_table(
    obs: List[CardObservation],
    *,
    hole_slots: Sequence[str] = SLOT_HERO,
    board_slots: Sequence[str] = SLOT_BOARD,
) -> List[CardObservation]:
    """Apply local VISIBLE gates then enforce unique labels across the table.

    Local gate (hole vs board thresholds from types.py):
      top confidence, rank margin, stable_frames, temporal_agreement, box_stable.
    Failures become UNKNOWN / ??. Among local passers, duplicate labels keep the
    higher-confidence observation and demote the rest.
    """
    passed: List[CardObservation] = []
    for o in obs:
        min_top, min_margin, min_temporal, min_stable = _thresholds_for(
            o, hole_slots, board_slots
        )
        if _passes_local_gate(o, min_top, min_margin, min_temporal, min_stable):
            _promote(o)
            passed.append(o)
        else:
            _demote(o)

    by_label: Dict[str, List[CardObservation]] = {}
    for o in passed:
        by_label.setdefault(o.label, []).append(o)

    for group in by_label.values():
        if len(group) <= 1:
            continue
        winner = max(group, key=lambda x: x.confidence)
        for o in group:
            if o is not winner:
                _demote(o)

    return obs
