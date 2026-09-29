"""Acceptance gates and deck uniqueness for table card observations."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .types import (
    BOARD_MIN_MARGIN,
    BOARD_MIN_STABLE_FRAMES,
    BOARD_MIN_TEMPORAL,
    BOARD_MIN_TOP,
    HOLE_MIN_MARGIN,
    HOLE_MIN_STABLE_FRAMES,
    HOLE_MIN_TEMPORAL,
    HOLE_MIN_TOP,
    HOLE_STRONG_MARGIN,
    HOLE_STRONG_STABLE_FRAMES,
    HOLE_STRONG_TOP,
    SLOT_BOARD,
    SLOT_HERO,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
    top_from_probs,
)


def _is_hole(
    obs: CardObservation,
    hole_slots: Sequence[str],
    board_slots: Sequence[str],
) -> bool:
    if obs.slot_id in hole_slots:
        return True
    if obs.slot_id in board_slots:
        return False
    return bool(obs.hole_mode)


def _passes_board_gate(obs: CardObservation) -> bool:
    return (
        obs.confidence >= BOARD_MIN_TOP
        and obs.rank_margin >= BOARD_MIN_MARGIN
        and obs.stable_frames >= BOARD_MIN_STABLE_FRAMES
        and obs.temporal_agreement >= BOARD_MIN_TEMPORAL
        and obs.box_stable
    )


def _passes_hole_gate(obs: CardObservation) -> bool:
    """Adaptive hole acceptance: strong → frame 1; moderate → 2 + temporal."""
    if not obs.box_stable:
        return False

    # STRONG: high conf + high margin → accept on first quality frame.
    if (
        obs.confidence >= HOLE_STRONG_TOP
        and obs.rank_margin >= HOLE_STRONG_MARGIN
        and obs.stable_frames >= HOLE_STRONG_STABLE_FRAMES
    ):
        return True

    # MODERATE: baseline thresholds need two confirming frames + temporal.
    if (
        obs.confidence >= HOLE_MIN_TOP
        and obs.rank_margin >= HOLE_MIN_MARGIN
        and obs.stable_frames >= HOLE_MIN_STABLE_FRAMES
        and obs.temporal_agreement >= HOLE_MIN_TEMPORAL
    ):
        return True

    return False


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

    Hole cards use adaptive gates (strong → 1 frame, moderate → 2 + temporal).
    Board keeps first-frame acceptance. Failures become UNKNOWN / ??.
    Among local passers, duplicate labels keep the higher-confidence observation
    and demote the rest.
    """
    passed: List[CardObservation] = []
    for o in obs:
        if _is_hole(o, hole_slots, board_slots):
            ok = _passes_hole_gate(o)
        else:
            ok = _passes_board_gate(o)
        if ok:
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
