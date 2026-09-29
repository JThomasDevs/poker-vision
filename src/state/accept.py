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

# Known hole accept/reject reason strings (instrumentation / tests).
HOLE_REASON_ACCEPTED_STRONG = "accepted_strong"
HOLE_REASON_ACCEPTED_MODERATE = "accepted_moderate"
HOLE_REASON_NOT_BOX_STABLE = "not_box_stable"
HOLE_REASON_WEAK_CONF = "weak_conf"
HOLE_REASON_WEAK_MARGIN = "weak_margin"
HOLE_REASON_NEED_STABLE_2 = "need_stable_2"
HOLE_REASON_WEAK_TEMPORAL = "weak_temporal"
HOLE_REASON_DUPLICATE = "duplicate_label"
HOLE_REASON_UNKNOWN = "UNKNOWN"


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


def hole_accept_reason(obs: CardObservation) -> str:
    """Explain hole gate outcome without changing thresholds.

    Reasons: accepted_strong | accepted_moderate | not_box_stable |
    weak_conf | weak_margin | need_stable_2 | weak_temporal | UNKNOWN
    """
    if not obs.box_stable:
        return HOLE_REASON_NOT_BOX_STABLE

    # STRONG: high conf + high margin → accept on first quality frame.
    if (
        obs.confidence >= HOLE_STRONG_TOP
        and obs.rank_margin >= HOLE_STRONG_MARGIN
        and obs.stable_frames >= HOLE_STRONG_STABLE_FRAMES
    ):
        return HOLE_REASON_ACCEPTED_STRONG

    # Diagnose moderate-path failures first (order matches user-facing labels).
    if obs.confidence < HOLE_MIN_TOP:
        return HOLE_REASON_WEAK_CONF
    if obs.rank_margin < HOLE_MIN_MARGIN:
        return HOLE_REASON_WEAK_MARGIN
    if obs.stable_frames < HOLE_MIN_STABLE_FRAMES:
        return HOLE_REASON_NEED_STABLE_2
    if obs.temporal_agreement < HOLE_MIN_TEMPORAL:
        return HOLE_REASON_WEAK_TEMPORAL

    return HOLE_REASON_ACCEPTED_MODERATE


def _passes_hole_gate(obs: CardObservation) -> bool:
    """Adaptive hole acceptance: strong → frame 1; moderate → 2 + temporal."""
    reason = hole_accept_reason(obs)
    return reason in (
        HOLE_REASON_ACCEPTED_STRONG,
        HOLE_REASON_ACCEPTED_MODERATE,
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

    Hole cards use adaptive gates (strong → 1 frame, moderate → 2 + temporal).
    Board keeps first-frame acceptance. Failures become UNKNOWN / ??.
    Among local passers, duplicate labels keep the higher-confidence observation
    and demote the rest. Hole observations get ``accept_reason`` set for debug.
    """
    passed: List[CardObservation] = []
    for o in obs:
        if _is_hole(o, hole_slots, board_slots):
            reason = hole_accept_reason(o)
            o.accept_reason = reason
            ok = reason in (
                HOLE_REASON_ACCEPTED_STRONG,
                HOLE_REASON_ACCEPTED_MODERATE,
            )
        else:
            ok = _passes_board_gate(o)
            o.accept_reason = None
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
                if _is_hole(o, hole_slots, board_slots):
                    o.accept_reason = HOLE_REASON_DUPLICATE

    return obs
