"""Assemble validated TableState from accepted card observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .types import (
    SLOT_BOARD,
    SLOT_HERO,
    UNKNOWN_LABEL,
    VISIBLE,
    CardObservation,
)


@dataclass
class TableState:
    """Frame-level table view after tracking + acceptance."""

    hero: List[CardObservation] = field(default_factory=list)
    board: List[CardObservation] = field(default_factory=list)
    state_valid: bool = False
    vision_confidence: float = 0.0
    uncertainty: float = 1.0
    street: Optional[str] = None


def _slot_order(slot_id: str, ordered: Sequence[str]) -> int:
    try:
        return list(ordered).index(slot_id)
    except ValueError:
        return 10_000


def _street_hint(board: Sequence[CardObservation]) -> Optional[str]:
    n = sum(1 for o in board if o.visibility == VISIBLE and o.label != UNKNOWN_LABEL)
    if n == 0:
        return "preflop"
    if n == 3:
        return "flop"
    if n == 4:
        return "turn"
    if n == 5:
        return "river"
    return None


def build_table_state(
    obs: Sequence[CardObservation],
    *,
    hole_slots: Sequence[str] = SLOT_HERO,
    board_slots: Sequence[str] = SLOT_BOARD,
    board_detected: Optional[bool] = None,
) -> TableState:
    """Partition observations into hero/board and compute validity metrics.

    ``state_valid`` is True iff both hero cards are VISIBLE with real labels.
    Unresolved board slots (UNKNOWN/TRANSITIONING) do not invalidate the table;
    callers should emit only VISIBLE board labels for equity/display.
    ``board_detected`` is retained for API compatibility and ignored for validity.
    """
    hole_set = set(hole_slots)
    board_set = set(board_slots)

    hero = sorted(
        [o for o in obs if o.slot_id in hole_set],
        key=lambda o: _slot_order(o.slot_id, hole_slots),
    )
    board = sorted(
        [o for o in obs if o.slot_id in board_set],
        key=lambda o: _slot_order(o.slot_id, board_slots),
    )

    hero_ok = [
        o
        for o in hero
        if o.visibility == VISIBLE
        and o.label
        and o.label != UNKNOWN_LABEL
        and len(o.label) >= 2
    ]
    # Soft board: hero VISIBLE is enough; unresolved board slots are omitted downstream.
    state_valid = len(hero_ok) >= 2

    visible = [
        o
        for o in list(hero) + list(board)
        if o.visibility == VISIBLE and o.label != UNKNOWN_LABEL
    ]
    if visible:
        vision_confidence = float(sum(o.confidence for o in visible) / len(visible))
    else:
        vision_confidence = 0.0

    all_slots = list(hero) + list(board)
    if all_slots:
        # High when margins/conf are weak or visibility is not VISIBLE.
        unc_parts: List[float] = []
        for o in all_slots:
            if o.visibility != VISIBLE:
                unc_parts.append(1.0)
            else:
                unc_parts.append(max(0.0, 1.0 - float(o.confidence)))
        uncertainty = float(sum(unc_parts) / len(unc_parts))
    else:
        uncertainty = 1.0

    return TableState(
        hero=hero,
        board=board,
        state_valid=state_valid,
        vision_confidence=vision_confidence,
        uncertainty=uncertainty,
        street=_street_hint(board),
    )
