"""Frozen I/O contracts for Jev validate + decide stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


# Coarse actions emitted by decide (engine strings are mapped into these).
DECISION_ACTIONS = frozenset({"fold", "call", "raise", "check", "wait"})


@dataclass(frozen=True)
class ValidateInput:
    """Facts for soft / interpretive state validation (advisory only)."""

    hole_labels: Sequence[str]
    board_labels: Sequence[str]
    vision_confidence: float
    uncertainty: float
    state_valid: bool
    street: Optional[str] = None


@dataclass(frozen=True)
class ValidateOutput:
    """Result of soft validation. Never overrides deterministic state_valid."""

    soft_ok: bool
    reason: str
    wait: bool
    confidence: float


@dataclass(frozen=True)
class DecisionInput:
    """Poker facts + engine analysis for the decide stage."""

    hole_labels: Sequence[str]
    board_labels: Sequence[str]
    vision_confidence: float
    uncertainty: float
    state_valid: bool
    win_prob: float
    hand_type: str
    recommendation: str
    street: Optional[str] = None
    soft_ok: bool = True
    wait: bool = False

    @classmethod
    def from_hand_result(
        cls,
        *,
        hole_labels: Sequence[str],
        board_labels: Sequence[str],
        vision_confidence: float,
        uncertainty: float,
        state_valid: bool,
        hand_result,
        street: Optional[str] = None,
        soft_ok: bool = True,
        wait: bool = False,
    ) -> "DecisionInput":
        """Build from a HandResult-like object or mapping."""
        if isinstance(hand_result, dict):
            win_prob = float(hand_result.get("win_probability", hand_result.get("win_prob", 0.0)))
            hand_type = str(hand_result.get("hand_type", ""))
            recommendation = str(hand_result.get("recommendation", ""))
        else:
            win_prob = float(getattr(hand_result, "win_probability", 0.0))
            hand_type = str(getattr(hand_result, "hand_type", ""))
            recommendation = str(getattr(hand_result, "recommendation", ""))
        return cls(
            hole_labels=list(hole_labels),
            board_labels=list(board_labels),
            vision_confidence=float(vision_confidence),
            uncertainty=float(uncertainty),
            state_valid=bool(state_valid),
            win_prob=win_prob,
            hand_type=hand_type,
            recommendation=recommendation,
            street=street,
            soft_ok=bool(soft_ok),
            wait=bool(wait),
        )


@dataclass(frozen=True)
class DecisionOutput:
    """Stub / future Jev decide result. Offline / replay oriented."""

    action: str  # fold | call | raise | check | wait
    confidence: float
    probs: Optional[Dict[str, float]] = None
    notes: Optional[str] = None
    # Optional free-form notes alias used by some callers / future adapters.
    noul: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in DECISION_ACTIONS:
            raise ValueError(f"invalid action {self.action!r}; expected one of {sorted(DECISION_ACTIONS)}")
