"""
Jev decision reasoning (stub).

Maps PokerEngine analysis into a coarse DecisionOutput when the gate allows
acting. Offline / replay analysis only — not live real-money RTA.

Hard constraints remain in ``src.state``; ``decide`` must not override
``state_valid``. Callers should gate with ``may_act`` before trusting actions
other than ``wait``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Union

from src.engine.evaluator import HandResult
from src.state.table import TableState

from .gate import may_act
from .jev_state import table_to_validate_input, validate_state
from .schemas import DecisionInput, DecisionOutput, ValidateOutput


def _map_recommendation(rec: str) -> str:
    """Map engine recommendation strings onto DecisionOutput actions."""
    r = (rec or "").strip().lower()
    if not r or r in ("error", "wait", "-"):
        return "wait"
    if r in ("fold",):
        return "fold"
    if r in ("call",):
        return "call"
    if r in ("raise", "bet"):
        return "raise"
    if r.startswith("check"):
        return "check"
    # Unknown engine string → wait rather than inventing an action.
    return "wait"


def _analysis_to_fields(analysis: Union[HandResult, Mapping[str, Any], DecisionInput]):
    if isinstance(analysis, DecisionInput):
        return analysis
    if isinstance(analysis, HandResult):
        return {
            "win_prob": float(analysis.win_probability),
            "hand_type": analysis.hand_type,
            "recommendation": analysis.recommendation,
        }
    if isinstance(analysis, Mapping):
        # Accept HandResult-ish dict or already-flat DecisionInput fields.
        if "recommendation" in analysis or "win_probability" in analysis or "win_prob" in analysis:
            return {
                "win_prob": float(analysis.get("win_probability", analysis.get("win_prob", 0.0))),
                "hand_type": str(analysis.get("hand_type", "")),
                "recommendation": str(analysis.get("recommendation", "")),
            }
    raise TypeError(f"unsupported analysis type: {type(analysis)!r}")


def _build_decision_input(
    state: Union[TableState, DecisionInput, ValidateOutput],
    analysis: Union[HandResult, Mapping[str, Any], DecisionInput],
    validate_out: Optional[ValidateOutput] = None,
) -> tuple[DecisionInput, Optional[ValidateOutput], Optional[TableState]]:
    """Normalize call shapes into DecisionInput + optional gate inputs."""
    if isinstance(analysis, DecisionInput) and not isinstance(state, TableState):
        din = analysis
        return din, validate_out, None

    table: Optional[TableState] = state if isinstance(state, TableState) else None

    if isinstance(analysis, DecisionInput):
        din = analysis
    elif table is not None:
        vin = table_to_validate_input(table)
        fields = _analysis_to_fields(analysis)
        assert isinstance(fields, dict)
        vout = validate_out if validate_out is not None else validate_state(table)
        din = DecisionInput(
            hole_labels=list(vin.hole_labels),
            board_labels=list(vin.board_labels),
            vision_confidence=vin.vision_confidence,
            uncertainty=vin.uncertainty,
            state_valid=vin.state_valid,
            win_prob=fields["win_prob"],
            hand_type=fields["hand_type"],
            recommendation=fields["recommendation"],
            street=vin.street,
            soft_ok=vout.soft_ok,
            wait=vout.wait,
        )
        return din, vout, table
    else:
        # state is ValidateOutput or unused; analysis must carry poker facts.
        raise TypeError(
            "decide(state, analysis) expects TableState + HandResult/dict, "
            "or a DecisionInput as analysis"
        )

    return din, validate_out, table


def _maybe_remote_decide(din: DecisionInput) -> Optional[DecisionOutput]:
    """Optional future Jev API hook. Offline unless adapter is implemented."""
    if not os.environ.get("JEV_API_KEY", "").strip():
        return None
    return None


def decide(
    state: Union[TableState, DecisionInput],
    analysis: Union[HandResult, Mapping[str, Any], DecisionInput, None] = None,
    *,
    validate_out: Optional[ValidateOutput] = None,
) -> DecisionOutput:
    """Produce a DecisionOutput. Returns action=wait when the gate blocks.

    Preferred call: ``decide(table, hand_result)`` or ``decide(table, hand_result, validate_out=...)``.
    Also accepts a pre-built ``DecisionInput`` as ``state`` (analysis optional).
    """
    if analysis is None:
        if not isinstance(state, DecisionInput):
            raise TypeError("decide requires analysis unless state is DecisionInput")
        din = state
        table = None
        vout = validate_out
        if vout is None:
            # Reconstruct a minimal ValidateOutput from DecisionInput flags.
            vout = ValidateOutput(
                soft_ok=din.soft_ok,
                reason="from_decision_input",
                wait=din.wait,
                confidence=1.0 if din.soft_ok and not din.wait else 0.0,
            )
    else:
        din, vout, table = _build_decision_input(state, analysis, validate_out)
        if vout is None:
            vout = ValidateOutput(
                soft_ok=din.soft_ok,
                reason="from_decision_input",
                wait=din.wait,
                confidence=1.0 if din.soft_ok and not din.wait else 0.0,
            )

    # Gate: need state_valid + soft_ok + not wait.
    if table is not None:
        allowed = may_act(table, vout)
    else:
        allowed = bool(din.state_valid and vout.soft_ok and not vout.wait)

    if not allowed:
        return DecisionOutput(
            action="wait",
            confidence=float(vout.confidence),
            notes=f"gated:{vout.reason}",
        )

    remote = _maybe_remote_decide(din)
    if remote is not None:
        return remote

    action = _map_recommendation(din.recommendation)
    conf = max(0.0, min(1.0, float(din.win_prob)))
    return DecisionOutput(
        action=action,
        confidence=conf,
        probs={action: conf} if action != "wait" else None,
        notes=f"stub_map:{din.recommendation}",
    )
