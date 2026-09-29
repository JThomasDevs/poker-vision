"""Tests for decisions gate, schemas, and stub validate/decide."""

from __future__ import annotations

import numpy as np
import pytest

from src.decisions import (
    DECISION_ACTIONS,
    DecisionInput,
    DecisionOutput,
    ValidateInput,
    ValidateOutput,
    decide,
    may_act,
    validate_state,
)
from src.decisions.jev_state import VISION_CONF_MIN, UNCERTAINTY_MAX
from src.engine.evaluator import HandResult
from src.state.table import TableState
from src.state.types import VISIBLE, CardObservation


def _obs(slot: str, label: str, conf: float = 0.9) -> CardObservation:
    rp = np.ones(13)
    sp = np.ones(4)
    o = CardObservation(
        slot_id=slot,
        bbox=(0, 0, 10, 20),
        rank_probs=rp,
        suit_probs=sp,
        visibility=VISIBLE,
        label=label,
        confidence=conf,
    )
    return o


def _table(
    *,
    state_valid: bool = True,
    vision_confidence: float = 0.8,
    uncertainty: float = 0.2,
    street: str = "flop",
) -> TableState:
    return TableState(
        hero=[_obs("hero_0", "As"), _obs("hero_1", "Kh")],
        board=[_obs("board_0", "2c"), _obs("board_1", "7d"), _obs("board_2", "9s")],
        state_valid=state_valid,
        vision_confidence=vision_confidence,
        uncertainty=uncertainty,
        street=street,
    )


def test_validate_output_schema_fields():
    out = ValidateOutput(soft_ok=True, reason="ok", wait=False, confidence=0.9)
    assert out.soft_ok is True
    assert out.reason == "ok"
    assert out.wait is False
    assert out.confidence == 0.9


def test_decision_output_rejects_bad_action():
    with pytest.raises(ValueError):
        DecisionOutput(action="allin", confidence=0.5)


def test_decision_actions_contract():
    assert DECISION_ACTIONS == frozenset({"fold", "call", "raise", "check", "wait"})


def test_validate_deterministic_invalid():
    table = _table(state_valid=False, vision_confidence=0.99, uncertainty=0.01)
    out = validate_state(table)
    assert out.soft_ok is False
    assert out.wait is True
    assert out.reason == "deterministic_invalid"
    assert may_act(table, out) is False


def test_validate_low_vision_confidence():
    table = _table(state_valid=True, vision_confidence=VISION_CONF_MIN - 0.05, uncertainty=0.1)
    out = validate_state(table)
    assert out.soft_ok is False
    assert out.wait is True
    assert out.reason == "low_vision_confidence"
    assert may_act(table, out) is False


def test_validate_high_uncertainty():
    table = _table(state_valid=True, vision_confidence=0.9, uncertainty=UNCERTAINTY_MAX + 0.05)
    out = validate_state(table)
    assert out.soft_ok is False
    assert out.wait is True
    assert out.reason == "high_uncertainty"
    assert may_act(table, out) is False


def test_validate_ok_and_may_act():
    table = _table(state_valid=True, vision_confidence=0.8, uncertainty=0.2)
    out = validate_state(table)
    assert out.soft_ok is True
    assert out.wait is False
    assert out.reason == "ok"
    assert may_act(table, out) is True


def test_validate_accepts_validate_input():
    inp = ValidateInput(
        hole_labels=["As", "Kh"],
        board_labels=["2c"],
        vision_confidence=0.8,
        uncertainty=0.2,
        state_valid=True,
        street="flop",
    )
    out = validate_state(inp)
    assert out.soft_ok is True


def test_may_act_requires_all_three():
    table = _table(state_valid=True)
    assert may_act(table, ValidateOutput(True, "ok", False, 1.0)) is True
    assert may_act(table, ValidateOutput(True, "ok", True, 1.0)) is False
    assert may_act(table, ValidateOutput(False, "x", False, 1.0)) is False
    bad = _table(state_valid=False)
    assert may_act(bad, ValidateOutput(True, "ok", False, 1.0)) is False


def test_decide_waits_when_gated():
    table = _table(state_valid=False)
    hr = HandResult("Pair", 100, 0.8, "raise", 0.5)
    out = decide(table, hr)
    assert out.action == "wait"
    assert out.notes and out.notes.startswith("gated:")


def test_decide_maps_engine_recommendation():
    table = _table(state_valid=True, vision_confidence=0.9, uncertainty=0.1)
    hr = HandResult("Pair", 100, 0.72, "raise", 0.5)
    out = decide(table, hr)
    assert out.action == "raise"
    assert out.confidence == pytest.approx(0.72)


def test_decide_maps_check_call_to_check():
    table = _table(state_valid=True, vision_confidence=0.9, uncertainty=0.1)
    hr = HandResult("High Card", 500, 0.5, "check/call", 0.1)
    out = decide(table, hr)
    assert out.action == "check"


def test_decide_maps_bet_to_raise():
    table = _table(state_valid=True, vision_confidence=0.9, uncertainty=0.1)
    hr = HandResult("Two Pair", 50, 0.6, "bet", 0.4)
    out = decide(table, hr)
    assert out.action == "raise"


def test_decide_from_decision_input():
    din = DecisionInput(
        hole_labels=["As", "Kh"],
        board_labels=[],
        vision_confidence=0.9,
        uncertainty=0.1,
        state_valid=True,
        win_prob=0.3,
        hand_type="Pre-flop",
        recommendation="fold",
        soft_ok=True,
        wait=False,
    )
    out = decide(din)
    assert out.action == "fold"


def test_decide_hand_result_dict():
    table = _table(state_valid=True, vision_confidence=0.9, uncertainty=0.1)
    out = decide(
        table,
        {
            "hand_type": "Pair",
            "win_probability": 0.4,
            "recommendation": "call",
        },
    )
    assert out.action == "call"


def test_decision_input_from_hand_result():
    hr = HandResult("Pair", 100, 0.55, "bet", 0.3)
    din = DecisionInput.from_hand_result(
        hole_labels=["As", "Kd"],
        board_labels=["2c", "2d", "9s"],
        vision_confidence=0.8,
        uncertainty=0.2,
        state_valid=True,
        hand_result=hr,
        street="flop",
    )
    assert din.win_prob == pytest.approx(0.55)
    assert din.recommendation == "bet"
    assert din.street == "flop"
