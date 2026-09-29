"""
Two-stage Jev decisions scaffold (validate → decide), gated by TableState.

Architecture:
  Vision → CardTrack/accept → TableState (authoritative state_valid)
           ↓
  Jev validate (soft / interpretive) — advisory only
           ↓
  PokerEngine analysis
           ↓
  Jev decide — only if state_valid AND soft_ok (and not wait)

Hard constraints stay deterministic in ``src.state``. Jev never overrides
structural invalidity.

Offline / replay analysis only — not live real-money RTA. Default stubs do
not call the network; an optional ``JEV_API_KEY`` env may enable a future
adapter without changing the gate contract.
"""

from .gate import may_act
from .jev_decision import decide
from .jev_state import table_to_validate_input, validate_state
from .schemas import (
    DECISION_ACTIONS,
    DecisionInput,
    DecisionOutput,
    ValidateInput,
    ValidateOutput,
)

__all__ = [
    "DECISION_ACTIONS",
    "DecisionInput",
    "DecisionOutput",
    "ValidateInput",
    "ValidateOutput",
    "decide",
    "may_act",
    "table_to_validate_input",
    "validate_state",
]
