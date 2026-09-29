"""Combine deterministic TableState.state_valid with soft Jev validate → may_act."""

from __future__ import annotations

from src.state.table import TableState

from .schemas import ValidateOutput


def may_act(table: TableState, validate_out: ValidateOutput) -> bool:
    """True only when hard state is valid and soft validate allows acting.

    Jev soft signals never revive a structurally invalid table.
    """
    return bool(table.state_valid and validate_out.soft_ok and not validate_out.wait)
