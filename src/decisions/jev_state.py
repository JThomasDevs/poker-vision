"""
Jev soft state validation (interpretive / advisory).

Hard structural validity stays in ``src.state`` (``TableState.state_valid``).
This stage never overrides deterministic invalidity — it only adds soft
signals (vision confidence, uncertainty) that can force wait.

Offline / heuristic stub by default. Optional future network adapter may
activate when ``JEV_API_KEY`` is set; tests and default path stay offline.
"""

from __future__ import annotations

import os
from typing import Union

from src.state.table import TableState
from src.state.types import UNKNOWN_LABEL, VISIBLE

from .schemas import ValidateInput, ValidateOutput

# Soft thresholds (heuristic stub — tune later / replace with real Jev).
VISION_CONF_MIN = 0.35
UNCERTAINTY_MAX = 0.65


def _labels_from_obs(obs_list) -> list:
    return [
        o.label
        for o in obs_list
        if getattr(o, "visibility", None) == VISIBLE
        and o.label
        and o.label != UNKNOWN_LABEL
    ]


def table_to_validate_input(table: TableState) -> ValidateInput:
    """Project authoritative TableState into ValidateInput."""
    return ValidateInput(
        hole_labels=_labels_from_obs(table.hero),
        board_labels=_labels_from_obs(table.board),
        vision_confidence=float(table.vision_confidence),
        uncertainty=float(table.uncertainty),
        state_valid=bool(table.state_valid),
        street=table.street,
    )


def _heuristic_validate(inp: ValidateInput) -> ValidateOutput:
    """Pure-Python soft checks — no network."""
    if not inp.state_valid:
        return ValidateOutput(
            soft_ok=False,
            reason="deterministic_invalid",
            wait=True,
            confidence=0.0,
        )

    conf = float(inp.vision_confidence)
    unc = float(inp.uncertainty)

    if conf < VISION_CONF_MIN:
        return ValidateOutput(
            soft_ok=False,
            reason="low_vision_confidence",
            wait=True,
            confidence=max(0.0, conf),
        )

    if unc > UNCERTAINTY_MAX:
        return ValidateOutput(
            soft_ok=False,
            reason="high_uncertainty",
            wait=True,
            confidence=max(0.0, 1.0 - unc),
        )

    # Soft confidence blends vision quality with inverse uncertainty.
    soft_conf = max(0.0, min(1.0, 0.5 * conf + 0.5 * (1.0 - unc)))
    return ValidateOutput(
        soft_ok=True,
        reason="ok",
        wait=False,
        confidence=soft_conf,
    )


def _maybe_remote_validate(inp: ValidateInput) -> ValidateOutput | None:
    """Optional future Jev API hook. Returns None to fall back to stub.

    Only considered when ``JEV_API_KEY`` is present. Default implementation
    does not call the network so offline tests always pass.
    """
    if not os.environ.get("JEV_API_KEY", "").strip():
        return None
    # Adapter placeholder: keep offline until a real client is wired.
    return None


def validate_state(table: Union[TableState, ValidateInput]) -> ValidateOutput:
    """Soft-validate table facts. Advisory only — does not mutate TableState."""
    if isinstance(table, ValidateInput):
        inp = table
    else:
        inp = table_to_validate_input(table)

    remote = _maybe_remote_validate(inp)
    if remote is not None:
        return remote
    return _heuristic_validate(inp)
