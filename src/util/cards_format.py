"""Format internal card labels (e.g. Kc) for user-facing display."""

from __future__ import annotations

import sys
from typing import Iterable, List, Optional

_SUIT_UNICODE = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
_SUIT_ASCII = {"h": "[h]", "d": "[d]", "c": "[c]", "s": "[s]"}

_use_unicode: Optional[bool] = None


def _stdout_supports_unicode_suits() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        for ch in _SUIT_UNICODE.values():
            ch.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def use_unicode_suits(force: Optional[bool] = None) -> bool:
    """Whether to render suit symbols as Unicode (else [h]-style fallback)."""
    global _use_unicode
    if force is not None:
        _use_unicode = force
    if _use_unicode is None:
        _use_unicode = _stdout_supports_unicode_suits()
    return _use_unicode


def format_card_label(label: str, *, unicode_suits: Optional[bool] = None) -> str:
    """
    Convert pipeline label like ``Kc`` or ``7s`` to ``K♣`` / ``7♠``.
    Unknown or placeholder labels are returned unchanged.
    """
    if not label or label == "??":
        return label
    if len(label) < 2:
        return label

    rank, suit_char = label[:-1], label[-1].lower()
    if suit_char not in _SUIT_UNICODE:
        return label

    suits = _SUIT_UNICODE if use_unicode_suits(unicode_suits) else _SUIT_ASCII
    return f"{rank}{suits[suit_char]}"


def format_cards_list(
    labels: Optional[Iterable[str]],
    *,
    separator: str = " ",
    empty: str = "",
    unicode_suits: Optional[bool] = None,
) -> str:
    """Format a sequence of card labels for display."""
    if not labels:
        return empty
    parts = [format_card_label(l, unicode_suits=unicode_suits) for l in labels]
    return separator.join(parts)
