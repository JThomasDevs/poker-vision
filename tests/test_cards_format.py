"""Tests for user-facing card label formatting."""

from src.util.cards_format import (
    format_card_label,
    format_cards_list,
    use_unicode_suits,
)


def test_format_card_label_unicode():
    use_unicode_suits(force=True)
    assert format_card_label("Kc") == "K♣"
    assert format_card_label("7s") == "7♠"
    assert format_card_label("Ah") == "A♥"
    assert format_card_label("Td") == "T♦"


def test_format_card_label_ascii_fallback():
    use_unicode_suits(force=False)
    assert format_card_label("Kc") == "K[c]"
    assert format_card_label("7s") == "7[s]"


def test_format_card_label_passthrough():
    use_unicode_suits(force=True)
    assert format_card_label("??") == "??"
    assert format_card_label("") == ""
    assert format_card_label("X") == "X"
    assert format_card_label("Kx") == "Kx"


def test_format_cards_list():
    use_unicode_suits(force=True)
    assert format_cards_list(["Kc", "7s"]) == "K♣ 7♠"
    assert format_cards_list(None, empty="N/A") == "N/A"
    assert format_cards_list([], empty="--") == "--"
