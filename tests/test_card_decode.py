"""Unit tests for YOLO class_id → card decode (rank-major, matches cards.yaml)."""

from src.detection.cards import CARD_NAMES, decode_class_id


def test_card_names_length_and_order():
    assert len(CARD_NAMES) == 52
    assert CARD_NAMES[0] == "2h"
    assert CARD_NAMES[1] == "2d"
    assert CARD_NAMES[2] == "2c"
    assert CARD_NAMES[3] == "2s"
    assert CARD_NAMES[4] == "3h"
    assert CARD_NAMES[51] == "As"


def test_decode_class_id_rank_major():
    assert decode_class_id(0) == ("2", "h")
    assert decode_class_id(1) == ("2", "d")
    assert decode_class_id(3) == ("2", "s")
    assert decode_class_id(4) == ("3", "h")
    assert decode_class_id(32) == ("T", "h")
    assert decode_class_id(35) == ("T", "s")
    assert decode_class_id(48) == ("A", "h")
    assert decode_class_id(51) == ("A", "s")


def test_decode_class_id_out_of_range():
    assert decode_class_id(-1) == ("?", "?")
    assert decode_class_id(52) == ("?", "?")


def test_decode_differs_from_suit_major_bug():
    """Old formula used cls%13 / cls//13 (suit-major); class 1 must be 2d, not 3h."""
    rank, suit = decode_class_id(1)
    assert (rank, suit) == ("2", "d")
    assert (rank, suit) != ("3", "h")
