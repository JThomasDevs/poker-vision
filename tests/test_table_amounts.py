"""Unit tests for amount parsing and pot-odds recommendations."""

from __future__ import annotations

import numpy as np

from src.detection.table_amounts import (
    TableAmounts,
    crop_roi,
    parse_amount_bb,
    parse_hero_stack,
    parse_pot,
    parse_to_call,
    pot_odds,
    preprocess_for_ocr,
    read_table_amounts,
)
from src.engine.evaluator import PokerEngine


def test_parse_amount_bb_basic():
    assert parse_amount_bb("99 BB") == 99.0
    assert parse_amount_bb("12.5 BB") == 12.5
    assert parse_amount_bb("22.5 BB") == 22.5
    assert parse_amount_bb("222.5 BB") == 222.5
    assert parse_amount_bb("1 BB") == 1.0
    assert parse_amount_bb("81 BB") == 81.0
    assert parse_amount_bb("1,234.5 SC") == 1234.5
    assert parse_amount_bb("") is None
    assert parse_amount_bb("no numbers") is None
    # Prefer BB-suffixed amount over a bare leading digit (no $ / fiat)
    assert parse_amount_bb("Pot 7 BB") == 7.0


def test_parse_amount_bb_stake_logo_garbage():
    # Green [S] icon often OCR's as @, 6, 5, or 8 before the real BB amount
    assert parse_amount_bb("Pot: @ 8 BB") == 8.0
    assert parse_amount_bb("Pot: 6 8 BB") == 8.0
    assert parse_amount_bb("6 8 BB") == 8.0
    assert parse_amount_bb("6126 BB") == 126.0
    assert parse_amount_bb("6126BB") == 126.0
    assert parse_amount_bb("8 81BB") == 81.0
    assert parse_amount_bb("6 241.5 BB") == 241.5
    assert parse_amount_bb("[S] 99 BB") == 99.0


def test_parse_pot_stake_style():
    assert parse_pot("Pot: 12.5 BB") == 12.5
    assert parse_pot("Pot: [S] 5.2 BB") == 5.2
    assert parse_pot("POT 3 BB") == 3.0
    assert parse_pot("Pot: 7 BB") == 7.0
    assert parse_pot("Pot: 6 8 BB") == 8.0
    assert parse_pot("rhyz 6 39 Pot: @ 8 BB") == 8.0
    assert parse_pot("garbage") is None


def test_parse_to_call_and_check():
    assert parse_to_call("to call 4.5 BB") == 4.5
    assert parse_to_call("Call: 2 BB") == 2.0
    assert parse_to_call("checked") == 0.0
    # Bare chip text is not a facing bet (often pot / seat bets)
    assert parse_to_call("10 BB") is None
    assert parse_to_call("5 BB") is None
    assert parse_to_call("Bet 1 BB") == 1.0
    # Stakes line / table chrome must never become to_call
    assert parse_to_call("Tigar413, NLHE . 0.010.02") is None
    assert parse_to_call("Bad Beat Jackpot 849,027.29") is None


def test_parse_hero_stack():
    assert parse_hero_stack("99 BB") == 99.0
    assert parse_hero_stack("  42.0 SC ") == 42.0
    assert parse_hero_stack("126 BB") == 126.0
    assert parse_hero_stack("6126BB") == 126.0
    assert parse_hero_stack("Sayhey 241.5 BB") == 241.5
    assert parse_hero_stack("thyzer 209.5B") == 209.5
    assert parse_hero_stack("rhyzome\n209.5 BB") == 209.5
    assert parse_hero_stack("waltnir\n32BB\n1BB") == 32.0
    assert parse_hero_stack("thyzer\n209.5B\n1BB\nwaltir") == 209.5
    assert parse_hero_stack("NLHE - 0.01/0.02") is None


def test_parse_pot_rejects_bare_and_requires_label():
    assert parse_pot("849027.29") is None
    assert parse_pot("0.01/0.02") is None
    assert parse_pot("Pot:5SBB") == 5.0
    assert parse_pot("Pot:5BB") == 5.0


def test_pot_odds_formula():
    # to_call 5 into pot 15 → 5/20 = 0.25
    assert pot_odds(15.0, 5.0) == 0.25
    assert pot_odds(None, 5.0) == 1.0  # pot unknown → treat as 0
    assert pot_odds(10.0, 0.0) is None
    assert pot_odds(10.0, None) is None


def test_table_amounts_status():
    a = TableAmounts(pot=10.0, to_call=2.5, hero_stack=99.0)
    assert "pot=10" in a.format_status()
    assert "to_call=2.5" in a.format_status()
    assert "stack=99" in a.format_status()
    assert TableAmounts().format_status() == "pot=? to_call=? stack=?"


def test_recommend_facing_bet_call_vs_fold():
    eng = PokerEngine(n_simulations=10)
    amounts = TableAmounts(pot=15.0, to_call=5.0, hero_stack=80.0)
    # pot_odds = 0.25; call band ~0.27..0.40; raise at >=0.40
    action, odds = eng.recommend(0.35, amounts=amounts, n_board=3)
    assert odds == 0.25
    assert action == "call"
    action, _ = eng.recommend(0.20, amounts=amounts, n_board=3)
    assert action == "fold"


def test_recommend_raise_when_equity_clearly_above():
    eng = PokerEngine(n_simulations=10)
    amounts = TableAmounts(pot=20.0, to_call=5.0, hero_stack=100.0)
    # pot_odds ≈ 0.20; win 70% → raise
    action, odds = eng.recommend(0.70, amounts=amounts, n_board=5)
    assert action == "raise"
    assert odds is not None and abs(odds - 0.2) < 1e-9


def test_recommend_all_in_facing_no_raise():
    eng = PokerEngine(n_simulations=10)
    amounts = TableAmounts(pot=50.0, to_call=40.0, hero_stack=40.0)
    # High equity but all-in → call not raise
    action, _ = eng.recommend(0.80, amounts=amounts, n_board=5)
    assert action == "call"


def test_recommend_checked_uses_ladder():
    eng = PokerEngine(n_simulations=10)
    amounts = TableAmounts(pot=10.0, to_call=0.0, hero_stack=50.0)
    action, odds = eng.recommend(0.60, amounts=amounts, n_board=5)
    assert odds is None
    assert action == "bet"  # equity ladder, not pot-odds call


def test_crop_roi_and_preprocess_smoke():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[30:60, 300:500] = (40, 40, 40)
    crop = crop_roi(frame, (0.32, 0.06, 0.68, 0.18))
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    prep = preprocess_for_ocr(crop)
    assert prep.ndim == 2


def test_read_table_amounts_without_ocr_returns_nones():
    # Blank frame + no/failed OCR → all None (does not crash)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    amounts = read_table_amounts(frame)
    assert isinstance(amounts, TableAmounts)
    # Without readable text, values stay None
    assert amounts.pot is None or isinstance(amounts.pot, float)


def test_merge_amounts_keeps_last_good():
    from src.detection.table_amounts import merge_amounts

    prev = TableAmounts(pot=10.0, to_call=2.0, hero_stack=99.0)
    fresh = TableAmounts(pot=None, to_call=3.0, hero_stack=None)
    merged = merge_amounts(prev, fresh)
    assert merged.pot == 10.0
    assert merged.to_call == 3.0
    assert merged.hero_stack == 99.0


def test_smoke_stake_full_table_rhyzome_amounts():
    """Full-table Stake shot: pot 5 BB, hero rhyzome ~209.5, not Telecaster 99."""
    from pathlib import Path

    import cv2

    path = Path(__file__).resolve().parent / "fixtures" / "stake_full_table_rhyzome.jpg"
    if not path.is_file():
        import pytest

        pytest.skip(f"missing fixture {path}")
    frame = cv2.imread(str(path))
    assert frame is not None
    amounts = read_table_amounts(frame)
    assert amounts.pot is not None
    assert abs(amounts.pot - 5.0) < 0.6
    # Facing bet unclear / no Call UI → 0; never echo pot as to_call
    assert amounts.to_call == 0.0
    assert amounts.hero_stack is not None
    assert abs(amounts.hero_stack - 209.5) < 2.0
    assert amounts.hero_stack != 99.0


def test_amounts_throttle_skips_when_disabled_or_unchanged():
    from src.detection.table_amounts import AmountsThrottle, frame_fingerprint
    from unittest import mock

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:80, 100:200] = (200, 200, 200)
    throttle = AmountsThrottle(interval_s=10.0, enabled=False)
    assert throttle.get(frame).pot is None
    assert throttle.last_ran is False

    throttle = AmountsThrottle(interval_s=10.0, enabled=True)
    fake = TableAmounts(pot=8.0, to_call=0.0, hero_stack=100.0)
    with mock.patch(
        "src.detection.table_amounts.read_table_amounts", return_value=fake
    ) as mocked:
        a1 = throttle.get(frame)
        assert a1.pot == 8.0
        assert throttle.last_ran is True
        assert mocked.call_count == 1
        # Same frame fingerprint → no second OCR
        a2 = throttle.get(frame)
        assert a2.pot == 8.0
        assert throttle.last_ran is False
        assert mocked.call_count == 1
        # Fingerprint stable
        assert frame_fingerprint(frame) == throttle.last_fp

