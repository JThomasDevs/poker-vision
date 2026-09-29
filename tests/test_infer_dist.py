"""Tests for ClassifierDist APIs (no checkpoint required)."""

from __future__ import annotations

import numpy as np

from src.classification.infer import ClassifierDist
from src.classification.model import RANKS, SUITS


def test_classifier_dist_shapes_and_sum():
    rp = np.zeros(13)
    rp[7] = 0.8
    rp[5] = 0.2
    sp = np.zeros(4)
    sp[3] = 0.9
    sp[0] = 0.1
    d = ClassifierDist(rank_probs=rp, suit_probs=sp, ranks=list(RANKS), suits=list(SUITS))
    assert d.rank_probs.shape == (13,)
    assert d.suit_probs.shape == (4,)
    assert abs(float(d.rank_probs.sum()) - 1.0) < 1e-9
    assert abs(float(d.suit_probs.sum()) - 1.0) < 1e-9
    assert d.top_label() == "9s"
    assert d.top_conf() == d.rank_probs[7] * d.suit_probs[3]
    assert d.rank_margin() > 0.5
    assert d.as_tuple()[0] == "9s"[0] or d.as_tuple() == ("9", "s", d.top_conf())


def test_classifier_dist_normalizes_unnormalized():
    d = ClassifierDist(
        rank_probs=np.ones(13) * 2.0,
        suit_probs=np.ones(4) * 5.0,
        ranks=list(RANKS),
        suits=list(SUITS),
    )
    assert abs(float(d.rank_probs.sum()) - 1.0) < 1e-9
    assert abs(float(d.suit_probs.sum()) - 1.0) < 1e-9
