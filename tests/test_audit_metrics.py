"""Tests for audit_metrics pure helpers."""

from __future__ import annotations

import numpy as np

from src.state.audit_metrics import (
    confusion_matrix,
    margin_stats,
    pair_error_counts,
    top_k_accuracy,
)


def test_confusion_matrix_basic():
    labels = ["6", "9", "A"]
    cm = confusion_matrix(
        y_true=["6", "9", "6", "A"],
        y_pred=["9", "9", "6", "A"],
        labels=labels,
    )
    assert cm.shape == (3, 3)
    assert int(cm[0, 1]) == 1  # true 6 pred 9
    assert int(cm[1, 1]) == 1
    assert int(cm[0, 0]) == 1
    assert int(cm[2, 2]) == 1


def test_top_k_accuracy():
    # probs [0.2, 0.5, 0.3] → top1=idx1, top2={1,2}
    rows = [np.array([0.2, 0.5, 0.3])]
    assert top_k_accuracy(rows, [0], k=1) == 0.0
    assert top_k_accuracy(rows, [2], k=1) == 0.0
    assert top_k_accuracy(rows, [2], k=2) == 1.0
    assert top_k_accuracy(rows, [1], k=1) == 1.0


def test_margin_stats():
    s = margin_stats([0.1, 0.2, 0.3, 0.4])
    assert abs(s["mean"] - 0.25) < 1e-9
    assert "p50" in s and "p90" in s
    assert margin_stats([]) == {"mean": 0.0, "p50": 0.0, "p90": 0.0}


def test_pair_error_counts_merges_direction():
    labels = ["6", "9"]
    cm = np.array([[0, 3], [2, 0]], dtype=np.int64)
    pairs = pair_error_counts(cm, labels)
    assert pairs[0] == ("6", "9", 5)
