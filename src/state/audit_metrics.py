"""Pure metrics for classifier audit / confusion analysis."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    """Rows = true label, cols = predicted label."""
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            continue
        cm[index[t], index[p]] += 1
    return cm


def top_k_accuracy(
    prob_rows: Sequence[np.ndarray] | np.ndarray,
    true_indices: Sequence[int],
    k: int,
) -> float:
    """Fraction of rows where true class is among top-k predicted indices."""
    rows = np.asarray(prob_rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if len(true_indices) == 0:
        return 0.0
    k = max(1, int(k))
    hits = 0
    for row, ti in zip(rows, true_indices):
        top = np.argpartition(row, -k)[-k:]
        if int(ti) in top.tolist():
            hits += 1
    return float(hits) / float(len(true_indices))


def margin_stats(margins: Sequence[float]) -> Dict[str, float]:
    """mean / p50 / p90 of margins; empty → zeros."""
    if not margins:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(list(margins), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def pair_error_counts(
    cm: np.ndarray,
    labels: Sequence[str],
) -> List[Tuple[str, str, int]]:
    """Off-diagonal pairs sorted by count desc (a,b,count) with a < b lexicographically merged."""
    n = len(labels)
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            a, b = labels[i], labels[j]
            key = (a, b) if a <= b else (b, a)
            counts[key] = counts.get(key, 0) + c
    pairs = [(a, b, c) for (a, b), c in counts.items()]
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    return pairs
