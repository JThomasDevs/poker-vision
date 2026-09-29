"""Offline classifier audit: JSONL probs + rank/suit confusion matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification.infer import load_classifier, predict_image_dist
from src.classification.model import RANKS, SUITS
from src.detection.pipeline import default_classifier_path
from src.state.audit_metrics import (
    confusion_matrix,
    margin_stats,
    pair_error_counts,
    top_k_accuracy,
)


def _iter_labeled_images(
    root: Path,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, Path]]:
    """Walk ``root/{label}/*.png|jpg``; label = folder name (e.g. As)."""
    pairs: List[Tuple[str, Path]] = []
    if not root.is_dir():
        return pairs
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        if len(label) < 2:
            continue
        for path in sorted(label_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            pairs.append((label, path))
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs


def _split_label(label: str) -> Tuple[str, str]:
    return label[:-1], label[-1]


def run_audit(
    *,
    data_root: Path,
    checkpoint: Path,
    limit: Optional[int],
    jsonl_path: Optional[Path],
    device: Optional[str] = None,
) -> int:
    pairs = _iter_labeled_images(data_root, limit=limit)
    if not pairs:
        print(f"No images under {data_root}", file=sys.stderr)
        return 1

    torch_device = None
    if device:
        import torch

        torch_device = torch.device(device)

    model, meta, dev = load_classifier(checkpoint, device=torch_device)
    ranks = list(meta.get("ranks", RANKS))
    suits = list(meta.get("suits", SUITS))

    y_true_rank: List[str] = []
    y_pred_rank: List[str] = []
    y_true_suit: List[str] = []
    y_pred_suit: List[str] = []
    rank_prob_rows: List[np.ndarray] = []
    suit_prob_rows: List[np.ndarray] = []
    true_rank_idx: List[int] = []
    true_suit_idx: List[int] = []
    rank_margins: List[float] = []
    suit_margins: List[float] = []

    jsonl_f = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = jsonl_path.open("w", encoding="utf-8")

    try:
        for label, path in pairs:
            true_rank, true_suit = _split_label(label)
            if true_rank not in ranks or true_suit not in suits:
                continue
            dist = predict_image_dist(path, model, meta, dev)
            pred_rank = dist.top_rank()
            pred_suit = dist.top_suit()
            pred_label = dist.top_label()

            y_true_rank.append(true_rank)
            y_pred_rank.append(pred_rank)
            y_true_suit.append(true_suit)
            y_pred_suit.append(pred_suit)
            rank_prob_rows.append(dist.rank_probs.copy())
            suit_prob_rows.append(dist.suit_probs.copy())
            true_rank_idx.append(ranks.index(true_rank))
            true_suit_idx.append(suits.index(true_suit))
            rank_margins.append(dist.rank_margin())
            suit_margins.append(dist.suit_margin())

            if jsonl_f is not None:
                rec: Dict = {
                    "path": str(path),
                    "true_label": label,
                    "pred_label": pred_label,
                    "true_rank": true_rank,
                    "true_suit": true_suit,
                    "pred_rank": pred_rank,
                    "pred_suit": pred_suit,
                    "top_conf": dist.top_conf(),
                    "rank_margin": dist.rank_margin(),
                    "suit_margin": dist.suit_margin(),
                    "rank_probs": dist.rank_probs.tolist(),
                    "suit_probs": dist.suit_probs.tolist(),
                    "ranks": ranks,
                    "suits": suits,
                }
                jsonl_f.write(json.dumps(rec) + "\n")
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    n = len(y_true_rank)
    if n == 0:
        print("No evaluable samples", file=sys.stderr)
        return 1

    rank_cm = confusion_matrix(y_true_rank, y_pred_rank, ranks)
    suit_cm = confusion_matrix(y_true_suit, y_pred_suit, suits)
    top1_rank = top_k_accuracy(rank_prob_rows, true_rank_idx, k=1)
    top2_rank = top_k_accuracy(rank_prob_rows, true_rank_idx, k=2)
    top1_suit = top_k_accuracy(suit_prob_rows, true_suit_idx, k=1)
    top2_suit = top_k_accuracy(suit_prob_rows, true_suit_idx, k=2)
    # Joint top-1: both rank and suit correct
    joint_ok = sum(
        1
        for tr, pr, ts, ps in zip(y_true_rank, y_pred_rank, y_true_suit, y_pred_suit)
        if tr == pr and ts == ps
    )
    joint_top1 = float(joint_ok) / float(n)

    print(f"samples={n}  root={data_root}")
    print(f"joint top-1={joint_top1:.4f}")
    print(f"rank  top-1={top1_rank:.4f}  top-2={top2_rank:.4f}")
    print(f"suit  top-1={top1_suit:.4f}  top-2={top2_suit:.4f}")
    print(f"rank margin stats: {margin_stats(rank_margins)}")
    print(f"suit margin stats: {margin_stats(suit_margins)}")
    print("rank confusion pairs (top 10):")
    for a, b, c in pair_error_counts(rank_cm, ranks)[:10]:
        print(f"  {a}<->{b}: {c}")
    print("suit confusion pairs:")
    for a, b, c in pair_error_counts(suit_cm, suits)[:10]:
        print(f"  {a}<->{b}: {c}")
    if jsonl_path is not None:
        print(f"wrote JSONL: {jsonl_path}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Audit dual-head classifier on by_card crops"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images to score (default: all)",
    )
    p.add_argument(
        "--ul",
        action="store_true",
        help="Use data/cards/by_card_ul instead of by_card",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Classifier checkpoint (default: data/cards/models/classifier.pt)",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Write per-image full probs as JSONL",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override dataset root (else by_card / by_card_ul)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.data_root is not None:
        data_root = args.data_root
    else:
        rel = "by_card_ul" if args.ul else "by_card"
        data_root = _ROOT / "data" / "cards" / rel
    ckpt = args.checkpoint if args.checkpoint is not None else default_classifier_path()
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    return run_audit(
        data_root=data_root,
        checkpoint=ckpt,
        limit=args.limit,
        jsonl_path=args.jsonl,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
