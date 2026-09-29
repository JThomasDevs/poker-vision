"""Train dual-head rank+suit card classifier from data/cards/by_card/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow `python src/classification/train_classifier.py` and `python -m ...`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification.dataset import (
    CardCropDataset,
    build_eval_transform,
    build_train_transform,
    discover_samples,
    stratified_holdout_split,
)
from src.classification.model import (
    CARD_NAMES,
    RANKS,
    SUITS,
    DualHeadCardClassifier,
    load_card_names,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate(model: DualHeadCardClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    n = 0
    rank_ok = 0
    suit_ok = 0
    card_ok = 0
    for images, ranks, suits in loader:
        images = images.to(device)
        ranks = ranks.to(device)
        suits = suits.to(device)
        rank_logits, suit_logits = model(images)
        pred_r = rank_logits.argmax(dim=-1)
        pred_s = suit_logits.argmax(dim=-1)
        rank_ok += (pred_r == ranks).sum().item()
        suit_ok += (pred_s == suits).sum().item()
        card_ok += ((pred_r == ranks) & (pred_s == suits)).sum().item()
        n += images.size(0)
    if n == 0:
        return {"n": 0, "rank_acc": 0.0, "suit_acc": 0.0, "card_acc": 0.0}
    return {
        "n": n,
        "rank_acc": rank_ok / n,
        "suit_acc": suit_ok / n,
        "card_acc": card_ok / n,
    }


def train_one_epoch(
    model: DualHeadCardClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank_crit: nn.Module,
    suit_crit: nn.Module,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, ranks, suits in loader:
        images = images.to(device)
        ranks = ranks.to(device)
        suits = suits.to(device)
        optimizer.zero_grad(set_to_none=True)
        rank_logits, suit_logits = model(images)
        loss = rank_crit(rank_logits, ranks) + suit_crit(suit_logits, suits)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train dual-head card crop classifier")
    root = project_root()
    p.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[root / "data" / "cards" / "by_card_ul", root / "data" / "cards" / "by_card"],
        help="One or more roots with {rank}{suit}/*.png (UL crops preferred)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "data" / "cards" / "models" / "classifier.pt",
        help="Checkpoint path",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Label map JSON (default: next to --out as label_map.json)",
    )
    p.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    return p.parse_args(argv)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _oversample_min_per_card(samples, min_per_card: int, seed: int):
    import random

    rng = random.Random(seed)
    by_card: dict = {}
    for s in samples:
        by_card.setdefault(s[1], []).append(s)
    out = list(samples)
    for card, items in by_card.items():
        need = min_per_card - len(items)
        for _ in range(max(0, need)):
            out.append(items[rng.randrange(len(items))])
    rng.shuffle(out)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    card_names = load_card_names()
    samples: list = []
    for data_root in args.data:
        found = discover_samples(data_root, card_names)
        print(f"  {data_root}: {len(found)} images")
        samples.extend(found)
    if not samples:
        print(f"No images under {args.data}", file=sys.stderr)
        return 1

    # Deduplicate by path
    seen = set()
    uniq = []
    for s in samples:
        key = str(s[0].resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    samples = uniq

    train_s, val_s = stratified_holdout_split(samples, val_frac=args.val_frac, seed=args.seed)
    # Oversample rare cards in train so 3h/5s/9s aren't drowned by common crops.
    train_s = _oversample_min_per_card(train_s, min_per_card=20, seed=args.seed)
    print(
        f"samples={len(samples)} train={len(train_s)} val={len(val_s)} "
        f"cards_with_data={len({s[1] for s in samples})}/52"
    )

    train_ds = CardCropDataset(train_s, transform=build_train_transform(args.image_size))
    val_ds = CardCropDataset(val_s, transform=build_eval_transform(args.image_size))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(args.device)
    print(f"device={device} backbone={args.backbone} epochs={args.epochs}")

    model = DualHeadCardClassifier(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    rank_counts = torch.zeros(len(RANKS), dtype=torch.float32)
    suit_counts = torch.zeros(len(SUITS), dtype=torch.float32)
    for _, _, ri, si in train_s:
        rank_counts[ri] += 1
        suit_counts[si] += 1
    rank_w = (1.0 / rank_counts.clamp(min=1.0))
    rank_w = rank_w / rank_w.sum() * len(RANKS)
    # Extra boost for digit-like ranks that the index corner confuses.
    for r in ("3", "5", "6", "8", "9", "T"):
        if r in RANKS:
            rank_w[RANKS.index(r)] *= 1.35
    rank_w = rank_w / rank_w.sum() * len(RANKS)
    suit_w = (1.0 / suit_counts.clamp(min=1.0))
    suit_w = suit_w / suit_w.sum() * len(SUITS)
    rank_crit = nn.CrossEntropyLoss(weight=rank_w.to(device))
    suit_crit = nn.CrossEntropyLoss(weight=suit_w.to(device))

    best_card_acc = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, rank_crit, suit_crit)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, **metrics})
        print(
            f"epoch {epoch:02d}/{args.epochs}  loss={loss:.4f}  "
            f"val_card={metrics['card_acc']:.3f}  "
            f"rank={metrics['rank_acc']:.3f}  suit={metrics['suit_acc']:.3f}  "
            f"n={metrics['n']}"
        )
        if metrics["card_acc"] >= best_card_acc:
            best_card_acc = metrics["card_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    label_map_path = args.label_map or args.out.with_name("label_map.json")

    label_map = {
        "ranks": RANKS,
        "suits": SUITS,
        "card_names": card_names if card_names else CARD_NAMES,
        "backbone": args.backbone,
        "image_size": args.image_size,
    }
    label_map_path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")

    final_metrics = evaluate(model, val_loader, device)
    ckpt = {
        "model": model.state_dict(),
        "backbone": args.backbone,
        "ranks": RANKS,
        "suits": SUITS,
        "card_names": label_map["card_names"],
        "image_size": args.image_size,
        "metrics": final_metrics,
        "history": history,
        "best_card_acc": best_card_acc,
        "train_n": len(train_s),
        "val_n": len(val_s),
        "device_trained": str(device),
    }
    torch.save(ckpt, args.out)
    print(f"saved {args.out}")
    print(f"saved {label_map_path}")
    print(
        f"holdout card_acc={final_metrics['card_acc']:.4f} "
        f"rank_acc={final_metrics['rank_acc']:.4f} "
        f"suit_acc={final_metrics['suit_acc']:.4f} "
        f"(n={final_metrics['n']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
