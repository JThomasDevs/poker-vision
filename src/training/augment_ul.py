"""
Generate upper-left (UL) / stacked-hole-card occlusion variants.

Hole cards on Stake often sit stacked so only the top-left index
(rank + small suit) stays visible. This script reads crops from
data/cards/by_card/{cid}/*.png and writes occluded copies into
data/cards/by_card_ul/{cid}/ — bottom 40–70% painted out, slight XY shift.

Usage (from project root):
  python -m src.training.augment_ul
  python -m src.training.augment_ul --variants 4 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.training.card_names import CARD_NAMES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
# Bottom fraction hidden (stack cover). Keep top 30–60% of the card.
HIDE_BOTTOM_MIN = 0.40
HIDE_BOTTOM_MAX = 0.70
# Horizontal/vertical shift as fraction of width/height (pixels after shift are
# filled with the same occlusion color so the canvas size stays fixed).
SHIFT_X_FRAC = 0.04
SHIFT_Y_FRAC = 0.03
# Felt-like fill for the hidden region (BGR). Soft blue-gray, not pure black.
OCCLUSION_BGR = (95, 72, 48)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cards_root(root: Optional[Path] = None) -> Path:
    return (root or project_root()) / "data" / "cards"


def list_source_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def missing_source_cards(
    by_card: Path,
    check: Sequence[str] = ("Qd", "5s"),
) -> List[str]:
    """Card ids that have no real PNG/JPG crops in by_card (folders alone do not count)."""
    missing: List[str] = []
    for cid in check:
        folder = by_card / cid
        if not list_source_images(folder):
            missing.append(cid)
    return missing


def occlude_ul_stack(
    img: np.ndarray,
    hide_bottom_frac: float,
    shift_x: int,
    shift_y: int,
    fill_bgr: Tuple[int, int, int] = OCCLUSION_BGR,
) -> np.ndarray:
    """
    Keep the upper strip of the card; hide the bottom fraction and apply a
    small translation so the visible index is not always dead-center.
    Output size matches the input.
    """
    h, w = img.shape[:2]
    hide = float(np.clip(hide_bottom_frac, HIDE_BOTTOM_MIN, HIDE_BOTTOM_MAX))
    keep_h = max(1, int(round(h * (1.0 - hide))))

    canvas = np.empty_like(img)
    canvas[:] = fill_bgr

    # Visible top strip pasted onto the canvas with (shift_x, shift_y).
    # Negative shifts clip against the top/left edges.
    src = img[:keep_h]
    dst_y0 = shift_y
    dst_x0 = shift_x
    src_y0 = max(0, -dst_y0)
    src_x0 = max(0, -dst_x0)
    dst_y0 = max(0, dst_y0)
    dst_x0 = max(0, dst_x0)
    copy_h = min(src.shape[0] - src_y0, h - dst_y0)
    copy_w = min(src.shape[1] - src_x0, w - dst_x0)
    if copy_h > 0 and copy_w > 0:
        canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = (
            src[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
        )

    # Force the bottom hide band so a downward shift never reveals lower face art.
    occ_y = max(1, int(round(h * (1.0 - hide))))
    if occ_y < h:
        canvas[occ_y:, :] = fill_bgr
    return canvas


def variants_for_image(
    img: np.ndarray,
    n: int,
    rng: random.Random,
) -> List[np.ndarray]:
    h, w = img.shape[:2]
    out: List[np.ndarray] = []
    for i in range(n):
        hide = rng.uniform(HIDE_BOTTOM_MIN, HIDE_BOTTOM_MAX)
        # Deterministic ends of the range for the first two slots when n >= 2.
        if i == 0:
            hide = HIDE_BOTTOM_MIN
        elif i == 1 and n >= 2:
            hide = HIDE_BOTTOM_MAX
        max_dx = max(1, int(round(w * SHIFT_X_FRAC)))
        max_dy = max(1, int(round(h * SHIFT_Y_FRAC)))
        shift_x = rng.randint(-max_dx, max_dx)
        shift_y = rng.randint(-max_dy, max(0, max_dy // 2))  # prefer slight down/up, not large up
        out.append(occlude_ul_stack(img, hide, shift_x, shift_y))
    return out


def augment_card_folder(
    src_dir: Path,
    dst_dir: Path,
    variants: int,
    rng: random.Random,
) -> int:
    images = list_source_images(src_dir)
    if not images:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip unreadable: {img_path}")
            continue
        for vi, aug in enumerate(variants_for_image(img, variants, rng)):
            out_name = f"{img_path.stem}_ul{vi}.png"
            out_path = dst_dir / out_name
            if not cv2.imwrite(str(out_path), aug):
                print(f"  failed write: {out_path}")
                continue
            written += 1
    return written


def run_augment(
    cards_dir: Optional[Path] = None,
    variants: int = 3,
    seed: int = 42,
    card_ids: Optional[Iterable[str]] = None,
) -> dict:
    root = cards_root(cards_dir)
    by_card = root / "by_card"
    by_card_ul = root / "by_card_ul"
    by_card_ul.mkdir(parents=True, exist_ok=True)

    ids = list(card_ids) if card_ids is not None else list(CARD_NAMES)
    rng = random.Random(seed)
    total_src = 0
    total_out = 0
    cards_with_src = 0
    cards_empty = []

    for cid in ids:
        src = by_card / cid
        dst = by_card_ul / cid
        n_src = len(list_source_images(src))
        if n_src == 0:
            cards_empty.append(cid)
            dst.mkdir(parents=True, exist_ok=True)
            continue
        cards_with_src += 1
        total_src += n_src
        # Per-card RNG stream so adding a new card later does not reshuffle others.
        card_rng = random.Random(rng.randint(0, 2**31 - 1))
        n_out = augment_card_folder(src, dst, variants, card_rng)
        total_out += n_out

    missing_qd_5s = missing_source_cards(by_card, ("Qd", "5s"))
    summary = {
        "by_card": str(by_card),
        "by_card_ul": str(by_card_ul),
        "cards_with_source": cards_with_src,
        "cards_empty": cards_empty,
        "source_images": total_src,
        "variants_per_image": variants,
        "written": total_out,
        "missing_qd_5s": missing_qd_5s,
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="UL/stack occlusion augmentations → data/cards/by_card_ul/",
    )
    parser.add_argument(
        "--cards-root",
        type=Path,
        default=None,
        help="Override data/cards directory (default: <project>/data/cards)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Occlusion variants per source PNG (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.variants < 1:
        print("--variants must be >= 1")
        return 2

    summary = run_augment(
        cards_dir=args.cards_root,
        variants=args.variants,
        seed=args.seed,
    )
    print(
        f"UL augment: {summary['written']} images "
        f"from {summary['source_images']} sources "
        f"across {summary['cards_with_source']} cards -> {summary['by_card_ul']}"
    )
    print(f"  variants per image: {summary['variants_per_image']}")
    if summary["missing_qd_5s"]:
        print(
            "  still missing real crops in by_card (do not invent Stake art): "
            + ", ".join(summary["missing_qd_5s"])
        )
    else:
        print("  Qd and 5s both have at least one source crop in by_card")
    empty_n = len(summary["cards_empty"])
    if empty_n:
        print(f"  {empty_n} card folders had no source images (UL dirs created empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
