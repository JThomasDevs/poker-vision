"""
Scan assets folder and delete images that are only blue background (no cards, not even partial).
Uses the same card detection as crop_cards_from_image; if no regions are found and the image
is predominantly uniform blue, the file is removed.

Usage:
  python -m src.training.clean_assets_blue_only
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.training.crop_cards_from_image import find_card_regions

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

# Blue background only: low variance and B dominant
MAX_STD_FOR_UNIFORM = 120
MIN_B_DOMINANCE = 0  # mean B >= mean R and mean B >= mean G


def get_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets"


def _is_uniform_blue(img: np.ndarray) -> bool:
    """True if image is predominantly uniform blue (B dominant, low variance)."""
    if len(img.shape) == 2:
        return False
    b, g, r = cv2.split(img)
    mean_b, mean_g, mean_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    if std > MAX_STD_FOR_UNIFORM:
        return False
    return mean_b >= mean_r + MIN_B_DOMINANCE and mean_b >= mean_g + MIN_B_DOMINANCE


def is_blue_background_only(img: np.ndarray, regions: Optional[list] = None) -> bool:
    """True if image is only blue background: no cards (or only small UI/text blobs) and uniform blue."""
    if regions is None:
        regions = find_card_regions(img)
    if not _is_uniform_blue(img):
        return False
    if not regions:
        return True
    h, w = img.shape[:2]
    area = h * w
    # One contour that's almost the whole image = table edge, not cards
    if len(regions) == 1:
        x1, y1, x2, y2 = regions[0]
        if (x2 - x1) * (y2 - y1) >= 0.85 * area:
            return True
    # All contours together are small = text/UI only (e.g. "Stake" logo), not cards
    total_region_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in regions)
    if total_region_area < 0.25 * area:
        return True
    return False


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Delete asset images that are only blue background.")
    parser.add_argument("--dry-run", action="store_true", help="List would-be deleted files only (no delete)")
    args = parser.parse_args()

    assets = get_assets_dir()
    if not assets.is_dir():
        print("Assets folder not found:", assets)
        return 1
    to_delete = []
    zero_regions = 0
    total = 0
    for f in sorted(assets.iterdir(), key=lambda p: p.name):
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total += 1
        img = cv2.imread(str(f))
        if img is None:
            continue
        regions = find_card_regions(img)
        if not regions:
            zero_regions += 1
        if is_blue_background_only(img, regions):
            to_delete.append(f)
    if not to_delete:
        print("No blue-background-only images found.")
        print(f"(Images with 0 card regions: {zero_regions} of {total} total)")
        return 0
    print(f"{'Would delete' if args.dry_run else 'Deleting'} {len(to_delete)} image(s) that are only blue background:")
    for p in to_delete:
        print(" ", p.name)
        if not args.dry_run:
            p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
