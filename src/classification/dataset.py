"""Dataset + augmentations for card crops under data/cards/by_card/{rank}{suit}/."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .model import CARD_NAMES, RANKS, SUITS, card_to_indices, load_card_names

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ULStackCrop:
    """Keep upper-left; occlude bottom and/or right (stacked / overlapped cards)."""

    def __init__(
        self,
        p: float = 0.5,
        bottom_frac: Tuple[float, float] = (0.15, 0.45),
        right_frac: Tuple[float, float] = (0.15, 0.45),
        fill: int = 0,
    ):
        self.p = p
        self.bottom_frac = bottom_frac
        self.right_frac = right_frac
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        out = img.copy()
        # Bottom occlusion
        if random.random() < 0.85:
            bf = random.uniform(*self.bottom_frac)
            y0 = int(h * (1.0 - bf))
            band = Image.new(out.mode, (w, h - y0), color=self.fill if out.mode == "L" else (self.fill,) * len(out.getbands()))
            out.paste(band, (0, y0))
        # Right occlusion
        if random.random() < 0.85:
            rf = random.uniform(*self.right_frac)
            x0 = int(w * (1.0 - rf))
            band = Image.new(out.mode, (w - x0, h), color=self.fill if out.mode == "L" else (self.fill,) * len(out.getbands()))
            out.paste(band, (x0, 0))
        return out


def imagenet_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_train_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            ULStackCrop(p=0.55),
            transforms.RandomRotation(degrees=8, fill=0),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )


def build_eval_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )


def discover_samples(
    by_card_root: Path,
    card_names: Optional[Sequence[str]] = None,
) -> List[Tuple[Path, str, int, int]]:
    """Return list of (path, card_name, rank_idx, suit_idx)."""
    names = list(card_names) if card_names is not None else load_card_names()
    name_set = set(names)
    samples: List[Tuple[Path, str, int, int]] = []
    if not by_card_root.is_dir():
        return samples
    for folder in sorted(by_card_root.iterdir()):
        if not folder.is_dir():
            continue
        card = folder.name
        if card not in name_set:
            # Still accept valid rank+suit folders even if yaml differs
            try:
                ri, si = card_to_indices(card)
            except ValueError:
                continue
        else:
            ri, si = card_to_indices(card)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, card, ri, si))
    return samples


def stratified_holdout_split(
    samples: Sequence[Tuple[Path, str, int, int]],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, str, int, int]], List[Tuple[Path, str, int, int]]]:
    """Per-card holdout: ~val_frac of each card with >=2 images; singletons stay in train."""
    rng = random.Random(seed)
    by_card: dict[str, List[Tuple[Path, str, int, int]]] = {}
    for s in samples:
        by_card.setdefault(s[1], []).append(s)
    train: List[Tuple[Path, str, int, int]] = []
    val: List[Tuple[Path, str, int, int]] = []
    for card, items in by_card.items():
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        n_val = max(1, int(round(len(items) * val_frac)))
        n_val = min(n_val, len(items) - 1)  # keep at least one in train
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CardCropDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str, int, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, _card, rank_i, suit_i = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rank_i, suit_i
