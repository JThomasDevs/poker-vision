"""Load classifier.pt and predict (rank, suit, conf) for one crop image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification.dataset import build_eval_transform
from src.classification.model import (
    RANKS,
    SUITS,
    DualHeadCardClassifier,
)

# Digit-like ranks that the index corner often swaps under occlusion / tiny crops.
# Prefer full-card logits on these pairs: center pips disambiguate when UL alone
# confuses thin glyphs (3↔5, 8↔3, 5↔6/9, 2↔A, …).
RANK_CONFUSION_PAIRS = frozenset(
    {
        ("3", "5"),
        ("5", "3"),
        ("3", "8"),
        ("8", "3"),
        ("5", "6"),
        ("6", "5"),
        ("5", "9"),
        ("9", "5"),
        ("5", "T"),
        ("T", "5"),
        ("9", "T"),
        ("T", "9"),
        ("9", "6"),
        ("6", "9"),
        ("9", "7"),
        ("7", "9"),
        ("2", "A"),
        ("A", "2"),
        ("2", "3"),
        ("3", "2"),
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_classifier(
    checkpoint: Union[str, Path],
    device: Optional[torch.device] = None,
    label_map: Optional[Union[str, Path]] = None,
) -> Tuple[DualHeadCardClassifier, Dict[str, Any], torch.device]:
    ckpt_path = Path(checkpoint)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = blob.get("backbone", "mobilenet_v3_small")
    image_size = int(blob.get("image_size", 224))
    ranks = blob.get("ranks", RANKS)
    suits = blob.get("suits", SUITS)

    if label_map is not None:
        lm = json.loads(Path(label_map).read_text(encoding="utf-8"))
        ranks = lm.get("ranks", ranks)
        suits = lm.get("suits", suits)
        backbone = lm.get("backbone", backbone)
        image_size = int(lm.get("image_size", image_size))

    model = DualHeadCardClassifier(backbone=backbone, pretrained=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    meta = {
        "backbone": backbone,
        "image_size": image_size,
        "ranks": ranks,
        "suits": suits,
        "card_names": blob.get("card_names") if isinstance(blob, dict) else None,
        "metrics": blob.get("metrics") if isinstance(blob, dict) else None,
        "checkpoint": str(ckpt_path),
    }
    return model, meta, device


@dataclass
class ClassifierDist:
    """Full rank/suit probability distributions from one classification."""

    rank_probs: np.ndarray
    suit_probs: np.ndarray
    ranks: List[str]
    suits: List[str]

    def __post_init__(self) -> None:
        self.rank_probs = np.asarray(self.rank_probs, dtype=np.float64).reshape(-1)
        self.suit_probs = np.asarray(self.suit_probs, dtype=np.float64).reshape(-1)
        rs = float(self.rank_probs.sum())
        ss = float(self.suit_probs.sum())
        if rs > 0:
            self.rank_probs = self.rank_probs / rs
        if ss > 0:
            self.suit_probs = self.suit_probs / ss

    def top_rank(self) -> str:
        return self.ranks[int(self.rank_probs.argmax())]

    def top_suit(self) -> str:
        return self.suits[int(self.suit_probs.argmax())]

    def top_label(self) -> str:
        return f"{self.top_rank()}{self.top_suit()}"

    def top_conf(self) -> float:
        ri = int(self.rank_probs.argmax())
        si = int(self.suit_probs.argmax())
        return float(self.rank_probs[ri] * self.suit_probs[si])

    def rank_margin(self) -> float:
        p = self.rank_probs
        if p.size < 2:
            return float(p[0]) if p.size else 0.0
        top2 = np.partition(p, -2)[-2:]
        return float(top2[-1] - top2[-2])

    def suit_margin(self) -> float:
        p = self.suit_probs
        if p.size < 2:
            return float(p[0]) if p.size else 0.0
        top2 = np.partition(p, -2)[-2:]
        return float(top2[-1] - top2[-2])

    def as_tuple(self) -> Tuple[str, str, float]:
        return self.top_rank(), self.top_suit(), self.top_conf()


def _tensor_probs_to_dist(
    rp: torch.Tensor,
    sp: torch.Tensor,
    meta: Dict[str, Any],
) -> ClassifierDist:
    ranks = list(meta.get("ranks", RANKS))
    suits = list(meta.get("suits", SUITS))
    return ClassifierDist(
        rank_probs=rp.detach().cpu().numpy().astype(np.float64),
        suit_probs=sp.detach().cpu().numpy().astype(np.float64),
        ranks=ranks,
        suits=suits,
    )


def _bgr_to_probs(
    bgr: np.ndarray,
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tfm = build_eval_transform(int(meta.get("image_size", 224)))
    x = tfm(img).unsqueeze(0).to(device)
    rank_logits, suit_logits = model(x)
    return torch.softmax(rank_logits, dim=-1)[0], torch.softmax(suit_logits, dim=-1)[0]


def _prefer_full_rank(
    rp: torch.Tensor,
    rp_f: torch.Tensor,
    rp_u: torch.Tensor,
    ranks: Sequence[str],
    full_prefer_conf: float,
) -> Tuple[torch.Tensor, bool]:
    """Optionally replace blend mass with full-card one-hot-ish preference on confusion pairs."""
    ri = int(rp.argmax().item())
    ri_f = int(rp_f.argmax().item())
    ri_u = int(rp_u.argmax().item())
    prefer_full = False
    if ranks[ri] != ranks[ri_f] and (ranks[ri], ranks[ri_f]) in RANK_CONFUSION_PAIRS:
        prefer_full = True
    if ranks[ri_f] != ranks[ri_u] and (ranks[ri_f], ranks[ri_u]) in RANK_CONFUSION_PAIRS:
        prefer_full = True
    if prefer_full and float(rp_f[ri_f].item()) >= full_prefer_conf:
        return rp_f, True
    return rp, False


@torch.no_grad()
def predict_image_dist(
    image: Union[str, Path, Image.Image],
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
) -> ClassifierDist:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    tfm = build_eval_transform(int(meta.get("image_size", 224)))
    x = tfm(img).unsqueeze(0).to(device)
    rank_logits, suit_logits = model(x)
    rank_p = torch.softmax(rank_logits, dim=-1)[0]
    suit_p = torch.softmax(suit_logits, dim=-1)[0]
    return _tensor_probs_to_dist(rank_p, suit_p, meta)


@torch.no_grad()
def predict_image(
    image: Union[str, Path, Image.Image],
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, str, float]:
    """Return (rank, suit, confidence) with confidence = P(rank) * P(suit)."""
    return predict_image_dist(image, model, meta, device).as_tuple()


@torch.no_grad()
def predict_full_and_ul_dist(
    full_bgr: np.ndarray,
    ul_bgr: np.ndarray,
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
    *,
    full_rank_weight: float = 0.65,
    full_prefer_conf: float = 0.35,
) -> ClassifierDist:
    """Blend full-card + UL index distributions; break digit ties toward full."""
    rp_f, sp_f = _bgr_to_probs(full_bgr, model, meta, device)
    rp_u, sp_u = _bgr_to_probs(ul_bgr, model, meta, device)

    w = float(full_rank_weight)
    w = min(max(w, 0.0), 1.0)
    rp = w * rp_f + (1.0 - w) * rp_u
    sp = 0.50 * sp_f + 0.50 * sp_u

    ranks = meta.get("ranks", RANKS)
    rp, _ = _prefer_full_rank(rp, rp_f, rp_u, ranks, full_prefer_conf)
    return _tensor_probs_to_dist(rp, sp, meta)


@torch.no_grad()
def predict_full_and_ul(
    full_bgr: np.ndarray,
    ul_bgr: np.ndarray,
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
    *,
    full_rank_weight: float = 0.65,
    full_prefer_conf: float = 0.35,
) -> Tuple[str, str, float]:
    """Blend full-card + UL index predictions; break digit ties toward full.

    Full-card center pips disambiguate 3/5/6/8/9; UL alone often swaps those
    ranks (and 5/9 ↔ T) when the index is tiny or partially occluded.
    """
    return predict_full_and_ul_dist(
        full_bgr,
        ul_bgr,
        model,
        meta,
        device,
        full_rank_weight=full_rank_weight,
        full_prefer_conf=full_prefer_conf,
    ).as_tuple()


def _clahe_bgr(bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Local contrast restore so avatar-dimmed faces match bright training crops."""
    import cv2

    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
    return cv2.cvtColor(
        cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2BGR
    )


def _suit_pip_crop(card_bgr: np.ndarray) -> np.ndarray:
    """UL suit pip below the rank glyph (rank alone confuses heart↔diamond)."""
    if card_bgr is None or card_bgr.size == 0:
        return card_bgr
    ch, cw = card_bgr.shape[:2]
    y0 = int(ch * 0.18)
    y1 = max(y0 + 1, int(ch * 0.52))
    x1 = max(1, int(cw * 0.52))
    return card_bgr[y0:y1, :x1].copy()


@torch.no_grad()
def predict_hole_card_dist(
    full_bgr: np.ndarray,
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
    *,
    ul_bgr: Optional[np.ndarray] = None,
) -> ClassifierDist:
    """Hole-card distributions with avatar-dim restore (same blends as predict_hole_card)."""
    from src.detection.layout import (
        avatar_contaminated,
        hole_index_crop,
        mask_circular_avatar,
        ul_crop,
    )

    index = ul_bgr if ul_bgr is not None else hole_index_crop(full_bgr)
    contaminated = avatar_contaminated(full_bgr)

    if contaminated:
        restored = _clahe_bgr(full_bgr, clip=2.0)
        idx = hole_index_crop(restored, frac_h=0.42, frac_w=0.50)
        pip = _suit_pip_crop(restored)
        rp_f, sp_f = _bgr_to_probs(restored, model, meta, device)
        rp_u, sp_u = _bgr_to_probs(idx, model, meta, device)
        rp_p, sp_p = _bgr_to_probs(pip, model, meta, device)

        ranks = meta.get("ranks", RANKS)

        rp = 0.70 * rp_f + 0.30 * rp_u
        sp = 0.25 * sp_f + 0.25 * sp_u + 0.50 * sp_p

        ri_f = int(rp_f.argmax().item())
        ri_u = int(rp_u.argmax().item())
        if (
            ranks[ri_f] != ranks[ri_u]
            and (ranks[ri_f], ranks[ri_u]) in RANK_CONFUSION_PAIRS
            and float(rp_f[ri_f].item()) >= 0.30
        ):
            rp = rp_f

        dist = _tensor_probs_to_dist(rp, sp, meta)
        if dist.top_conf() >= 0.12:
            return dist

        masked = mask_circular_avatar(full_bgr)
        return predict_full_and_ul_dist(
            masked,
            index if index is not None and index.size else hole_index_crop(masked),
            model,
            meta,
            device,
            full_rank_weight=0.15,
            full_prefer_conf=1.01,
        )

    if index is None or index.size == 0:
        index = ul_crop(full_bgr, 0.50)
    return predict_full_and_ul_dist(
        full_bgr,
        index,
        model,
        meta,
        device,
        full_rank_weight=0.55,
        full_prefer_conf=0.35,
    )


@torch.no_grad()
def predict_hole_card(
    full_bgr: np.ndarray,
    model: DualHeadCardClassifier,
    meta: Dict[str, Any],
    device: torch.device,
    *,
    ul_bgr: Optional[np.ndarray] = None,
) -> Tuple[str, str, float]:
    """Classify a hero hole crop; restore avatar-dimmed faces before the CNN.

    Stake seats draw a circular portrait over the hole pair and darken the face
    (train crops are near-white; live mid-right seats often mean≈100). Raw UL
    max-conf then swaps 9↔6 and 2↔A. CLAHE restores contrast so the full face
    (plus a suit-pip crop) reads correctly without a retrain.
    """
    return predict_hole_card_dist(
        full_bgr, model, meta, device, ul_bgr=ul_bgr
    ).as_tuple()


def parse_args(argv=None):
    root = project_root()
    p = argparse.ArgumentParser(description="Infer rank+suit for one card crop")
    p.add_argument("image", type=Path, help="Path to a card crop image")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "data" / "cards" / "models" / "classifier.pt",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional label_map.json (defaults to sibling of checkpoint)",
    )
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    label_map = args.label_map
    if label_map is None:
        sibling = args.checkpoint.with_name("label_map.json")
        if sibling.is_file():
            label_map = sibling

    device = None
    if args.device:
        device = torch.device(args.device)

    model, meta, device = load_classifier(args.checkpoint, device=device, label_map=label_map)
    rank, suit, conf = predict_image(args.image, model, meta, device)
    card = f"{rank}{suit}"
    print(f"{card}  rank={rank}  suit={suit}  conf={conf:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
