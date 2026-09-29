"""Dual-head card classifier: rank (13) + suit (4) on a shared ImageNet backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

# Default order matches data/cards/cards.yaml / src/training/card_names.py
RANKS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS: List[str] = ["h", "d", "c", "s"]


def get_card_names(ranks: Iterable[str] = RANKS, suits: Iterable[str] = SUITS) -> List[str]:
    return [r + s for r in ranks for s in suits]


def load_card_names(cards_yaml: Optional[Path] = None) -> List[str]:
    """CARD_NAMES from cards.yaml if present, else rank×suit default order."""
    if cards_yaml is None:
        cards_yaml = Path(__file__).resolve().parents[2] / "data" / "cards" / "cards.yaml"
    if cards_yaml.is_file():
        try:
            import yaml
        except ImportError:
            yaml = None
        if yaml is not None:
            with cards_yaml.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            names = cfg.get("names")
            if isinstance(names, dict) and names:
                ordered = [names[i] for i in sorted(names.keys(), key=lambda k: int(k))]
                if len(ordered) == 52:
                    return ordered
            if isinstance(names, list) and len(names) == 52:
                return list(names)
        # Minimal YAML parse without PyYAML: "  N: Xx" lines under names:
        ordered: List[Tuple[int, str]] = []
        in_names = False
        for line in cards_yaml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    break
                if stripped.startswith("nc:") or not line.startswith(" "):
                    break
                key, _, val = stripped.partition(":")
                try:
                    ordered.append((int(key.strip()), val.strip()))
                except ValueError:
                    break
        if len(ordered) == 52:
            ordered.sort(key=lambda t: t[0])
            return [v for _, v in ordered]
    return get_card_names()


CARD_NAMES: List[str] = load_card_names()


def parse_card_name(name: str) -> Tuple[str, str]:
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid card name: {name!r}")
    rank, suit = name[:-1], name[-1].lower()
    if rank == "10":
        rank = "T"
    else:
        rank = rank.upper() if rank.isalpha() else rank
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"Invalid card name: {name!r} (rank={rank!r} suit={suit!r})")
    return rank, suit


def card_to_indices(name: str) -> Tuple[int, int]:
    rank, suit = parse_card_name(name)
    return RANKS.index(rank), SUITS.index(suit)


def indices_to_card(rank_i: int, suit_i: int) -> str:
    return RANKS[rank_i] + SUITS[suit_i]


def _backbone_and_dim(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    name = name.lower().replace("-", "_")
    weights = "DEFAULT" if pretrained else None
    if name in ("mobilenet_v3_small", "mobilenetv3_small", "mobilenetv3"):
        try:
            w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=w)
        except Exception:
            net = models.mobilenet_v3_small(pretrained=pretrained)
        feat_dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, feat_dim
    if name in ("resnet18", "resnet_18"):
        try:
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=w)
        except Exception:
            net = models.resnet18(pretrained=pretrained)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    raise ValueError(f"Unknown backbone: {name!r} (use mobilenet_v3_small or resnet18)")


class DualHeadCardClassifier(nn.Module):
    """Shared CNN trunk with separate rank and suit heads."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.trunk, feat_dim = _backbone_and_dim(backbone, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.rank_head = nn.Linear(feat_dim, len(RANKS))
        self.suit_head = nn.Linear(feat_dim, len(SUITS))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        feat = self.dropout(feat)
        return self.rank_head(feat), self.suit_head(feat)

    def predict_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rank_idx, suit_idx, confidence) with confidence = P(rank)*P(suit)."""
        rank_logits, suit_logits = self.forward(x)
        rank_p = torch.softmax(rank_logits, dim=-1)
        suit_p = torch.softmax(suit_logits, dim=-1)
        rank_conf, rank_i = rank_p.max(dim=-1)
        suit_conf, suit_i = suit_p.max(dim=-1)
        conf = rank_conf * suit_conf
        return rank_i, suit_i, conf
