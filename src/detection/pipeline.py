"""Fast card path: blob find -> layout split -> CNN classify (+ optional tracker).

Uses ``layout.py`` (Wave B1) when present; otherwise a thin local Y-split
fallback (note: B1 lag — replace once ``src/detection/layout.py`` lands).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification.infer import (
    load_classifier,
    predict_full_and_ul,
    predict_full_and_ul_dist,
    predict_hole_card,
    predict_hole_card_dist,
    predict_image,
    predict_image_dist,
)
from src.detection.blobs import find_card_blobs
from src.detection.cards import DetectedCard
from src.state.accept import accept_table
from src.state.table import TableState, build_table_state
from src.state.tracker import CardTracker
from src.state.types import (
    SLOT_BOARD,
    SLOT_HERO,
    VISIBLE,
    RawDet,
)

Box = Tuple[int, int, int, int]

try:
    from src.detection.layout import (
        filter_card_like_boxes,
        select_hero_hole_pair,
        split_community_and_holes,
        ul_crop,
    )

    _LAYOUT_BACKEND = "layout"
except ImportError:
    _LAYOUT_BACKEND = "fallback"
    filter_card_like_boxes = None  # type: ignore
    ul_crop = None  # type: ignore
    warnings.warn(
        "src.detection.layout missing (Wave B1 lag); "
        "using thin Y-split fallback in pipeline.py",
        RuntimeWarning,
        stacklevel=1,
    )

    def split_community_and_holes(
        boxes: Sequence[Box],
        image_shape: Tuple[int, ...],
    ) -> Tuple[List[Box], List[Box]]:
        """Crude top-Y cluster = community; rest = holes (B1 placeholder)."""
        del image_shape
        if not boxes:
            return [], []
        ordered = sorted(boxes, key=lambda b: b[1])
        y0 = ordered[0][1]
        y_threshold = y0 + 100
        community = [b for b in boxes if b[1] < y_threshold]
        holes = [b for b in boxes if b[1] >= y_threshold]
        if not holes and len(community) >= 2:
            by_bottom = sorted(community, key=lambda b: b[3], reverse=True)
            holes = by_bottom[:2]
            community = [b for b in community if b not in holes]
        return community, holes

    def select_hero_hole_pair(
        hole_boxes: Sequence[Box],
        image_bgr: Optional[np.ndarray] = None,
    ) -> List[Box]:
        del image_bgr
        if not hole_boxes:
            return []
        if len(hole_boxes) <= 2:
            return sorted(hole_boxes, key=lambda b: b[0])
        by_cy = sorted(
            hole_boxes,
            key=lambda b: (b[1] + b[3]) / 2.0,
            reverse=True,
        )
        return sorted(by_cy[:2], key=lambda b: b[0])


def _crop_box(image_bgr: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2].copy()


def default_classifier_path() -> Path:
    return _ROOT / "data" / "cards" / "models" / "classifier.pt"


def classifier_available(path: Optional[Union[str, Path]] = None) -> bool:
    p = Path(path) if path is not None else default_classifier_path()
    return p.is_file() and p.stat().st_size > 0


@dataclass
class LabeledCard:
    """One classified card with its source box."""

    rank: str
    suit: str
    confidence: float
    bbox: Box

    @property
    def label(self) -> str:
        return f"{self.rank}{self.suit}"

    def to_detected(self) -> DetectedCard:
        return DetectedCard(
            rank=self.rank,
            suit=self.suit,
            confidence=self.confidence,
            bbox=self.bbox,
        )


@dataclass
class FastCardsResult:
    """Structured output of the blob + CNN path."""

    community: List[LabeledCard] = field(default_factory=list)
    holes: List[LabeledCard] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    community_boxes: List[Box] = field(default_factory=list)
    hole_boxes: List[Box] = field(default_factory=list)
    layout_backend: str = _LAYOUT_BACKEND
    table: Optional[TableState] = None

    @property
    def community_labels(self) -> List[str]:
        return [c.label for c in self.community]

    @property
    def hole_labels(self) -> List[str]:
        return [c.label for c in self.holes]

    def as_detected_cards(self) -> List[DetectedCard]:
        return [c.to_detected() for c in self.community + self.holes]


def _boxes_key(boxes: Sequence[Box], *, quant: int = 8) -> Tuple[Tuple[int, int, int, int], ...]:
    """Quantized box tuple for stable-box cache hits (jitter within ``quant`` px)."""
    keyed = []
    for b in boxes:
        keyed.append(
            (
                int(b[0]) // quant,
                int(b[1]) // quant,
                int(b[2]) // quant,
                int(b[3]) // quant,
            )
        )
    return tuple(sorted(keyed))


def _obs_to_labeled(obs) -> Optional[LabeledCard]:
    if obs.visibility != VISIBLE:
        return None
    if not obs.label or len(obs.label) < 2 or obs.label == "??":
        return None
    return LabeledCard(
        rank=obs.rank,
        suit=obs.suit,
        confidence=float(obs.confidence),
        bbox=obs.bbox,
    )


class FastCardsPipeline:
    """find_card_blobs -> layout split -> classify crops with classifier.pt."""

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        label_map: Optional[Union[str, Path]] = None,
        *,
        reuse_stable_boxes: bool = True,
        box_quant: int = 12,
        use_tracker: bool = True,
    ):
        ckpt = Path(checkpoint) if checkpoint else default_classifier_path()
        if not classifier_available(ckpt):
            raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt}")

        if label_map is None:
            sibling = ckpt.with_name("label_map.json")
            if sibling.is_file():
                label_map = sibling

        torch_device = None
        if device:
            import torch

            torch_device = torch.device(device)

        self.model, self.meta, self.device = load_classifier(
            ckpt, device=torch_device, label_map=label_map
        )
        self.checkpoint = ckpt
        self.layout_backend = _LAYOUT_BACKEND
        self.reuse_stable_boxes = reuse_stable_boxes
        self.box_quant = max(1, int(box_quant))
        self.use_tracker = bool(use_tracker)
        self.tracker = CardTracker() if self.use_tracker else None
        self._cached_key: Optional[
            Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[Tuple[int, int, int, int], ...]]
        ] = None
        self._cached_result: Optional[FastCardsResult] = None
        self.last_classify_ran: bool = False

    def process(self, frame_bgr: np.ndarray) -> FastCardsResult:
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_classify_ran = False
            return FastCardsResult(layout_backend=self.layout_backend)

        raw_boxes = find_card_blobs(frame_bgr)
        if filter_card_like_boxes is not None:
            boxes = filter_card_like_boxes(raw_boxes, frame_bgr)
        else:
            boxes = raw_boxes
        community_boxes, hole_boxes = split_community_and_holes(
            boxes, frame_bgr.shape
        )
        hero_boxes = select_hero_hole_pair(hole_boxes, frame_bgr)

        if self.use_tracker and self.tracker is not None:
            return self._process_tracked(
                frame_bgr,
                boxes=boxes,
                community_boxes=community_boxes,
                hero_boxes=hero_boxes,
            )

        key = (
            _boxes_key(community_boxes, quant=self.box_quant),
            _boxes_key(hero_boxes, quant=self.box_quant),
        )
        if (
            self.reuse_stable_boxes
            and self._cached_result is not None
            and key == self._cached_key
            and key != ((), ())
        ):
            # Boxes stable → reuse CNN labels (dual full+UL is expensive).
            self.last_classify_ran = False
            cached = self._cached_result
            return FastCardsResult(
                community=list(cached.community),
                holes=list(cached.holes),
                boxes=list(boxes),
                community_boxes=list(community_boxes),
                hole_boxes=list(hero_boxes),
                layout_backend=self.layout_backend,
                table=None,
            )

        community = self._dedupe_labels(
            self._classify_boxes(frame_bgr, community_boxes, hole_mode=False),
            max_n=5,
        )
        holes = self._dedupe_labels(
            self._classify_boxes(frame_bgr, hero_boxes, hole_mode=True),
            max_n=2,
        )

        result = FastCardsResult(
            community=community,
            holes=holes,
            boxes=list(boxes),
            community_boxes=list(community_boxes),
            hole_boxes=list(hero_boxes),
            layout_backend=self.layout_backend,
            table=None,
        )
        self._cached_key = key
        self._cached_result = result
        self.last_classify_ran = True
        return result

    def _process_tracked(
        self,
        frame_bgr: np.ndarray,
        *,
        boxes: Sequence[Box],
        community_boxes: Sequence[Box],
        hero_boxes: Sequence[Box],
    ) -> FastCardsResult:
        """Always classify with dist APIs, then tracker → accept → TableState."""
        assert self.tracker is not None
        community_sorted = sorted(community_boxes, key=lambda b: b[0])[:5]
        hero_sorted = sorted(hero_boxes, key=lambda b: b[0])[:2]

        raw_dets: List[RawDet] = []
        raw_dets.extend(
            self._classify_boxes_to_raw(
                frame_bgr,
                community_sorted,
                hole_mode=False,
                slot_ids=SLOT_BOARD[: len(community_sorted)],
            )
        )
        raw_dets.extend(
            self._classify_boxes_to_raw(
                frame_bgr,
                hero_sorted,
                hole_mode=True,
                slot_ids=SLOT_HERO[: len(hero_sorted)],
            )
        )

        table, community, holes = self.apply_tracker_dets(
            raw_dets,
            board_detected=len(community_sorted) > 0,
        )
        self.last_classify_ran = True
        return FastCardsResult(
            community=community,
            holes=holes,
            boxes=list(boxes),
            community_boxes=list(community_sorted),
            hole_boxes=list(hero_sorted),
            layout_backend=self.layout_backend,
            table=table,
        )

    def apply_tracker_dets(
        self,
        raw_dets: Sequence[RawDet],
        *,
        board_detected: bool = False,
    ) -> Tuple[TableState, List[LabeledCard], List[LabeledCard]]:
        """Public test seam: tracker.update → accept_table → TableState + VISIBLE labels."""
        if self.tracker is None:
            self.tracker = CardTracker()
            self.use_tracker = True
        obs = self.tracker.update(list(raw_dets))
        obs = accept_table(obs)
        table = build_table_state(obs, board_detected=board_detected)
        community: List[LabeledCard] = []
        holes: List[LabeledCard] = []
        for o in table.board:
            labeled = _obs_to_labeled(o)
            if labeled is not None:
                community.append(labeled)
        for o in table.hero:
            labeled = _obs_to_labeled(o)
            if labeled is not None:
                holes.append(labeled)
        return table, community[:5], holes[:2]

    def _classify_boxes_to_raw(
        self,
        frame_bgr: np.ndarray,
        boxes: Sequence[Box],
        *,
        hole_mode: bool,
        slot_ids: Sequence[str],
    ) -> List[RawDet]:
        dets: List[RawDet] = []
        for i, box in enumerate(boxes):
            crop = _crop_box(frame_bgr, box)
            if crop is None or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            if hole_mode:
                dist = predict_hole_card_dist(
                    crop, self.model, self.meta, self.device
                )
            elif ul_crop is not None:
                index = ul_crop(crop, 0.55)
                if index is not None and index.size > 0:
                    dist = predict_full_and_ul_dist(
                        crop, index, self.model, self.meta, self.device
                    )
                else:
                    rgb_full = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    dist = predict_image_dist(
                        Image.fromarray(rgb_full),
                        self.model,
                        self.meta,
                        self.device,
                    )
            else:
                rgb_full = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                dist = predict_image_dist(
                    Image.fromarray(rgb_full),
                    self.model,
                    self.meta,
                    self.device,
                )
            slot_hint = slot_ids[i] if i < len(slot_ids) else None
            dets.append(
                RawDet(
                    bbox=box,
                    rank_probs=dist.rank_probs,
                    suit_probs=dist.suit_probs,
                    hole_mode=hole_mode,
                    slot_hint=slot_hint,
                )
            )
        return dets

    def _classify_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: Sequence[Box],
        *,
        hole_mode: bool = False,
    ) -> List[LabeledCard]:
        labeled: List[LabeledCard] = []
        for box in boxes:
            crop = _crop_box(frame_bgr, box)
            if crop is None or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            if hole_mode:
                # Avatar covers center pips — UL / hole path.
                rank, suit, conf = predict_hole_card(
                    crop, self.model, self.meta, self.device
                )
            elif ul_crop is not None:
                # Board: full + UL blend; prefer full on digit confusion pairs.
                index = ul_crop(crop, 0.55)
                if index is not None and index.size > 0:
                    rank, suit, conf = predict_full_and_ul(
                        crop, index, self.model, self.meta, self.device
                    )
                else:
                    rgb_full = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    rank, suit, conf = predict_image(
                        Image.fromarray(rgb_full),
                        self.model,
                        self.meta,
                        self.device,
                    )
            else:
                rgb_full = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rank, suit, conf = predict_image(
                    Image.fromarray(rgb_full), self.model, self.meta, self.device
                )
            if conf < 0.12 if hole_mode else conf < 0.20:
                continue
            labeled.append(
                LabeledCard(rank=rank, suit=suit, confidence=conf, bbox=box)
            )
        return labeled

    @staticmethod
    def _dedupe_labels(
        cards: Sequence[LabeledCard], *, max_n: int
    ) -> List[LabeledCard]:
        """Keep highest-confidence unique labels, left-to-right, capped."""
        by_label: dict[str, LabeledCard] = {}
        for c in cards:
            prev = by_label.get(c.label)
            if prev is None or c.confidence > prev.confidence:
                by_label[c.label] = c
        ordered = sorted(
            by_label.values(),
            key=lambda c: ((c.bbox[0] + c.bbox[2]) * 0.5, -c.confidence),
        )
        return ordered[:max_n]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Smoke: blob + layout + CNN card pipeline on one image"
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="BGR table screenshot (png/jpg)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=default_classifier_path(),
    )
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Failed to read image: {args.image}", file=sys.stderr)
        return 1

    pipe = FastCardsPipeline(checkpoint=args.checkpoint, device=args.device)
    result = pipe.process(frame)
    print(f"layout={result.layout_backend}  blobs={len(result.boxes)}")
    if result.table is not None:
        print(
            f"table: valid={result.table.state_valid} "
            f"conf={result.table.vision_confidence:.3f} "
            f"unc={result.table.uncertainty:.3f}"
        )
    print(f"community: {' '.join(result.community_labels) or '(none)'}")
    print(f"holes:     {' '.join(result.hole_labels) or '(none)'}")
    for kind, cards in (("board", result.community), ("hero", result.holes)):
        for c in cards:
            print(
                f"  {kind} {c.label}  conf={c.confidence:.3f}  box={c.bbox}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
