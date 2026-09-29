"""Temporal IoU association + EMA probability banking for card detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.state.types import (
    BOX_STABLE_IOU,
    EMA_ALPHA,
    IOU_MATCH,
    LOW_MARGIN_WEIGHT,
    MARGIN_WEAK,
    MAX_MISSED_FRAMES,
    TRANSITIONING,
    UNKNOWN,
    UNKNOWN_LABEL,
    VISIBLE,
    Box,
    CardObservation,
    RawDet,
    normalize_probs,
    rank_margin,
    suit_margin,
    top_from_probs,
)


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-union for axis-aligned boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _obs_quality(rank_probs: np.ndarray) -> float:
    if rank_margin(rank_probs) < MARGIN_WEAK:
        return LOW_MARGIN_WEIGHT
    return 1.0


@dataclass
class _Track:
    track_id: int
    slot_id: str
    bbox: Box
    rank_probs: np.ndarray
    suit_probs: np.ndarray
    hole_mode: bool
    frame_idx: int
    missed: int = 0
    first_seen: int = 0
    last_seen: int = 0
    ever_quality: bool = False
    low_quality_streak: int = 0
    prev_bbox: Optional[Box] = None
    stable_frames: int = 0
    recent_labels: List[str] = field(default_factory=list)
    last_frame_label: str = UNKNOWN_LABEL

    def apply_detection(self, det: RawDet, frame_idx: int) -> None:
        quality = _obs_quality(det.rank_probs)
        w = EMA_ALPHA * quality
        self.rank_probs = normalize_probs(
            (1.0 - w) * self.rank_probs + w * det.rank_probs, n=13
        )
        self.suit_probs = normalize_probs(
            (1.0 - w) * self.suit_probs + w * det.suit_probs, n=4
        )

        self.prev_bbox = self.bbox
        self.bbox = det.bbox
        self.hole_mode = det.hole_mode
        if det.slot_hint:
            self.slot_id = det.slot_hint
        self.missed = 0
        self.last_seen = frame_idx
        self.frame_idx = frame_idx

        if quality >= 1.0:
            self.ever_quality = True
            self.low_quality_streak = 0
        else:
            self.low_quality_streak += 1

        r, s, _ = top_from_probs(self.rank_probs, self.suit_probs)
        banked_label = f"{r}{s}"
        if banked_label == self.last_frame_label and self.ever_quality:
            self.stable_frames += 1
        else:
            self.stable_frames = 1 if self.ever_quality else 0
        self.last_frame_label = banked_label
        self.recent_labels.append(banked_label)
        if len(self.recent_labels) > 8:
            self.recent_labels = self.recent_labels[-8:]

    def visibility(self) -> str:
        if not self.ever_quality:
            return UNKNOWN
        box_ok = True
        if self.prev_bbox is not None:
            box_ok = box_iou(self.prev_bbox, self.bbox) >= BOX_STABLE_IOU
        if (not box_ok) or self.low_quality_streak >= 2:
            return TRANSITIONING
        return VISIBLE

    def to_observation(self) -> CardObservation:
        vis = self.visibility()
        box_stable = True
        if self.prev_bbox is not None:
            box_stable = box_iou(self.prev_bbox, self.bbox) >= BOX_STABLE_IOU

        r, s, conf = top_from_probs(self.rank_probs, self.suit_probs)
        label = f"{r}{s}" if vis == VISIBLE else UNKNOWN_LABEL

        # Temporal agreement: banked top vs most recent frame top in history
        temporal = 0.0
        if self.recent_labels:
            agree = sum(1 for lab in self.recent_labels if lab == f"{r}{s}")
            temporal = float(agree) / float(len(self.recent_labels))

        return CardObservation(
            slot_id=self.slot_id,
            bbox=self.bbox,
            rank_probs=self.rank_probs.copy(),
            suit_probs=self.suit_probs.copy(),
            visibility=vis,
            label=label,
            confidence=conf if vis == VISIBLE else 0.0,
            rank_margin=rank_margin(self.rank_probs),
            suit_margin=suit_margin(self.suit_probs),
            stable_frames=self.stable_frames,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            box_stable=box_stable,
            temporal_agreement=temporal,
            hole_mode=self.hole_mode,
            recent_labels=list(self.recent_labels),
        )


class CardTracker:
    """Associate RawDet boxes across frames and bank EMA class distributions."""

    def __init__(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        self._frame = 0

    def update(self, observations: List[RawDet]) -> List[CardObservation]:
        frame_idx = self._frame
        self._frame += 1

        matches = self._associate(observations)
        matched_track_ids: Set[int] = set()
        matched_det_idxs: Set[int] = set()

        for tid, di in matches:
            track = self._tracks[tid]
            track.apply_detection(observations[di], frame_idx)
            matched_track_ids.add(tid)
            matched_det_idxs.add(di)

        for di, det in enumerate(observations):
            if di in matched_det_idxs:
                continue
            self._spawn(det, frame_idx, order_index=di)

        drop_ids: List[int] = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids:
                continue
            # Newly spawned this frame are already in matched via spawn path;
            # unmatched existing tracks miss.
            if track.last_seen == frame_idx:
                continue
            track.missed += 1
            if track.missed >= MAX_MISSED_FRAMES:
                drop_ids.append(tid)

        for tid in drop_ids:
            del self._tracks[tid]

        # Emit currently seen tracks (matched or newly spawned this frame).
        out: List[CardObservation] = []
        for tid in sorted(self._tracks.keys()):
            track = self._tracks[tid]
            if track.last_seen != frame_idx:
                continue
            out.append(track.to_observation())
        return out

    def _spawn(self, det: RawDet, frame_idx: int, order_index: int) -> _Track:
        tid = self._next_id
        self._next_id += 1
        slot_id = det.slot_hint if det.slot_hint else f"temp_{order_index}"
        quality = _obs_quality(det.rank_probs)
        r, s, _ = top_from_probs(det.rank_probs, det.suit_probs)
        label = f"{r}{s}"
        track = _Track(
            track_id=tid,
            slot_id=slot_id,
            bbox=det.bbox,
            rank_probs=normalize_probs(det.rank_probs, n=13).copy(),
            suit_probs=normalize_probs(det.suit_probs, n=4).copy(),
            hole_mode=det.hole_mode,
            frame_idx=frame_idx,
            missed=0,
            first_seen=frame_idx,
            last_seen=frame_idx,
            ever_quality=quality >= 1.0,
            low_quality_streak=0 if quality >= 1.0 else 1,
            prev_bbox=None,
            stable_frames=1 if quality >= 1.0 else 0,
            recent_labels=[label],
            last_frame_label=label,
        )
        self._tracks[tid] = track
        return track

    def _associate(self, observations: List[RawDet]) -> List[Tuple[int, int]]:
        """Greedy IoU matching; prefer same hole_mode cluster first."""
        if not self._tracks or not observations:
            return []

        track_ids = list(self._tracks.keys())
        pairs: List[Tuple[float, int, int, bool]] = []
        for ti, tid in enumerate(track_ids):
            track = self._tracks[tid]
            for di, det in enumerate(observations):
                iou = box_iou(track.bbox, det.bbox)
                if iou < IOU_MATCH:
                    continue
                same = track.hole_mode == det.hole_mode
                pairs.append((iou, ti, di, same))

        # Same-cluster first, then IoU desc.
        pairs.sort(key=lambda t: (0 if t[3] else 1, -t[0]))

        used_t: Set[int] = set()
        used_d: Set[int] = set()
        matches: List[Tuple[int, int]] = []
        for _iou, ti, di, _same in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((track_ids[ti], di))
        return matches
