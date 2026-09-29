"""Per-tick hero hole-card path diagnostics (debug logging only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class HeroSlotDiag:
    """Diagnostics for one hero hole slot this tick."""

    slot_id: str
    box: Optional[Box] = None
    box_stable: Optional[bool] = None
    path: Optional[str] = None  # cheap | enhanced
    cheap_label: Optional[str] = None
    cheap_conf: Optional[float] = None
    cheap_margin: Optional[float] = None
    enhanced_label: Optional[str] = None
    enhanced_conf: Optional[float] = None
    enhanced_margin: Optional[float] = None
    banked_label: Optional[str] = None
    banked_conf: Optional[float] = None
    banked_margin: Optional[float] = None
    reason: str = "UNKNOWN"

    def _fmt_box(self) -> str:
        if self.box is None:
            return "-"
        x1, y1, x2, y2 = self.box
        cx = (int(x1) + int(x2)) // 2
        cy = (int(y1) + int(y2)) // 2
        return f"{cx},{cy}"

    def _fmt_pred(
        self,
        label: Optional[str],
        conf: Optional[float],
        margin: Optional[float],
    ) -> str:
        if label is None or conf is None:
            return "-"
        m = f"{margin:.2f}" if margin is not None else "?"
        return f"{label}@{conf:.2f}/m{m}"

    def format_short(self) -> str:
        stable = (
            "?" if self.box_stable is None else ("1" if self.box_stable else "0")
        )
        parts = [
            f"{self.slot_id}",
            f"box={self._fmt_box()}",
            f"stable={stable}",
            f"cheap={self._fmt_pred(self.cheap_label, self.cheap_conf, self.cheap_margin)}",
        ]
        if self.path == "enhanced" or self.enhanced_label is not None:
            parts.append(
                f"enh={self._fmt_pred(self.enhanced_label, self.enhanced_conf, self.enhanced_margin)}"
            )
        if self.path:
            parts.append(f"path={self.path}")
        parts.append(
            f"bank={self._fmt_pred(self.banked_label, self.banked_conf, self.banked_margin)}"
        )
        parts.append(f"reason={self.reason}")
        return " ".join(parts)


@dataclass
class HeroTickDiag:
    """Accumulated hero-path diagnostics for one detect tick."""

    hero_blobs: int = 0
    filtered_blobs: int = 0
    raw_blobs: int = 0
    candidate_boxes: List[Box] = field(default_factory=list)
    slots: List[HeroSlotDiag] = field(default_factory=list)
    stage_reason: Optional[str] = None  # no_blobs | one_blob | filter_white_low | …

    def set_stage_from_counts(self) -> None:
        """Derive early-stage reason when holes are not yet classifiable."""
        if self.stage_reason:
            return
        if self.hero_blobs <= 0:
            if self.raw_blobs > 0 and self.filtered_blobs < self.raw_blobs:
                self.stage_reason = "filter_white_low"
            else:
                self.stage_reason = "no_blobs"
        elif self.hero_blobs == 1:
            self.stage_reason = "one_blob"

    def format_line(self) -> str:
        """Compact single-line summary for console / optional log."""
        self.set_stage_from_counts()
        boxes_s = ",".join(
            f"{(b[0] + b[2]) // 2},{(b[1] + b[3]) // 2}" for b in self.candidate_boxes
        ) or "-"
        head = (
            f"[holes] hero={self.hero_blobs}/filt={self.filtered_blobs}"
            f"/raw={self.raw_blobs} boxes=[{boxes_s}]"
        )
        if self.stage_reason and not self.slots:
            return f"{head} reason={self.stage_reason}"
        if not self.slots:
            reason = self.stage_reason or "UNKNOWN"
            return f"{head} reason={reason}"
        slot_bits = " | ".join(s.format_short() for s in self.slots)
        return f"{head} {slot_bits}"


def boxes_short(boxes: Sequence[Box]) -> str:
    """Short cx,cy list for candidate boxes."""
    if not boxes:
        return "-"
    return ",".join(
        f"{(int(b[0]) + int(b[2])) // 2},{(int(b[1]) + int(b[3])) // 2}"
        for b in boxes
    )
