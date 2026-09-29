"""One-off diagnostic: blobs/layout/pipeline on a fixture image."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detection.blobs import find_card_blobs
from src.detection.layout import (
    filter_card_like_boxes,
    select_community_row,
    split_community_and_holes,
    select_hero_hole_pair,
    _face_up_score_robust,
)
from src.detection.pipeline import FastCardsPipeline


def main() -> int:
    img_path = _ROOT / "tests" / "fixtures" / "stake_turn_4card_holes_down.png"
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"failed to read {img_path}")
        return 1
    h, w = frame.shape[:2]
    print(f"image {w}x{h}")

    raw = find_card_blobs(frame)
    print(f"raw_blobs={len(raw)}")
    for i, b in enumerate(raw):
        print(f"  raw[{i}] box={b} white={_face_up_score_robust(frame, b):.3f}")

    filt = filter_card_like_boxes(raw, frame)
    print(f"filtered={len(filt)}")
    for i, b in enumerate(filt):
        print(f"  filt[{i}] box={b} white={_face_up_score_robust(frame, b):.3f}")

    comm, holes = split_community_and_holes(filt, frame.shape)
    row = select_community_row(filt, frame.shape)
    hero = select_hero_hole_pair(holes, frame)
    print(f"community_row={len(row)} boxes={[list(b) for b in row]}")
    print(f"community_boxes={len(comm)} {[list(b) for b in comm]}")
    print(f"hole_candidates={len(holes)} hero={[list(b) for b in hero]}")

    for use_tracker in (True, False):
        pipe = FastCardsPipeline(use_tracker=use_tracker)
        r = pipe.process(frame)
        print(f"\n--- use_tracker={use_tracker} ---")
        print(f"layout={r.layout_backend} blobs={len(r.boxes)}")
        print(f"community_boxes={len(r.community_boxes)} {r.community_boxes}")
        print(f"hole_boxes={len(r.hole_boxes)} {r.hole_boxes}")
        print(f"community_labels={r.community_labels}")
        print(f"hole_labels={r.hole_labels}")
        if r.table is not None:
            print(
                f"table valid={r.table.state_valid} street={r.table.street} "
                f"board_vis={[ (o.slot_id, o.visibility, o.label) for o in r.table.board ]}"
            )
            print(
                f"hero_vis={[ (o.slot_id, o.visibility, o.label) for o in r.table.hero ]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
