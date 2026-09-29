"""Hole-card inference: avatar-dim restore + fixture smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.classification.infer import _clahe_bgr, _suit_pip_crop


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "stake_full_table_rhyzome.jpg"
CKPT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cards"
    / "models"
    / "classifier.pt"
)


def test_clahe_raises_dim_crop_contrast():
    """CLAHE should lift a dark gray face toward training-like contrast."""
    dim = np.full((80, 48, 3), 90, dtype=np.uint8)
    # Dark ink blob in UL (fake rank)
    dim[8:28, 6:22] = 40
    out = _clahe_bgr(dim, clip=2.0)
    assert out.shape == dim.shape
    assert float(out.std()) > float(dim.std())


def test_suit_pip_crop_geometry():
    card = np.zeros((100, 50, 3), dtype=np.uint8)
    pip = _suit_pip_crop(card)
    # y0=18, y1=52 → h=34; x1=26
    assert pip.shape == (34, 26, 3)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="rhyzome fixture missing")
@pytest.mark.skipif(not CKPT.is_file() or CKPT.stat().st_size < 1000, reason="classifier.pt missing")
def test_rhyzome_avatar_holes_read_9c_2h():
    """Mid-right avatar-dimmed holes: layout boxes stay, labels → 9c 2h."""
    import cv2

    from src.detection.pipeline import FastCardsPipeline

    frame = cv2.imread(str(FIXTURE))
    assert frame is not None
    pipe = FastCardsPipeline(checkpoint=CKPT, reuse_stable_boxes=False)
    result = pipe.process(frame)

    assert len(result.hole_boxes) == 2
    # Geometry: mid-right pair (layout must not regress)
    for box in result.hole_boxes:
        cx = 0.5 * (box[0] + box[2]) / frame.shape[1]
        cy = 0.5 * (box[1] + box[3]) / frame.shape[0]
        assert cx >= 0.65
        assert cy <= 0.35

    labels = [c.label for c in result.holes]
    assert labels == ["9c", "2h"], f"got {labels} boxes={result.hole_boxes}"
