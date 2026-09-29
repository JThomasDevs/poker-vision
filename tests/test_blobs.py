"""Smoke tests for white-on-blue card blob finding."""

import numpy as np

from src.detection.blobs import (
    filter_by_aspect_ratio,
    filter_by_min_area,
    find_card_blobs,
)


def _blue_felt(h: int = 400, w: int = 600) -> np.ndarray:
    """BGR blue felt background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (160, 90, 30)  # B, G, R — blue-dominant
    return img


def _paint_white_card(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    img[y1:y2, x1:x2] = (220, 220, 220)


def test_find_card_blobs_finds_white_rect_on_blue():
    img = _blue_felt()
    # Portrait card ~70x100 at a known place
    _paint_white_card(img, 200, 120, 270, 220)

    boxes = find_card_blobs(img)
    assert len(boxes) >= 1

    x1, y1, x2, y2 = boxes[0]
    # Blob should overlap the painted card substantially
    assert x1 < 250 < x2
    assert y1 < 170 < y2
    assert (x2 - x1) >= 40
    assert (y2 - y1) >= 60


def test_find_card_blobs_empty_felt():
    img = _blue_felt()
    assert find_card_blobs(img) == []


def test_filter_by_aspect_ratio():
    boxes = [(0, 0, 70, 100), (0, 0, 200, 100)]  # 0.7 ok, 2.0 too wide
    kept = filter_by_aspect_ratio(boxes, min_aspect=0.4, max_aspect=0.9)
    assert kept == [(0, 0, 70, 100)]


def test_filter_by_min_area():
    boxes = [(0, 0, 10, 10), (0, 0, 80, 100)]
    kept = filter_by_min_area(boxes, min_area=500)
    assert kept == [(0, 0, 80, 100)]


def test_find_card_blobs_max_aspect_drops_wide():
    img = _blue_felt()
    _paint_white_card(img, 50, 100, 350, 180)  # wide strip, ratio ~3.75
    wide = find_card_blobs(img, max_aspect=None)
    # May or may not detect depending on edges; if present, filter should drop
    filtered = filter_by_aspect_ratio(wide, max_aspect=0.85)
    for x1, y1, x2, y2 in filtered:
        assert (x2 - x1) / float(y2 - y1) <= 0.85
