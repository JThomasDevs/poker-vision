"""Synthetic geometry tests for detection.layout."""

import numpy as np

from src.detection.layout import (
    crops_from_boxes,
    select_hero_hole_pair,
    split_community_and_holes,
    ul_crop,
)


def test_split_community_and_holes_by_y_gap():
    # Community row ~ mid-felt center; hero holes ~ bottom
    community = [(300, 180, 360, 280), (370, 185, 430, 285), (440, 182, 500, 282)]
    holes = [(350, 400, 410, 500), (420, 405, 480, 505)]
    boxes = community + holes
    shape = (600, 800, 3)

    got_c, got_h = split_community_and_holes(boxes, shape)
    assert got_c == community  # already L→R
    assert got_h == holes


def test_split_keeps_side_holes_when_board_has_three():
    """Live bug: flop of 3 must not absorb mid-right hero pair sharing Y band."""
    # Mid-felt flop (centered)
    board = [(300, 200, 360, 300), (370, 205, 430, 305), (440, 202, 500, 302)]
    # Mid-right hole pair (same Y band as board, cx>~0.68 on 800-wide)
    holes = [(560, 190, 620, 290), (630, 195, 690, 295)]
    shape = (600, 800, 3)
    got_c, got_h = split_community_and_holes(board + holes, shape)
    assert got_c == board
    assert got_h == holes


def test_split_bottom_holes_with_board():
    board = [(300, 180, 360, 280), (370, 185, 430, 285), (440, 182, 500, 282)]
    holes = [(350, 400, 410, 500), (420, 405, 480, 505)]
    shape = (600, 800)
    got_c, got_h = split_community_and_holes(board + holes, shape)
    assert got_c == board
    assert got_h == holes


def test_filter_keeps_avatar_dim_hole_seat():
    """Avatar overlay drops full-box white below 0.35; hole seat must still pass."""
    from src.detection.layout import filter_card_like_boxes

    img = np.zeros((500, 700, 3), dtype=np.uint8)
    img[:, :] = (160, 90, 30)
    # Mid-right hole card: bright UL index, dark avatar center → full white ~0.22
    box = (520, 80, 580, 180)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = (40, 40, 50)
    img[y1 : y1 + 35, x1 : x1 + 30] = (220, 220, 220)
    kept = filter_card_like_boxes([box], img)
    assert box in kept


def test_split_single_lower_band_is_holes():
    # Preflop: only two hole cards in the lower half
    holes = [(300, 400, 360, 500), (370, 405, 430, 505)]
    got_c, got_h = split_community_and_holes(holes, (600, 800))
    assert got_c == []
    assert got_h == holes


def test_select_hero_prefers_white_face_up():
    img = np.zeros((500, 600, 3), dtype=np.uint8)
    img[:, :] = (160, 90, 30)  # blue felt
    # Face-up white pair (hero) + dark face-down (villain)
    white_a, white_b = (250, 350, 310, 450), (320, 355, 380, 455)
    dark = (50, 200, 110, 300)
    img[350:450, 250:310] = (220, 220, 220)
    img[355:455, 320:380] = (220, 220, 220)
    img[200:300, 50:110] = (40, 40, 90)

    pair = select_hero_hole_pair([dark, white_b, white_a], image_bgr=img)
    assert pair == [white_a, white_b]


def test_select_hero_left_right_order_without_image():
    a, b = (400, 300, 460, 400), (200, 310, 260, 410)
    assert select_hero_hole_pair([a, b]) == [b, a]


def test_crops_from_boxes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:40, 20:50] = (1, 2, 3)
    crops = crops_from_boxes(img, [(20, 10, 50, 40)])
    assert len(crops) == 1
    assert crops[0].shape == (30, 30, 3)
    assert tuple(crops[0][0, 0]) == (1, 2, 3)


def test_ul_crop_frac():
    card = np.arange(100 * 80 * 3, dtype=np.uint8).reshape(100, 80, 3)
    ul = ul_crop(card, frac=0.45)
    assert ul.shape == (45, 36, 3)
    np.testing.assert_array_equal(ul, card[:45, :36])


def test_hole_index_crop_tighter_than_ul():
    from src.detection.layout import hole_index_crop

    card = np.zeros((100, 80, 3), dtype=np.uint8)
    hi = hole_index_crop(card, frac_h=0.40, frac_w=0.48)
    assert hi.shape == (40, 38, 3)


def test_avatar_contaminated_red_center():
    from src.detection.layout import avatar_contaminated

    card = np.full((120, 80, 3), 210, dtype=np.uint8)  # white face
    # Saturated red blob in the center (avatar hat).
    card[45:95, 25:70] = (40, 40, 200)
    assert avatar_contaminated(card) is True

    clean = np.full((120, 80, 3), 210, dtype=np.uint8)
    assert avatar_contaminated(clean) is False
