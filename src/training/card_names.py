"""
Card class names and IDs for YOLO (must match CardDetector in detection/cards.py).
Order: 2h, 2d, 2c, 2s, 3h, ... Ah, Ad, Ac, As  ->  class_id 0..51
"""
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["h", "d", "c", "s"]  # hearts, diamonds, clubs, spades

def get_card_names():
    """Return list of 52 card names in YOLO class order (class_id = index)."""
    return [r + s for r in RANKS for s in SUITS]

CARD_NAMES = get_card_names()
NUM_CLASSES = len(CARD_NAMES)
