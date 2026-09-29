"""
Card detection module using YOLO.
Has mock mode for testing without YOLO.
"""

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class DetectedCard:
    """A detected playing card."""
    rank: str  # 2-10, J, Q, K, A
    suit: str  # h, d, c, s (hearts, diamonds, clubs, spades)
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    def __str__(self):
        return f"{self.rank}{self.suit.upper()} ({self.confidence:.2f})"


class MockDetector:
    """Mock detector for testing without YOLO."""
    
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    SUITS = ["h", "d", "c", "s"]
    
    def __init__(self, model_path: str = None):
        self.confidence_threshold = 0.7
        self._frame_count = 0
    
    def detect(self, frame: np.ndarray) -> List[DetectedCard]:
        """Generate mock cards based on frame."""
        self._frame_count += 1
        h, w = frame.shape[:2]
        
        cards = []
        
        # Community cards (top center) - 3 to 5 cards
        if self._frame_count % 10 < 7:  # 70% chance of having community cards
            num_community = 3 + (self._frame_count % 3)  # 3, 4, or 5
            card_w, card_h = 60, 84
            start_x = w // 2 - (num_community * card_w + 20) // 2
            
            # Fixed board — never overlaps hole cards (As/Ks).
            board = ["Tc", "Jd", "2h", "9d", "8c"][:num_community]
            for i, card_str in enumerate(board):
                cx = start_x + i * (card_w + 5)
                cy = h // 4
                cards.append(DetectedCard(
                    rank=card_str[0],
                    suit=card_str[1],
                    confidence=0.95,
                    bbox=(cx, cy, cx + card_w, cy + card_h)
                ))
        
        # Hole cards (bottom center) — always AsKs, disjoint from board above
        hole_ranks = ["As", "Ks"]
        card_w, card_h = 60, 84
        for i, card_str in enumerate(hole_ranks):
            cx = w // 2 - card_w - 20 + i * (card_w + 40)
            cy = h - card_h - 50
            cards.append(DetectedCard(
                rank=card_str[0],
                suit=card_str[1],
                confidence=0.95,
                bbox=(cx, cy, cx + card_w, cy + card_h)
            ))
        
        return cards


# Rank-major order matching data/cards/cards.yaml and src/training/card_names.py:
# class 0=2h, 1=2d, 2=2c, 3=2s, 4=3h, ... 51=As
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["h", "d", "c", "s"]
CARD_NAMES = [r + s for r in RANKS for s in SUITS]


def decode_class_id(cls_id: int) -> Tuple[str, str]:
    """Map YOLO class id to (rank, suit). Returns ('?', '?') if out of range."""
    if not 0 <= cls_id < len(CARD_NAMES):
        return "?", "?"
    name = CARD_NAMES[cls_id]
    return name[0], name[1]


class CardDetector:
    """Detects playing cards in screen frames using YOLO."""

    CARD_NAMES = CARD_NAMES

    # Default model is generic YOLO (no card classes). For card detection, train on Stake.us card images and pass your .pt path.
    DEFAULT_MODEL = "yolo11n.pt"

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = self.DEFAULT_MODEL
        if not ULTRALYTICS_AVAILABLE:
            print("WARNING: ultralytics not installed, using mock detector")
            self._mock = MockDetector(model_path)
            self.model = None
            return

        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5  # Lower than 0.7 so a custom card-trained model picks up more detections
        if model_path == self.DEFAULT_MODEL:
            print(
                "NOTE: yolo11n.pt is a generic model (no playing-card classes). "
                "It will not detect cards until you train a model on card images. See README 'Training YOLO Model'."
            )
    
    def detect(self, frame: np.ndarray) -> List[DetectedCard]:
        if self.model is None:
            return self._mock.detect(frame)
            
        results = self.model(frame, conf=self.confidence_threshold)
        
        cards = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                rank, suit = decode_class_id(cls_id)
                
                cards.append(DetectedCard(
                    rank=rank,
                    suit=suit,
                    confidence=conf,
                    bbox=bbox
                ))
        
        return cards
    
    def classify_community_vs_holecards(self, cards: List[DetectedCard]) -> Tuple[List[DetectedCard], List[DetectedCard]]:
        if len(cards) < 2:
            return [], []
        
        sorted_by_y = sorted(cards, key=lambda c: c.bbox[1])
        y_threshold = sorted_by_y[0].bbox[1] + 100
        
        community = [c for c in cards if c.bbox[1] < y_threshold]
        holecards = [c for c in cards if c.bbox[1] >= y_threshold]
        
        return community, holecards
