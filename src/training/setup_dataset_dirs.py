"""
Create directory structure for card dataset:
  data/cards/by_card/<card>/   for each of 52 cards (2h, 2d, ... As)
  data/cards/models/           for trained .pt files
"""
from pathlib import Path

from src.training.card_names import CARD_NAMES


def setup_dirs(root: Path = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data" / "cards"
    root = Path(root)
    by_card = root / "by_card"
    models = root / "models"
    by_card.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for name in CARD_NAMES:
        (by_card / name).mkdir(parents=True, exist_ok=True)
    print(f"Created {len(CARD_NAMES)} card folders under {by_card}")
    print(f"Created models dir: {models}")


if __name__ == "__main__":
    setup_dirs()
