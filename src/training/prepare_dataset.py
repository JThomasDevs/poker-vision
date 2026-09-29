"""
Build YOLO-format dataset from data/cards/by_card/ (52 folders).
Copies images into yolo_dataset/train and yolo_dataset/val with one label file per image.
Assumes each image in a card folder is a crop of that card (one object per image);
the label is one box covering the full image.
For full-table screenshots with multiple cards, annotate with LabelImg/Roboflow and export to the same structure.
"""
import random
from pathlib import Path

from src.training.card_names import CARD_NAMES


def prepare_dataset(
    cards_root: Path = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    if cards_root is None:
        cards_root = Path(__file__).resolve().parents[2] / "data" / "cards"
    cards_root = Path(cards_root)
    by_card = cards_root / "by_card"
    out_root = cards_root / "yolo_dataset"
    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    val_img = out_root / "val" / "images"
    val_lbl = out_root / "val" / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    card_to_id = {name: i for i, name in enumerate(CARD_NAMES)}
    total = 0
    for card_name in CARD_NAMES:
        folder = by_card / card_name
        if not folder.is_dir():
            continue
        class_id = card_to_id[card_name]
        images = list(folder.iterdir())
        images = [p for p in images if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
        rng.shuffle(images)
        n_val = max(0, int(len(images) * val_ratio))
        n_train = len(images) - n_val
        for i, img_path in enumerate(images):
            is_val = i < n_val
            dst_img = (val_img if is_val else train_img) / f"{card_name}_{img_path.stem}{img_path.suffix}"
            dst_lbl = (val_lbl if is_val else train_lbl) / f"{card_name}_{img_path.stem}.txt"
            # Copy image (or symlink to save space; copy is simpler)
            import shutil
            shutil.copy2(img_path, dst_img)
            # YOLO label: one line per object: class_id x_center y_center width height (normalized 0-1)
            # Full image = one box at center 0.5,0.5 size 1,1
            with open(dst_lbl, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            total += 1
    print(f"Prepared {total} images in {out_root} (train: {train_img}, val: {val_img})")
    print("Run training with: yolo detect train data=data/cards/cards.yaml model=yolo11n.pt epochs=50 project=data/cards/models name=run1")


if __name__ == "__main__":
    prepare_dataset()
