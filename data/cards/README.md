# Playing card dataset for YOLO training

This folder holds images for training a custom YOLO model to detect playing cards (Stake.us style or similar).

## Directory layout

- **`by_card/`** – 52 subfolders, one per card. Put multiple images of each card here (in hand, on table, different angles, etc.).
- **`models/`** – Trained `.pt` model files go here (created when you run training).

### Card folder names (must match exactly)

Ranks: 2, 3, 4, 5, 6, 7, 8, 9, T (ten), J, Q, K, A  
Suits: h (hearts), d (diamonds), c (clubs), s (spades)

Examples: `2h`, `2d`, `2c`, `2s`, `Th`, `As`.

## Workflow

1. **Collect images**  
   Screenshot cards from Stake.us (or your target site). You can use a single table screenshot and crop out each card.

2. **Sort into `by_card/`**  
   - **Option A (interactive crop):** Run the cropping tool on a screenshot. It opens the image; you click two corners per card, enter the card name (e.g. `Ah`, `Td`), and each crop is saved into the right folder:
     ```bash
     python -m src.training.crop_cards_from_image path/to/your_screenshot.png
     ```
     Use two clicks per card (e.g. top-left then bottom-right). Enter card name as rank+suit; 10 = `T` (e.g. `Th`). Press `q` in the window to quit, `c` to clear the current selection.
   - **Option B:** Manually crop and save images into each card folder (e.g. all Ace of Spades in `by_card/As/`).

3. **Create YOLO dataset**  
   From project root:
   ```bash
   python -m src.training.prepare_dataset
   ```
   This reads `data/cards/by_card/` and builds `data/cards/yolo_dataset/` (train/val images + labels) and updates `data/cards/cards.yaml`.

4. **Train the model**  
   ```bash
   python -m src.training.train
   ```
   Or with Ultralytics CLI:
   ```bash
   yolo detect train data=data/cards/cards.yaml model=yolo11n.pt epochs=50 project=data/cards/models name=run1
   ```

5. **Use the trained model**  
   Point the app at your best weights, e.g. `data/cards/models/run1/weights/best.pt`, via the `--model` flag or by updating the default in code.

## Tips

- Aim for at least ~20–50 images per card; more variety (in hand, on table, different lighting) helps.
- Use the same card art as your target (e.g. Stake.us) so the model matches what it will see at runtime.
- After training, put `best.pt` (or your chosen weights) in `data/cards/models/` and reference it when running the app.
