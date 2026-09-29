# Poker Vision

Computer vision poker assistant for Stake.us. Captures screen, detects cards, and provides hand recommendations from Monte Carlo win% (no table amount OCR).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run in console mode (test first)
python -m src.main --console

# Run with overlay
python -m src.main
```

## Architecture

```
poker-vision/
├── data/
│   └── cards/           # Dataset & trained model
│       ├── by_card/     # 52 folders (2h, 2d, ... As) – put images per card here
│       ├── models/      # Trained .pt files go here
│       ├── cards.yaml   # YOLO dataset config
│       └── README.md    # Dataset instructions
├── src/
│   ├── capture/         # Screen capture
│   ├── detection/      # YOLO card detection
│   ├── engine/         # Hand evaluation (eval7)
│   ├── overlay/        # Display overlay
│   ├── training/       # Dataset setup & prep
│   └── main.py         # Main application
└── requirements.txt
```

## Training YOLO Model

The default detector uses a generic YOLO model (no card classes). To detect cards:

1. **Create dataset dirs** (once):
   ```bash
   python -m src.training.setup_dataset_dirs
   ```

2. **Collect images** into `data/cards/by_card/<card>/` (e.g. `2h`, `As`). Put multiple images per card (in hand, on table, etc.). Use Stake.us screenshots for best results.

3. **Build YOLO dataset** (after adding images):
   ```bash
   python -m src.training.prepare_dataset
   ```

4. **Train**:
   ```bash
   yolo detect train data=data/cards/cards.yaml model=yolo11n.pt epochs=50 project=data/cards/models name=run1
   ```

5. **Use your model**: Run the app with the trained weights, e.g. `data/cards/models/run1/weights/best.pt` (via `--model` or by updating the default in code).

## Configuration

Adjust in `src/capture/screen.py`:
- `get_poker_table_region()` - Set the screen region to capture

## Requirements

- Windows 10/11
- Python 3.9+
- Stake.us open in browser

## License

Non-commercial use only, with permanent credit required. See [LICENSE](LICENSE).

## Legal Note

This tool is for personal use at sweepstakes poker sites. Check Stake.us terms of service. Not affiliated with Stake.
