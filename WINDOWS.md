# Running Poker Vision on Windows

## Quick Start

1. **Run setup:**
   ```bat
   setup.bat
   ```

2. **Run the app:**
   ```bat
   venv\Scripts\python -m src.main --console
   ```

   Or with overlay:
   ```bat
   venv\Scripts\python -m src.main
   ```

## First-Time Setup (Manual)

If you prefer doing it manually:

```powershell
# Create venv
python -m venv venv
venv\Scripts\Activate.ps1

# Install deps
pip install mss numpy opencv-python Pillow ultralytics eval7 pywin32

# Run
python -m src.main --console
```

## Table amounts (pot / to_call / stack)

Stake shows chip amounts in **BB** (big blinds), e.g. `Pot: 8 BB`, `81 BB`,
`222.5 BB` — not dollars. Optional OCR uses **pytesseract** plus a system
**Tesseract** install (`table_amounts.py` also auto-detects
`C:\Program Files\Tesseract-OCR\tesseract.exe`):

```powershell
pip install pytesseract
# Install Tesseract OCR for Windows, then ensure `tesseract.exe` is on PATH
# https://github.com/UB-Mannheim/tesseract/wiki
```

Without Tesseract, card detection still works; amounts stay `?` and actions
fall back to the win% ladder (no pot-odds pricing).

Debug raw OCR text every few seconds:

```powershell
$env:POKER_VISION_OCR_DEBUG = "1"
venv\Scripts\python -m src.main --console
# or: venv\Scripts\python -m src.main --debug
```

## Configuration

Before running, edit `src/capture/screen.py` and adjust:

```python
def get_poker_table_region(self) -> Tuple[int, int, int, int]:
    # Set to your Stake.us table position
    # (x, y, width, height)
    return (100, 100, 1200, 800)
```

To find coordinates: Move your mouse to the top-left and bottom-right of the poker table and note the positions.

## Testing Screen Capture

```powershell
venv\Scripts\python -c "from src.capture.screen import ScreenCapture; cap = ScreenCapture(); frame = cap.capture_region(0, 0, 800, 600); print(f'Captured: {frame.shape}')"
```

This should print the frame shape if working correctly.
