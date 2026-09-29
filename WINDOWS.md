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
