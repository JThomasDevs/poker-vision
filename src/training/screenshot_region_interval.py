"""
Select a screen region (two clicks), then screenshot that region every N seconds
and save to the project assets folder.

Usage:
  python -m src.training.screenshot_region_interval [--interval N] [--output DIR]

  --interval N   Seconds between captures (default: 5)
  --output DIR   Folder to save images (default: assets)
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mss


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def select_region() -> tuple:
    """Show full screen, user clicks two corners. Returns (x1, y1, x2, y2) in screen coords."""
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # all monitors combined
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]
    h, w = img.shape[:2]
    # Scale down for display if huge
    max_display = 1600
    scale = 1.0
    if w > max_display or h > max_display:
        scale = min(max_display / w, max_display / h)
        display = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        display = img.copy()
    disp_h, disp_w = display.shape[:2]
    points = []

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))

    win = "Select region: click top-left, then bottom-right. Q=cancel"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cv2.putText(display, "Click top-left corner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    while True:
        show = display.copy()
        if len(points) == 1:
            cv2.circle(show, points[0], 6, (0, 255, 0), -1)
            cv2.putText(show, "Click bottom-right corner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(show, "Press Enter to start, R to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, show)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return None
        if key == ord("r"):
            points.clear()
        if key == 13:  # Enter
            if len(points) >= 2:
                break
            continue
    cv2.destroyAllWindows()
    x1, y1 = points[0]
    x2, y2 = points[1]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # Scale back to full screen coords if we scaled for display
    if scale != 1.0:
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
    return (x1, y1, x2, y2)


def run_capture_loop(region: tuple, interval_sec: float, output_dir: Path):
    """Capture region every interval_sec, save to output_dir. Ctrl+C to stop."""
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1
    mss_region = {"left": x1, "top": y1, "width": width, "height": height}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir} every {interval_sec}s. Press Ctrl+C to stop.")
    count = 0
    with mss.mss() as sct:
        while True:
            try:
                screenshot = sct.grab(mss_region)
                img = np.array(screenshot)[:, :, :3]
                count += 1
                name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{count:04d}.png"
                path = output_dir / name
                cv2.imwrite(str(path), img)
                print(f"Saved {path.name}")
            except KeyboardInterrupt:
                break
            import time
            time.sleep(interval_sec)
    print("Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Select screen region, then screenshot it every N seconds.")
    parser.add_argument("--interval", "-i", type=float, default=5.0, help="Seconds between captures (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output folder (default: project assets)")
    args = parser.parse_args()
    root = get_project_root()
    output_dir = Path(args.output) if args.output else root / "assets"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    region = select_region()
    if region is None:
        print("Cancelled.")
        sys.exit(0)
    run_capture_loop(region, args.interval, output_dir)


if __name__ == "__main__":
    main()
