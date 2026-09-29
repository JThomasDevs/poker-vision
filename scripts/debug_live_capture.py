"""Capture the real Stake.us poker Chrome window (not Gradio)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.capture.screen import ScreenCapture
from src.detection.blobs import filter_by_aspect_ratio, find_card_blobs
from src.detection.pipeline import FastCardsPipeline


def score_window(w: dict) -> float:
    title = w["title"].lower()
    left, top, right, bottom = w["rect"]
    area = max(0, right - left) * max(0, bottom - top)
    score = float(area)
    # Prefer live Stake poker; demote Gradio / identify tools.
    if "127.0.0.1" in title or "localhost" in title or "card identify" in title:
        score -= 1e12
    if "stake.us" in title or "stake.com" in title:
        score += 1e9
    if "poker" in title or "hold" in title:
        score += 1e8
    if "chrome" in title:
        score += 1e6
    return score


def main() -> int:
    caps = ScreenCapture()
    wins = caps.list_windows()
    ranked = sorted(wins, key=score_window, reverse=True)
    print("top windows:")
    for w in ranked[:8]:
        print(f"  score={score_window(w):.0f} {w['title'][:80]!r} {w['rect']}")

    target = ranked[0]
    if score_window(target) < 0:
        print("no good Stake poker window")
        return 1

    print("TARGET", target["title"][:100])
    caps.select_window(target["hwnd"])
    pw = caps.capture()
    left, top, right, bottom = target["rect"]
    mss = caps._capture_region(left, top, max(1, right - left), max(1, bottom - top))

    out = _ROOT / "data"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "live_table_pw.png"), pw)
    cv2.imwrite(str(out / "live_table_mss.png"), mss)
    print("pw", pw.shape, float(pw.mean()), "mss", mss.shape, float(mss.mean()))

    pipe = FastCardsPipeline()
    for name, frame in (("pw", pw), ("mss", mss)):
        raw = find_card_blobs(frame)
        filt = filter_by_aspect_ratio(raw, min_aspect=0.35, max_aspect=0.85)
        result = pipe.process(frame)
        print(name, "raw", len(raw), "filt", len(filt))
        for b in raw[:20]:
            x1, y1, x2, y2 = b
            print(
                f"  {b} aspect={(x2-x1)/max(1,y2-y1):.2f} "
                f"cy={(y1+y2)/2/frame.shape[0]:.2f}"
            )
        print(
            f"  holes={result.hole_labels} board={result.community_labels} "
            f"comm_boxes={len(result.community_boxes)} "
            f"hole_boxes={len(result.hole_boxes)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
