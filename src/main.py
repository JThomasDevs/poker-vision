"""
Poker Vision - Main Application
Captures Stake.us poker tables and provides hand recommendations.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from src.capture.screen import ScreenCapture, select_window_interactive
from src.detection.cards import CardDetector, DetectedCard, MockDetector
from src.engine.evaluator import HandResult, PokerEngine
from src.overlay.display import OverlayDisplay, SimpleConsoleDisplay, DebugDisplay
from src.util.cards_format import format_cards_list


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


class PokerVisionApp:
    """Main application controller."""

    def __init__(
        self,
        capture: ScreenCapture = None,
        use_overlay: bool = True,
        use_debug: bool = False,
        capture_interval: float = 1.0,
        confidence_threshold: float = 0.7,
        mock_mode: bool = False,
        fast_cards: bool = False,
        villain_range: str = "strong",
        debug_holes: bool = False,
        debug_holes_log: Optional[Path] = None,
    ):
        self.capture_interval = capture_interval
        self.confidence_threshold = confidence_threshold
        self.use_debug = use_debug
        self.debug_holes = bool(debug_holes) or _env_flag("POKER_VISION_DEBUG_HOLES")
        self.debug_holes_log = Path(debug_holes_log) if debug_holes_log else None
        self.fast_pipeline = None
        self._last_fast_result = None
        # Equity cache: recompute only when cards change
        self._equity_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._equity_result: Optional[HandResult] = None
        self._last_equity_ran = False
        self._timing_printed = False

        # Initialize capture
        if capture:
            self.capture = capture
        else:
            print("Initializing screen capture...")
            self.capture = ScreenCapture(mock_mode=mock_mode)

        if fast_cards:
            from src.detection.pipeline import FastCardsPipeline, default_classifier_path

            ckpt = default_classifier_path()
            print(f"Card path: blobs + CNN ({ckpt.name})")
            self.fast_pipeline = FastCardsPipeline()
            self.detector = None
            self._mock_detector = MockDetector()
        else:
            print("Card path: YOLO (or mock if no ultralytics)")
            self.detector = CardDetector()
            self.detector.confidence_threshold = confidence_threshold
            self._mock_detector = MockDetector()

        print(f"Initializing poker engine (villain_range={villain_range})...")
        self.engine = PokerEngine(n_simulations=500, villain_range=villain_range)

        print("Initializing display...")
        self.use_console = not use_overlay and not use_debug
        if use_debug:
            self.display = DebugDisplay()
        elif use_overlay:
            self.display = OverlayDisplay()
        else:
            self.display = SimpleConsoleDisplay()

        # Start display window if debug
        if use_debug:
            self.display.run()

        self.is_running = False
        self.current_frame = None
        self._mock_notice_printed = False
        print("Perf: equity/classify cached when stable")
        if self.debug_holes:
            print("Hero hole debug: --debug-holes on (per-tick path diagnostics)")

    def _maybe_print_hero_diag(self, fast) -> None:
        """Print compact hero-path diag when flag on or holes still incomplete."""
        if fast is None:
            return
        diag = getattr(fast, "hero_diag", None) or getattr(
            self.fast_pipeline, "last_hero_diag", None
        )
        if diag is None:
            return

        holes_accepted = len(getattr(fast, "holes", []) or [])
        board_n = len(getattr(fast, "community", []) or [])

        # --debug-holes / env: always print each tick.
        # Flag off: optional quiet hint when board cards exist but holes=[].
        if self.debug_holes:
            should_print = True
        else:
            should_print = holes_accepted == 0 and board_n > 0
        if not should_print:
            return

        line = diag.format_line()
        print(line)
        if self.debug_holes_log is not None:
            try:
                with self.debug_holes_log.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                print(f"[holes] log write failed: {exc}")

    def capture_frame(self):
        """Grab one frame only (no detection)."""
        frame = self.capture.capture()
        self.current_frame = frame
        return frame

    def detect_cards(self, frame):
        """Detect/classify cards on an already-captured frame."""
        if getattr(self.capture, "mock_mode", False):
            if not self._mock_notice_printed:
                print("[mock] synthetic cards only — pick a real Stake window to ID cards")
                self._mock_notice_printed = True
            self._last_fast_result = None
            return self._mock_detector.detect(frame)
        if self.fast_pipeline is not None:
            result = self.fast_pipeline.process(frame)
            self._last_fast_result = result
            return result.as_detected_cards()
        cards = self.detector.detect(frame)
        self._last_fast_result = None
        return cards

    def capture_and_detect(self):
        """Capture screen and detect cards. Returns (frame, cards)."""
        frame = self.capture_frame()
        return frame, self.detect_cards(frame)

    def process_cards(self, cards: List[DetectedCard]) -> dict:
        """Process detected cards into hand info."""
        self._last_equity_ran = False
        if len(cards) < 2:
            return None

        # Prefer tracked TableState when present; else layout split from fast path
        fast = self._last_fast_result
        table = getattr(fast, "table", None) if fast is not None else None
        if table is not None:
            if not table.state_valid:
                return None
            hole_strs = [
                o.label
                for o in table.hero
                if o.visibility == "VISIBLE" and o.label and o.label != "??"
            ][:2]
            comm_strs = [
                o.label
                for o in table.board
                if o.visibility == "VISIBLE" and o.label and o.label != "??"
            ][:5]
        elif fast is not None and (fast.holes or fast.community):
            holecards = [c.to_detected() for c in fast.holes]
            community = [c.to_detected() for c in fast.community]
            hole_strs = [f"{c.rank}{c.suit}" for c in holecards[:2]]
            comm_strs = [f"{c.rank}{c.suit}" for c in community[:5]]
        elif self.detector is not None:
            community, holecards = self.detector.classify_community_vs_holecards(cards)
            community = sorted(community, key=lambda c: c.bbox[0])
            holecards = sorted(holecards, key=lambda c: c.bbox[0])
            hole_strs = [f"{c.rank}{c.suit}" for c in holecards[:2]]
            comm_strs = [f"{c.rank}{c.suit}" for c in community[:5]]
        else:
            holecards = list(cards)
            hole_strs = [f"{c.rank}{c.suit}" for c in holecards[:2]]
            comm_strs = []

        if len(hole_strs) < 2:
            return None
        known = hole_strs + comm_strs
        if len(set(known)) != len(known):
            msg = (
                f"[skip] duplicate labels: holes={format_cards_list(hole_strs)} "
                f"board={format_cards_list(comm_strs)}"
            )
            if self.use_console and hasattr(self.display, "set_status"):
                self.display.set_status(msg)
            else:
                print(msg)
            return None

        key = (tuple(hole_strs), tuple(comm_strs))
        if key == self._equity_key and self._equity_result is not None:
            result = self._equity_result
        else:
            result = self.engine.evaluate_hand(hole_strs, comm_strs)
            self._equity_key = key
            self._equity_result = result
            self._last_equity_ran = True

        if result.recommendation == "error":
            msg = (
                f"[skip] eval: {result.hand_type} holes={format_cards_list(hole_strs)} "
                f"board={format_cards_list(comm_strs)}"
            )
            if self.use_console and hasattr(self.display, "set_status"):
                self.display.set_status(msg)
            else:
                print(msg)
            return None

        return {
            "hole_cards": hole_strs,
            "community_cards": comm_strs,
            "result": result,
            "cards": cards,
        }

    def run(self):
        """Main application loop."""
        self.is_running = True
        print("\n" + "=" * 50)
        print("POKER VISION - Running")
        print("Press Ctrl+C to stop")
        print("Need an open table with hole cards visible (lobby = no cards).")
        print("=" * 50 + "\n")

        try:
            while self.is_running:
                loop_t0 = time.perf_counter()

                t0 = time.perf_counter()
                frame = self.capture_frame()
                ms_capture = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                cards = self.detect_cards(frame)
                ms_cards = (time.perf_counter() - t0) * 1000.0

                fast = self._last_fast_result
                self._maybe_print_hero_diag(fast)
                classify_ran = (
                    bool(getattr(self.fast_pipeline, "last_classify_ran", True))
                    if self.fast_pipeline is not None
                    else True
                )

                n_blobs = len(fast.boxes) if fast is not None else len(cards or [])
                hole_labs = (
                    " ".join(fast.hole_labels)
                    if fast is not None
                    else " ".join(f"{c.rank}{c.suit}" for c in (cards or [])[:2])
                )
                board_labs = (
                    " ".join(fast.community_labels) if fast is not None else ""
                )
                h, w = frame.shape[:2] if frame is not None else (0, 0)

                def _display_labs(labs: str) -> str:
                    if not labs:
                        return labs
                    return format_cards_list(labs.split())

                ms_equity = 0.0

                if cards and len(cards) >= 2:
                    t0 = time.perf_counter()
                    hand_info = self.process_cards(cards)
                    ms_equity = (time.perf_counter() - t0) * 1000.0

                    if hand_info:
                        result = hand_info["result"]

                        if self.use_debug:
                            self.display.update(
                                frame=frame,
                                cards=hand_info["cards"],
                                hand_type=result.hand_type,
                                win_prob=result.win_probability,
                                recommendation=result.recommendation,
                                hole_cards=hand_info["hole_cards"],
                                community_cards=hand_info["community_cards"],
                            )
                        else:
                            self.display.update(
                                hand_type=result.hand_type,
                                win_prob=result.win_probability,
                                recommendation=result.recommendation,
                                hole_cards=hand_info["hole_cards"],
                                community_cards=hand_info["community_cards"],
                            )

                        # Overlay/debug: log to terminal. Console owns the screen — no scroll spam.
                        if not self.use_console:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                f"holes={hand_info['hole_cards']} board={hand_info['community_cards']} | "
                                f"{result.hand_type} | "
                                f"Win: {result.win_probability:.1%} | "
                                f"{result.recommendation}"
                            )
                    else:
                        if self.use_console:
                            wait_msg = (
                                f"blobs={n_blobs} holes=[{_display_labs(hole_labs)}] "
                                f"board=[{_display_labs(board_labs)}] "
                                f"- waiting for 2 hole cards"
                            )
                        else:
                            wait_msg = (
                                f"blobs={n_blobs} holes=[{hole_labs}] board=[{board_labs}] "
                                f"- waiting for 2 hole cards"
                            )
                        if self.use_console:
                            hole_list = hole_labs.split() if hole_labs else None
                            board_list = board_labs.split() if board_labs else None
                            self.display.clear(
                                hole_cards=hole_list,
                                community_cards=board_list,
                                message=wait_msg,
                            )
                        else:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                f"{w}x{h} {wait_msg}"
                            )
                else:
                    if self.use_console:
                        wait_msg = (
                            f"blobs={n_blobs} holes=[{_display_labs(hole_labs) or '-'}] "
                            f"board=[{_display_labs(board_labs) or '-'}] - no cards yet"
                        )
                    else:
                        wait_msg = (
                            f"blobs={n_blobs} holes=[{hole_labs or '-'}] "
                            f"board=[{board_labs or '-'}] - no cards yet"
                        )
                    if self.use_debug:
                        self.display.update(
                            frame=frame,
                            cards=[],
                            hand_type="No cards detected",
                            win_prob=0.0,
                            recommendation="-",
                            hole_cards=None,
                            community_cards=None,
                        )
                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{w}x{h} {wait_msg}"
                        )
                    elif self.use_console:
                        hole_list = hole_labs.split() if hole_labs else None
                        board_list = board_labs.split() if board_labs else None
                        self.display.clear(
                            hole_cards=hole_list,
                            community_cards=board_list,
                            message=wait_msg,
                        )
                    else:
                        self.display.clear()
                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{w}x{h} {wait_msg}"
                        )

                if not self._timing_printed:
                    self._timing_printed = True
                    cache_bits = []
                    if not classify_ran and self.fast_pipeline is not None:
                        cache_bits.append("cards=cache")
                    if not self._last_equity_ran:
                        cache_bits.append("equity=cache")
                    extra = f" ({', '.join(cache_bits)})" if cache_bits else ""
                    # Once at startup — console clears each tick so this won't scroll forever
                    print(
                        f"[timing] capture={ms_capture:.0f}ms "
                        f"cards={ms_cards:.0f}ms "
                        f"equity={ms_equity:.0f}ms{extra}"
                    )

                # Sleep only the remainder so interval is wall-clock, not stacked on work
                elapsed = time.perf_counter() - loop_t0
                sleep_for = max(0.0, self.capture_interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()

    def stop(self):
        self.is_running = False
        self.display.close()
        print("Stopped.")


def main():
    import argparse

    from src.detection.pipeline import classifier_available

    parser = argparse.ArgumentParser(description="Poker Vision - Stake.us Assistant")
    parser.add_argument("--console", action="store_true", help="Console output only")
    parser.add_argument(
        "--debug", action="store_true", help="Show debug window with bounding boxes"
    )
    parser.add_argument("--mock", action="store_true", help="Use mock capture for testing")
    parser.add_argument("--fullscreen", action="store_true", help="Capture full screen")
    parser.add_argument(
        "--select", action="store_true", help="Interactively select window to capture"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Capture interval in seconds (default: 1.0)",
    )
    parser.add_argument("--sims", type=int, default=500, help="Monte Carlo simulations")
    parser.add_argument(
        "--villain-range",
        choices=("random", "strong", "value"),
        default="strong",
        help=(
            "Opponent hole sampling for equity: random=uniform deck; "
            "strong=weighted value/draws (default); value=only hands that "
            "beat/tie hero on known board (≥3 cards)"
        ),
    )
    parser.add_argument(
        "--fast-cards",
        action="store_true",
        help="Blob + CNN classifier path (skip COCO YOLO)",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Force YOLO/mock detector even if classifier.pt exists",
    )
    parser.add_argument(
        "--debug-holes",
        action="store_true",
        help="Print per-tick hero hole recognition diagnostics",
    )
    parser.add_argument(
        "--debug-holes-log",
        type=Path,
        default=None,
        help="Optional path to append hero hole diag lines (with --debug-holes)",
    )

    args = parser.parse_args()

    # Sensible default: use CNN when classifier.pt is present; --fast-cards forces it;
    # --yolo keeps the existing COCO YOLO / mock detector path.
    use_fast = bool(args.fast_cards) or (classifier_available() and not args.yolo)

    # Select capture source
    capture = None
    if args.select:
        capture = select_window_interactive()
        if capture is None:
            return
    elif args.fullscreen:
        capture = ScreenCapture()
    elif args.mock:
        capture = ScreenCapture(mock_mode=True)
    else:
        # Default: interactive selection
        capture = select_window_interactive()
        if capture is None:
            print("Using default capture (mock mode)")
            capture = ScreenCapture(mock_mode=True)

    # Create and run app
    app = PokerVisionApp(
        capture=capture,
        use_overlay=not args.console and not args.debug,
        use_debug=args.debug,
        capture_interval=args.interval,
        fast_cards=use_fast,
        villain_range=args.villain_range,
        debug_holes=bool(args.debug_holes) or _env_flag("POKER_VISION_DEBUG_HOLES"),
        debug_holes_log=args.debug_holes_log,
    )

    app.engine.n_simulations = args.sims
    app.run()


if __name__ == "__main__":
    main()
