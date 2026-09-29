"""
Overlay display for poker recommendations.
Uses a simple transparent window overlay.
"""

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DisplayConfig:
    """Overlay display configuration."""
    width: int = 300
    height: int = 200
    x: int = 100
    y: int = 100
    bg_color: str = "#1a1a2e"
    text_color: str = "#00ff88"
    font_size: int = 16


class OverlayDisplay:
    """Transparent overlay window for poker recommendations."""
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()
        self.root = None
        self._setup_window()
    
    def _setup_window(self):
        """Create transparent overlay window."""
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # No window decorations
        self.root.attributes("-topmost", True)  # Always on top
        self.root.attributes("-alpha", 0.9)  # Slightly transparent
        
        # Position window
        self.root.geometry(f"{self.config.width}x{self.config.height}+{self.config.x}+{self.config.y}")
        
        # Make it transparent (click-through for non-active areas)
        # Note: This makes the whole window click-through
        # Use a transparent color for background instead
        self.root.configure(bg=self.config.bg_color)
        
        # Create label for displaying info
        self.info_label = tk.Label(
            self.root,
            text="Poker Vision\nWaiting for cards...",
            fg=self.config.text_color,
            bg=self.config.bg_color,
            font=("Arial", self.config.font_size),
            justify=tk.LEFT,
            padx=20,
            pady=20
        )
        self.info_label.pack(fill=tk.BOTH, expand=True)
        
        # Make window draggable
        self._make_draggable()
    
    def _make_draggable(self):
        """Allow dragging the overlay window."""
        def start_move(event):
            self._dx = event.x
            self._dy = event.y
        
        def do_move(event):
            deltax = event.x - self._dx
            deltay = event.y - self._dy
            x = self.root.winfo_x() + deltax
            y = self.root.winfo_y() + deltay
            self.root.geometry(f"+{x}+{y}")
        
        self.info_label.bind("<Button-1>", start_move)
        self.info_label.bind("<B1-Motion>", do_move)
    
    def update(
        self,
        hand_type: str,
        win_prob: float,
        recommendation: str,
        hole_cards: list = None,
        community_cards: list = None,
        amounts=None,
    ):
        """Update the overlay with new information."""
        
        # Format cards
        hole_str = " ".join(hole_cards) if hole_cards else "-- --"
        comm_str = " ".join(community_cards) if community_cards else ""
        amt_line = ""
        if amounts is not None and hasattr(amounts, "format_status"):
            amt_line = f"\n{amounts.format_status()}"
        
        # Color based on recommendation
        colors = {
            "raise": "#ff4444",       # Red
            "bet": "#ff8844",         # Orange
            "check/call": "#44ff44",  # Green (stronger mid)
            "check/fold": "#88cc44",  # Yellow-green (weaker mid)
            "call": "#ffff44",        # Yellow
            "fold": "#888888",        # Gray
        }
        color = colors.get(recommendation, self.config.text_color)
        
        text = f"""Hole: {hole_str}
Board: {comm_str}

{hand_type}
Win: {win_prob:.1%}{amt_line}

>>> {recommendation.upper()} <<<"""
        
        self.info_label.configure(text=text, fg=color)
    
    def clear(self):
        """Clear the display."""
        self.info_label.configure(
            text="Poker Vision\nWaiting for cards...",
            fg=self.config.text_color
        )
    
    def run(self):
        """Start the overlay event loop."""
        self.root.mainloop()
    
    def close(self):
        """Close the overlay."""
        if self.root:
            self.root.destroy()


class SimpleConsoleDisplay:
    """Fixed console panel: clear + redraw each tick so decisions stay on screen."""

    def __init__(self):
        self._last_sig = None
        self._status = ""

    @staticmethod
    def _clear_screen() -> None:
        """Reliable clear for Windows Terminal / PowerShell / ANSI hosts."""
        import os
        import sys

        if os.name == "nt":
            # cls works even when ANSI is off; try ANSI home+erase first for less flicker
            try:
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()
            except Exception:
                pass
            os.system("cls")
        else:
            os.system("clear")
            try:
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()
            except Exception:
                pass

    def set_status(self, message: str = "") -> None:
        """Optional one-line note under the panel (skipped frames, waiting, etc.)."""
        self._status = message or ""

    def update(
        self,
        hand_type: str,
        win_prob: float,
        recommendation: str,
        hole_cards: list = None,
        community_cards: list = None,
        amounts=None,
    ):
        """Redraw the full decision panel (screen clear first)."""
        hole_str = " ".join(hole_cards) if hole_cards else "N/A"
        board_str = " ".join(community_cards) if community_cards else "N/A"
        amt_str = ""
        if amounts is not None and hasattr(amounts, "format_status"):
            amt_str = amounts.format_status()

        sig = (hole_str, board_str, hand_type, round(win_prob, 4), recommendation, amt_str)
        changed = sig != self._last_sig
        self._last_sig = sig

        self._clear_screen()
        print("=" * 40)
        print("POKER VISION")
        print("=" * 40)
        print(f"Hole Cards: {hole_str}")
        print(f"Community:  {board_str}")
        print(f"Hand:       {hand_type}")
        print(f"Win %:      {win_prob:.1%}")
        if amt_str:
            print(f"Amounts:    {amt_str}")
        print(f"Action:     >>> {recommendation.upper()} <<<")
        print("=" * 40)
        if self._status:
            print(self._status)
            self._status = ""
        elif changed:
            from time import strftime
            print(f"[{strftime('%H:%M:%S')}] updated")

    def clear(
        self,
        hole_cards: list = None,
        community_cards: list = None,
        amounts=None,
        message: str = "Waiting for cards...",
    ):
        """Redraw a waiting panel so the view does not scroll or stick."""
        hole_str = " ".join(hole_cards) if hole_cards else "-- --"
        board_str = " ".join(community_cards) if community_cards else ""
        amt_str = ""
        if amounts is not None and hasattr(amounts, "format_status"):
            amt_str = amounts.format_status()

        self._clear_screen()
        print("=" * 40)
        print("POKER VISION")
        print("=" * 40)
        print(f"Hole Cards: {hole_str}")
        print(f"Community:  {board_str or 'N/A'}")
        print(f"Hand:       -")
        print(f"Win %:      -")
        if amt_str:
            print(f"Amounts:    {amt_str}")
        print(f"Action:     -")
        print("=" * 40)
        print(message)
        if self._status:
            print(self._status)
            self._status = ""
        self._last_sig = None

    def run(self):
        pass

    def close(self):
        pass


class DebugDisplay:
    """OpenCV window showing capture with bounding boxes."""
    
    def __init__(self, window_name: str = "Poker Vision Debug"):
        self.window_name = window_name
        self.frame = None
        
        # Try to import cv2
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
            print("WARNING: OpenCV not available for debug display")
    
    def update(self, frame: np.ndarray, cards: list, hand_type: str, 
               win_prob: float, recommendation: str,
               hole_cards: list = None, community_cards: list = None,
               amounts=None):
        """Update debug view with frame and detections."""
        if self.cv2 is None:
            return
            
        # Make a copy to draw on (capture pipeline already provides BGR; OpenCV imshow expects BGR)
        display = frame.copy()
        if display.shape[2] == 3:
            display = np.ascontiguousarray(display)
        
        # Draw bounding boxes for each card
        for card in cards:
            x1, y1, x2, y2 = card.bbox
            
            # Determine color (green for hole cards, blue for community)
            is_holecard = y1 > frame.shape[0] // 2
            color = (0, 255, 0) if is_holecard else (255, 0, 0)  # Green or Blue
            
            # Draw rectangle
            self.cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{card.rank}{card.suit.upper()} {card.confidence:.2f}"
            self.cv2.putText(display, label, (x1, y1 - 10),
                           self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if amounts is not None:
            try:
                from src.detection.table_amounts import draw_amount_rois
                display = draw_amount_rois(display, amounts)
            except Exception:
                pass
        
        # Draw recommendation overlay
        colors = {
            "raise": (0, 0, 255),         # Red
            "bet": (0, 165, 255),         # Orange
            "check/call": (0, 255, 0),    # Green (stronger mid)
            "check/fold": (0, 200, 100),  # Yellow-green (weaker mid)
            "call": (0, 255, 255),        # Yellow
            "fold": (128, 128, 128),      # Gray
        }
        color = colors.get(recommendation, (255, 255, 255))
        
        # Draw info panel on the right
        h, w = display.shape[:2]
        panel_width = 250
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        y_offset = 30
        line_height = 30
        
        # Title
        self.cv2.putText(panel, "POKER VISION", (10, y_offset),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 50
        
        # Hand info
        self.cv2.putText(panel, f"Hole: {' '.join(hole_cards) if hole_cards else 'N/A'}", 
                        (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        self.cv2.putText(panel, f"Board: {' '.join(community_cards) if community_cards else 'N/A'}", 
                        (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height * 2
        
        # Hand type
        self.cv2.putText(panel, f"Hand: {hand_type}", 
                        (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height
        
        # Win %
        self.cv2.putText(panel, f"Win: {win_prob:.1%}", 
                        (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y_offset += line_height

        if amounts is not None and hasattr(amounts, "format_status"):
            self.cv2.putText(panel, amounts.format_status()[:28],
                            (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
            y_offset += line_height
        
        y_offset += line_height
        
        # Recommendation
        self.cv2.putText(panel, f">>> {recommendation.upper()} <<<", 
                        (10, y_offset), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Combine frame and panel
        display = np.hstack([display, panel])
        
        # Resize to fit screen
        display = self.cv2.resize(display, (1200, 700))
        
        # Show
        self.cv2.imshow(self.window_name, display)
        self.cv2.waitKey(1)
    
    def clear(self):
        if self.cv2:
            # Create blank frame
            blank = np.zeros((700, 1200, 3), dtype=np.uint8)
            self.cv2.imshow(self.window_name, blank)
            self.cv2.waitKey(1)
    
    def run(self):
        """Start the display (creates window)."""
        if self.cv2:
            self.cv2.namedWindow(self.window_name)
    
    def close(self):
        """Close the window."""
        if self.cv2:
            self.cv2.destroyAllWindows()
