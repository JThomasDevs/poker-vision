"""
Screen capture module for poker vision.
Window capture like OBS - select a specific window to capture.
"""

import mss
import numpy as np
from typing import Tuple, List, Optional
import subprocess
import sys

# Set process DPI awareness so GetWindowRect and screen capture use the same coordinates
def _set_dpi_aware():
    try:
        from ctypes import windll
        # Per-Monitor V2 gives correct coords on multi-monitor and high-DPI (Win10 1703+)
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if hasattr(windll.user32, "SetProcessDpiAwarenessContext"):
            windll.user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        else:
            windll.user32.SetProcessDPIAware()
    except Exception:
        pass

_set_dpi_aware()


class WindowCaptureError(Exception):
    """Raised when window capture fails (no fallbacks)."""
    pass


class ScreenCapture:
    """Captures screen regions for poker table analysis."""
    
    def __init__(self, monitor_index: int = 1, mock_mode: bool = False):
        self.monitor_index = monitor_index
        self.mock_mode = mock_mode
        self.window_hwnd = None
        self.window_rect = None
        self.region = None
        
        if mock_mode:
            self.sct = None
            print("ScreenCapture: Running in MOCK mode")
        else:
            self.sct = mss.mss()
    
    def list_windows(self) -> List[dict]:
        """List all visible windows."""
        try:
            import win32gui
            import win32con
            
            windows = []
            
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and len(title) > 1:
                        # Get window rect
                        try:
                            rect = win32gui.GetWindowRect(hwnd)
                            if rect[2] - rect[0] > 100 and rect[3] - rect[1] > 100:  # Filter tiny windows
                                windows.append({
                                    'hwnd': hwnd,
                                    'title': title,
                                    'rect': rect
                                })
                        except:
                            pass
                return True
            
            win32gui.EnumWindows(callback, None)
            
            # Filter for browser windows
            browsers = [w for w in windows if any(b in w['title'].lower() for b in ['chrome', 'firefox', 'edge', 'browser', 'stake'])]
            
            return browsers if browsers else windows[:30]
            
        except ImportError:
            print("win32gui not available - install pywin32")
            return []
    
    def select_window(self, hwnd: int):
        """Select window to capture by handle."""
        self.window_hwnd = hwnd
        # Get and store window rect for capture
        try:
            import win32gui
            self.window_rect = win32gui.GetWindowRect(hwnd)
        except:
            self.window_rect = None
    
    def capture(self) -> np.ndarray:
        """Capture the selected window or region."""
        if self.mock_mode:
            return self._generate_mock_frame(1200, 800)
        
        if self.window_hwnd:
            return self._capture_window(self.window_hwnd)
        
        if self.region:
            x, y, w, h = self.region
            return self._capture_region(x, y, w, h)
        
        return self._capture_monitor()
    
    def _capture_window(self, hwnd: int) -> np.ndarray:
        """Capture a specific window. Prefer mss of the on-screen rect (what you see);
        fall back to PrintWindow if the window is off-screen / minimized.
        """
        import win32gui
        
        try:
            title = win32gui.GetWindowText(hwnd) or "(no title)"
        except Exception:
            title = "(unknown)"
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        if width <= 0 or height <= 0:
            raise WindowCaptureError(
                f"Window capture failed: window has invalid dimensions (width={width}, height={height}). "
                f"The window may be minimized, closed, or not visible. "
                f"Window title: {title!r}"
            )

        # Screen grab matches GPU-composited Chrome content better than PrintWindow.
        if self.sct is not None and top < 10000 and left < 10000:
            try:
                grab = self._capture_region(left, top, width, height)
                if grab is not None and grab.size > 0 and float(grab.mean()) > 1.0:
                    return grab
            except Exception:
                pass
        
        img, error_detail = self._capture_window_printwindow(hwnd, width, height)
        if img is None:
            raise WindowCaptureError(
                f"Window capture failed: PrintWindow returned no image. {error_detail} "
                f"Window: {title!r} (hwnd={hwnd}, size={width}x{height}). "
                "Ensure the window is visible and not minimized. "
                "For browsers, try disabling hardware acceleration in settings."
            )
        return img
    
    def _capture_window_printwindow(self, hwnd: int, width: int, height: int) -> Tuple[Optional[np.ndarray], str]:
        """Capture window via PrintWindow (PW_RENDERFULLCONTENT). Returns (image, error_detail)."""
        try:
            import win32gui
            import win32ui
            from ctypes import windll
            PW_RENDERFULLCONTENT = 2
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            ok = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            if not ok:
                return None, "PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT) returned 0 (failure)."
            bmpstr = bitmap.GetBitmapBits(True)
            arr = np.frombuffer(bmpstr, dtype=np.uint8).reshape((height, width, 4))[:, :, :3].copy()
            # PrintWindow returns top-down on this setup; no vertical flip needed (flipud was inverting it)
            return arr, ""
        except Exception as e:
            return None, f"Exception during PrintWindow: {type(e).__name__}: {e}"
    
    def _capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Capture a screen region."""
        region = {"top": y, "left": x, "width": width, "height": height}
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)[:, :, :3]
        return img
    
    def _capture_monitor(self) -> np.ndarray:
        """Capture entire monitor."""
        monitor = self.sct.monitors[self.monitor_index]
        screenshot = self.sct.grab(monitor)
        return np.array(screenshot)[:, :, :3]
    
    def _generate_mock_frame(self, width: int, height: int) -> np.ndarray:
        """Mock frame for testing."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (34, 139, 34)  # Green felt
        
        # Community cards
        for i in range(5):
            x = width // 2 - 150 + i * 65
            y = height // 4
            img[y:y+84, x:x+60] = (240, 240, 230)
        
        # Hole cards
        for i in range(2):
            x = width // 2 - 70 + i * 140
            y = height - 150
            img[y:y+84, x:x+60] = (240, 240, 230)
        
        return img
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get the size of captured frames."""
        if self.window_hwnd:
            try:
                import win32gui
                left, top, right, bottom = win32gui.GetWindowRect(self.window_hwnd)
                return (right - left, bottom - top)
            except:
                pass
        return (1200, 800)
    
    def __del__(self):
        if self.sct:
            try:
                self.sct.close()
            except:
                pass


def select_window_interactive() -> ScreenCapture:
    """Interactive window selection like OBS."""
    cap = ScreenCapture()
    windows = cap.list_windows()
    
    print("\n" + "="*60)
    print("Select window to capture (like OBS):")
    print("="*60)
    
    # Group by title
    seen = set()
    browser_windows = []
    other_windows = []
    
    for w in windows:
        if w['title'] not in seen:
            seen.add(w['title'])
            if any(b in w['title'].lower() for b in ['chrome', 'firefox', 'edge', 'stake', 'browser']):
                browser_windows.append(w)
            else:
                other_windows.append(w)
    
    # Show browsers first — demote Gradio / localhost identify tools.
    if browser_windows:
        def _browser_rank(w):
            t = w["title"].lower()
            demote = ("card identify" in t) or ("127.0.0.1" in t) or ("localhost" in t)
            prefer = ("stake.us" in t) or ("stake.com" in t) or ("poker" in t)
            return (0 if demote else 1, 1 if prefer else 0, w["title"])

        browser_windows = sorted(browser_windows, key=_browser_rank, reverse=True)
        print("\n[BROWSERS]  (pick the Stake.us poker tab, not 'card identify')")
        for i, w in enumerate(browser_windows):
            print(f"  [{i+1}] {w['title'][:55]}")
    
    print("\n[OTHER WINDOWS]")
    offset = len(browser_windows) + 1
    for i, w in enumerate(other_windows[:20]):
        print(f"  [{i+offset}] {w['title'][:55]}")
    
    print(f"\n  [F] Full screen")
    print("  [M] Mock test")
    print("  [Q] Quit")
    
    choice = input("\nSelect: ").strip().upper()
    
    if choice == 'Q':
        return None
    elif choice == 'F':
        print("Full screen capture selected")
        return cap
    elif choice == 'M':
        cap.mock_mode = True
        print("Mock mode selected")
        return cap
    else:
        try:
            idx = int(choice) - 1
            all_windows = browser_windows + other_windows[:20]
            if 0 <= idx < len(all_windows):
                w = all_windows[idx]
                cap.select_window(w['hwnd'])
                print(f"Selected: {w['title']}")
                return cap
        except:
            pass
    
    return None


if __name__ == "__main__":
    cap = select_window_interactive()
    if cap:
        print(f"\nCapturing... (press Ctrl+C to stop)")
        w, h = cap.get_frame_size()
        print(f"Frame size: {w}x{h}")
        
        import time
        for i in range(5):
            frame = cap.capture()
            print(f"Captured frame {i+1}: {frame.shape}")
            time.sleep(0.5)
