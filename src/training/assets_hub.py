"""
Central hub: browse images in assets, select one, and run the crop-cards script on it.

Usage:
  python -m src.training.assets_hub
"""
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter not available")
    sys.exit(1)
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_assets_dir() -> Path:
    return get_project_root() / "assets"


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def list_assets_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    out = []
    for f in sorted(folder.iterdir(), key=lambda p: (p.stat().st_mtime, p.name), reverse=True):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(f)
    return out


def image_content_hash(path: Path) -> Optional[str]:
    """Perceptual hash: same image (different format/quality) yields same hash. Returns hex string or None on error."""
    if Image is None:
        try:
            raw = path.read_bytes()
            return hashlib.sha256(raw).hexdigest()
        except Exception:
            return None
    try:
        img = Image.open(path).convert("L")
        resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or getattr(Image, "LANCZOS", 1)
        img = img.resize((32, 32), resample)
        return hashlib.sha256(img.tobytes()).hexdigest()
    except Exception:
        return None


def find_duplicate_groups(folder: Path) -> List[List[Path]]:
    """Group assets by image_content_hash; return only groups with more than one file."""
    paths = list_assets_images(folder)
    by_hash: Dict[str, List[Path]] = {}
    for p in paths:
        h = image_content_hash(p)
        if h is not None:
            by_hash.setdefault(h, []).append(p)
    return [sorted(g, key=lambda p: p.stat().st_mtime) for g in by_hash.values() if len(g) > 1]


def run_crop_script(image_path: Path) -> None:
    root = get_project_root()
    cmd = [sys.executable, "-m", "src.training.crop_cards_from_image", str(image_path)]
    subprocess.run(cmd, cwd=str(root))


def run_clear_all_blue() -> None:
    root = get_project_root()
    cmd = [sys.executable, "-m", "src.training.clean_assets_blue_only"]
    subprocess.run(cmd, cwd=str(root))


# Max size for preview pane (keep aspect ratio)
PREVIEW_MAX = 380


class DuplicatesWindow:
    """Toplevel to list duplicate groups and delete duplicates (keep one per group)."""

    def __init__(self, parent: tk.Tk, groups: List[List[Path]], on_deleted: Optional[Callable[[], None]] = None):
        self.groups = groups
        self.on_deleted = on_deleted
        self.win = tk.Toplevel(parent)
        self.win.title("Duplicate images")
        self.win.geometry("560x400")
        self.win.minsize(400, 200)
        ttk.Label(self.win, text=f"Found {sum(len(g) - 1 for g in groups)} duplicate(s) in {len(groups)} group(s). Keep oldest in each group, delete the rest.", wraplength=520).pack(anchor=tk.W, padx=8, pady=(8, 4))
        scroll_frame = ttk.Frame(self.win)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        scrollbar = ttk.Scrollbar(scroll_frame)
        self.canvas = tk.Canvas(scroll_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.canvas.yview)
        self._inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self._inner, anchor=tk.NW)
        self._inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._rows: List[Tuple[ttk.Frame, List[Path], ttk.Button]] = []
        for i, group in enumerate(groups):
            row = ttk.Frame(self._inner)
            row.pack(fill=tk.X, pady=2)
            keep_path = group[0]
            dup_paths = group[1:]
            ttk.Label(row, text=f"Group {i + 1}: keep {keep_path.name} ({len(dup_paths)} duplicate(s))", font=("", 9)).pack(side=tk.LEFT, padx=(0, 8))
            btn = ttk.Button(row, text="Delete duplicates (keep oldest)", command=lambda g=group: self._delete_duplicates_in_group(g))
            btn.pack(side=tk.RIGHT)
            self._rows.append((row, group, btn))
        btn_all = ttk.Button(self._inner, text="Delete all duplicates (keep oldest in each group)", command=self._delete_all_duplicates)
        btn_all.pack(pady=8)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_inner_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _delete_duplicates_in_group(self, group: List[Path]) -> None:
        kept = group[0]
        deleted = 0
        for p in group[1:]:
            if p.exists():
                try:
                    p.unlink()
                    deleted += 1
                except OSError:
                    pass
        if deleted and self.on_deleted:
            self.on_deleted()
        self._refresh_groups()

    def _delete_all_duplicates(self) -> None:
        for group in self.groups:
            for p in group[1:]:
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        if self.on_deleted:
            self.on_deleted()
        self._refresh_groups()

    def _refresh_groups(self) -> None:
        self.groups = [[p for p in g if p.exists()] for g in self.groups]
        self.groups = [g for g in self.groups if len(g) > 1]
        for row, _group, btn in self._rows:
            row.destroy()
        self._rows.clear()
        if not self.groups:
            self.win.destroy()
            return
        for i, group in enumerate(self.groups):
            row = ttk.Frame(self._inner)
            row.pack(fill=tk.X, pady=2)
            dup_paths = group[1:]
            ttk.Label(row, text=f"Group {i + 1}: keep {group[0].name} ({len(dup_paths)} duplicate(s))", font=("", 9)).pack(side=tk.LEFT, padx=(0, 8))
            btn = ttk.Button(row, text="Delete duplicates (keep oldest)", command=lambda g=group: self._delete_duplicates_in_group(g))
            btn.pack(side=tk.RIGHT)
            self._rows.append((row, group, btn))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_close(self) -> None:
        self.win.destroy()


def make_thumbnail(path: Path, max_size: int = PREVIEW_MAX):
    """Load image and resize to fit max_size; return PhotoImage or None."""
    if Image is None or ImageTk is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        resample = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or getattr(Image, "LANCZOS", 1)
        img.thumbnail((max_size, max_size), resample)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


class AssetsHub:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Poker Vision – Assets Hub")
        self.root.geometry("720x480")
        self.root.minsize(500, 360)
        self.assets_dir = get_assets_dir()
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._paths: List[Path] = []
        self._preview_photo: Optional[object] = None  # keep ref so image shows
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.BOTH, expand=True)
        ttk.Label(top, text="Images in assets/", font=("", 11, "bold")).pack(anchor=tk.W)
        paned = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        # Left: list
        list_frame = ttk.Frame(paned, width=220)
        list_frame.pack_propagate(False)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, font=("Consolas", 10))
        scroll = ttk.Scrollbar(list_frame)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scroll.set)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-1>", self._on_double_click)
        self.listbox.bind("<Return>", self._on_process)
        paned.add(list_frame, weight=0)
        # Right: preview
        preview_frame = ttk.Frame(paned)
        self.preview_label = tk.Label(preview_frame, text="Select an image", bg="#2b2b2b", fg="#aaa")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        paned.add(preview_frame, weight=1)
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Refresh", command=self._refresh).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Find duplicates", command=self._on_find_duplicates).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Clear all blue", command=self._on_clear_all_blue).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Process selected (crop cards)", command=self._on_process).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Delete selected", command=self._on_delete_selected).pack(side=tk.LEFT)
        self.root.bind("<Delete>", lambda e: self._on_delete_selected())
        self.root.bind("<F5>", lambda e: self._refresh())
        self.root.bind("<Key-r>", lambda e: self._refresh())
        self.root.bind("<Key-R>", lambda e: self._refresh())
        self.root.bind("<Up>", self._on_arrow_up)
        self.root.bind("<Down>", self._on_arrow_down)
        self.root.bind("<Return>", self._on_enter)
        ttk.Label(top, text="Arrow keys: navigate. Enter: process. Delete: delete. F5/R: refresh.", style="TLabel").pack(anchor=tk.W, pady=(4, 0))
        self._refresh()

    def _refresh(self):
        self.listbox.delete(0, tk.END)
        self._paths = list_assets_images(self.assets_dir)
        for p in self._paths:
            self.listbox.insert(tk.END, p.name)
        self._preview_photo = None
        if not self._paths:
            self.preview_label.config(image="", text="Select an image")
        else:
            self.listbox.selection_set(0)
            self.listbox.activate(0)
            self._on_select()

    def _on_select(self, event=None):
        path = self._get_selected_path()
        if path is None:
            return
        self._preview_photo = make_thumbnail(path)
        if self._preview_photo:
            self.preview_label.config(image=self._preview_photo, text="")
        else:
            self.preview_label.config(image="", text=path.name)

    def _move_selection(self, new_index: int):
        n = len(self._paths)
        if n == 0:
            return
        idx = max(0, min(new_index, n - 1))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.listbox.see(idx)
        self._on_select()

    def _on_arrow_up(self, event=None):
        sel = self.listbox.curselection()
        idx = int(sel[0]) if sel else 0
        if idx <= 0:
            self._move_selection(len(self._paths) - 1)
        else:
            self._move_selection(idx - 1)
        return "break"

    def _on_arrow_down(self, event=None):
        sel = self.listbox.curselection()
        idx = int(sel[0]) if sel else -1
        if idx >= len(self._paths) - 1:
            self._move_selection(0)
        else:
            self._move_selection(idx + 1)
        return "break"

    def _on_enter(self, event=None):
        self._on_process(event)
        return "break"

    def _get_selected_path(self) -> Optional[Path]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._paths):
            return None
        return self._paths[idx]

    def _on_double_click(self, event):
        self._on_process(event)

    def _on_process(self, event=None):
        path = self._get_selected_path()
        if path is None:
            messagebox.showinfo("No selection", "Select an image first.")
            return
        if not path.exists():
            messagebox.showerror("Error", f"File not found: {path}")
            self._refresh()
            return
        self.root.withdraw()
        try:
            run_crop_script(path)
            path.unlink(missing_ok=True)
            self._refresh()
        finally:
            self.root.deiconify()

    def _on_clear_all_blue(self):
        if not messagebox.askyesno("Clear all blue", "Delete all images that are only blue background (no cards) from assets?"):
            return
        self.root.withdraw()
        try:
            run_clear_all_blue()
            self._refresh()
        finally:
            self.root.deiconify()

    def _on_delete_selected(self):
        path = self._get_selected_path()
        if path is None:
            messagebox.showinfo("No selection", "Select an image first.")
            return
        if not path.exists():
            self._refresh()
            return
        if not messagebox.askyesno("Delete image", f"Delete from assets?\n{path.name}"):
            return
        path.unlink(missing_ok=True)
        self._refresh()

    def _on_find_duplicates(self):
        groups = find_duplicate_groups(self.assets_dir)
        if not groups:
            messagebox.showinfo("No duplicates", "No duplicate images found in assets.")
            return
        DuplicatesWindow(self.root, groups, on_deleted=self._refresh)

    def run(self):
        self.root.mainloop()


def main():
    hub = AssetsHub()
    hub.run()


if __name__ == "__main__":
    main()
