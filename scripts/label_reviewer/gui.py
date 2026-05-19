"""
gui.py — Tkinter GUI launcher for the Label Reviewer.

Simplified modes:
    1. Review         — Edit existing GT labels
    2. Review + Model — GT labels + model predictions overlay
    3. Auto-Label     — Model generates labels for unlabeled images
    4. Predict Only   — Run model, save labels (no review window)

Plus: Export to Dataset button.
"""
import hashlib
import json
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path


# Mode definitions

MODES = {
    "Review": {
        "description": "Review and edit existing ground truth labels",
        "fields": ["images_dir", "labels_dir", "detection_filter", "output_dir", "grouping", "pattern"],
        "required": ["images_dir", "labels_dir"],
    },
    "Review + Model": {
        "description": "Review GT labels (blue) with model predictions (purple) — promote, delete, or keep",
        "fields": ["images_dir", "labels_dir", "model_path", "confidence", "imgsz", "detection_filter", "output_dir", "grouping", "pattern"],
        "required": ["images_dir", "labels_dir", "model_path"],
    },
    "Compare Labels": {
        "description": "Compare two label directories side by side — Labels A (blue) vs Labels B (purple)",
        "fields": ["images_dir", "labels_dir", "labels_b_dir", "detection_filter", "grouping", "pattern"],
        "required": ["images_dir", "labels_dir", "labels_b_dir"],
    },
    "Auto-Label": {
        "description": "Run model on unlabeled images, then review and edit the predictions",
        "fields": ["images_dir", "model_path", "confidence", "imgsz", "grouping", "pattern"],
        "required": ["images_dir", "model_path"],
    },
    "Predict Only": {
        "description": "Run model inference and save labels + review images (no review window)",
        "fields": ["images_dir", "model_path", "confidence", "imgsz"],
        "required": ["images_dir", "model_path"],
    },
}

IMGSZ_OPTIONS = ["320", "416", "512", "640", "768", "896", "1024", "1280", "1536", "1920"]

GROUPING_OPTIONS = ["Auto-detect", "By filename pattern", "By parent folder", "None"]
GROUPING_MAP = {
    "Auto-detect": "auto",
    "By filename pattern": "pattern",
    "By parent folder": "folder",
    "None": "none",
}

DETECTION_FILTER_OPTIONS = ["All", "FP Only", "FN Only", "TP Only", "All Mismatches"]
REVIEW_FILTER_OPTIONS = ["All", "Negatives Only", "Positives Only"]

LAST_SESSION_FILE = Path("label_reviewer/.last_session.json")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class LabelReviewerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Label Reviewer")
        self.root.resizable(False, False)

        # theme colors
        self.bg = "#1e1e2e"
        self.fg = "#cdd6f4"
        self.accent = "#89b4fa"
        self.field_bg = "#313244"
        self.btn_bg = "#585b70"
        self.btn_active = "#89b4fa"
        self.border = "#45475a"
        self.export_color = "#a6e3a1"

        self.root.configure(bg=self.bg)

        # Load last session
        self.last_session = self._load_last_session()

        # Variables — fall back to "Review" if saved mode no longer exists
        saved_mode = self.last_session.get("mode", "Review")
        if saved_mode not in MODES:
            saved_mode = "Review"
        self.mode_var = tk.StringVar(value=saved_mode)
        self.images_dir_var = tk.StringVar(value=self.last_session.get("images_dir", ""))
        self.labels_dir_var = tk.StringVar(value=self.last_session.get("labels_dir", ""))
        self.labels_b_dir_var = tk.StringVar(value=self.last_session.get("labels_b_dir", ""))
        self.model_path_var = tk.StringVar(value=self.last_session.get("model_path", ""))
        self.output_dir_var = tk.StringVar(value=self.last_session.get("output_dir", "label_reviewer/"))
        self.grouping_var = tk.StringVar(value=self.last_session.get("grouping", "Auto-detect"))
        self.pattern_var = tk.StringVar(value=self.last_session.get("pattern", ""))
        self.confidence_var = tk.StringVar(value=self.last_session.get("confidence", "0.25"))
        self.imgsz_var = tk.StringVar(value=self.last_session.get("imgsz", "640"))
        self.detection_filter_var = tk.StringVar(value=self.last_session.get("detection_filter", "All"))

        self._build_ui()
        self._on_mode_change()

    def _load_last_session(self) -> dict:
        if LAST_SESSION_FILE.exists():
            try:
                with open(LAST_SESSION_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_last_session(self):
        LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode_var.get(),
            "images_dir": self.images_dir_var.get(),
            "labels_dir": self.labels_dir_var.get(),
            "labels_b_dir": self.labels_b_dir_var.get(),
            "model_path": self.model_path_var.get(),
            "output_dir": self.output_dir_var.get(),
            "grouping": self.grouping_var.get(),
            "pattern": self.pattern_var.get(),
            "confidence": self.confidence_var.get(),
            "imgsz": self.imgsz_var.get(),
            "detection_filter": self.detection_filter_var.get(),
        }
        with open(LAST_SESSION_FILE, "w") as f:
            json.dump(data, f, indent=2)

    # UI Building

    def _build_ui(self):
        pad = {"padx": 12, "pady": 4}
        row = 0

        # Title
        title = tk.Label(self.root, text="Label Reviewer",
                         font=("Segoe UI", 16, "bold"),
                         bg=self.bg, fg=self.accent)
        title.grid(row=row, column=0, columnspan=3, **pad, sticky="w")
        row += 1

        sep = tk.Frame(self.root, height=2, bg=self.border)
        sep.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
        row += 1

        # Mode dropdown
        self._add_label(row, "Mode:")
        mode_combo = ttk.Combobox(self.root, textvariable=self.mode_var,
                                  values=list(MODES.keys()), state="readonly",
                                  width=25)
        mode_combo.grid(row=row, column=1, columnspan=2, **pad, sticky="ew")
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
        row += 1

        # Description
        self.desc_label = tk.Label(self.root, text="", font=("Segoe UI", 9, "italic"),
                                   bg=self.bg, fg="#a6adc8", wraplength=450)
        self.desc_label.grid(row=row, column=0, columnspan=3, **pad, sticky="w")
        row += 1

        # Path fields
        self.field_rows = {}
        self.field_rows["images_dir"] = self._add_path_row(row, "Images Dir:", self.images_dir_var, is_dir=True)
        row += 1
        self.field_rows["labels_dir"] = self._add_path_row(row, "Labels Dir:", self.labels_dir_var, is_dir=True)
        row += 1
        self.field_rows["labels_b_dir"] = self._add_path_row(row, "Labels B Dir:", self.labels_b_dir_var, is_dir=True)
        row += 1
        self.field_rows["model_path"] = self._add_path_row(row, "Model Path:", self.model_path_var, is_dir=False)
        row += 1
        self.field_rows["output_dir"] = self._add_path_row(row, "Output Dir:", self.output_dir_var, is_dir=True)
        row += 1

        # Grouping dropdown
        grp_label = tk.Label(self.root, text="Grouping:", font=("Segoe UI", 10),
                             bg=self.bg, fg=self.fg, anchor="w")
        grp_combo = ttk.Combobox(self.root, textvariable=self.grouping_var,
                                 values=GROUPING_OPTIONS, state="readonly", width=25)
        grp_combo.bind("<<ComboboxSelected>>", lambda e: self._on_grouping_change())
        self.field_rows["grouping"] = (grp_label, grp_combo, None)
        grp_label.grid(row=row, column=0, **pad, sticky="w")
        grp_combo.grid(row=row, column=1, columnspan=2, **pad, sticky="ew")
        row += 1

        # Pattern field
        pat_label = tk.Label(self.root, text="Pattern:", font=("Segoe UI", 10),
                             bg=self.bg, fg=self.fg, anchor="w")
        pat_entry = tk.Entry(self.root, textvariable=self.pattern_var,
                             bg=self.field_bg, fg=self.fg, insertbackground=self.fg,
                             relief="flat", font=("Consolas", 10))
        self.field_rows["pattern"] = (pat_label, pat_entry, None)
        pat_label.grid(row=row, column=0, **pad, sticky="w")
        pat_entry.grid(row=row, column=1, columnspan=2, **pad, sticky="ew")
        row += 1

        # Confidence
        conf_label = tk.Label(self.root, text="Confidence:", font=("Segoe UI", 10),
                              bg=self.bg, fg=self.fg, anchor="w")
        conf_entry = tk.Entry(self.root, textvariable=self.confidence_var,
                              bg=self.field_bg, fg=self.fg, insertbackground=self.fg,
                              relief="flat", font=("Consolas", 10), width=8)
        self.field_rows["confidence"] = (conf_label, conf_entry, None)
        conf_label.grid(row=row, column=0, **pad, sticky="w")
        conf_entry.grid(row=row, column=1, **pad, sticky="w")
        row += 1

        # Image size (YOLO imgsz)
        imgsz_label = tk.Label(self.root, text="Image size:", font=("Segoe UI", 10),
                               bg=self.bg, fg=self.fg, anchor="w")
        imgsz_combo = ttk.Combobox(self.root, textvariable=self.imgsz_var,
                                   values=IMGSZ_OPTIONS, width=8)
        self.field_rows["imgsz"] = (imgsz_label, imgsz_combo, None)
        imgsz_label.grid(row=row, column=0, **pad, sticky="w")
        imgsz_combo.grid(row=row, column=1, **pad, sticky="w")
        row += 1

        # Detection filter dropdown (Review + Model only)
        df_label = tk.Label(self.root, text="Filter:", font=("Segoe UI", 10),
                            bg=self.bg, fg=self.fg, anchor="w")
        df_combo = ttk.Combobox(self.root, textvariable=self.detection_filter_var,
                                values=DETECTION_FILTER_OPTIONS, state="readonly", width=25)
        self.field_rows["detection_filter"] = (df_label, df_combo, None)
        df_label.grid(row=row, column=0, **pad, sticky="w")
        df_combo.grid(row=row, column=1, columnspan=2, **pad, sticky="ew")
        row += 1

        # Separator
        sep2 = tk.Frame(self.root, height=2, bg=self.border)
        sep2.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=8)
        row += 1

        # Launch button
        self.launch_btn = tk.Button(self.root, text="Launch",
                                    font=("Segoe UI", 12, "bold"),
                                    bg=self.accent, fg="#1e1e2e",
                                    activebackground="#b4d0fb",
                                    relief="flat", cursor="hand2",
                                    command=self._on_launch)
        self.launch_btn.grid(row=row, column=0, columnspan=3, padx=12, pady=(10, 4), sticky="ew")
        row += 1

        # Export to Dataset button
        self.export_btn = tk.Button(self.root, text="Export to Dataset",
                                    font=("Segoe UI", 10),
                                    bg=self.export_color, fg="#1e1e2e",
                                    activebackground="#c6f0c2",
                                    relief="flat", cursor="hand2",
                                    command=self._on_export)
        self.export_btn.grid(row=row, column=0, columnspan=3, padx=12, pady=(4, 10), sticky="ew")
        row += 1

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              font=("Segoe UI", 9), bg="#181825", fg="#a6adc8",
                              anchor="w")
        status_bar.grid(row=row, column=0, columnspan=3, sticky="ew")

    def _add_label(self, row, text):
        label = tk.Label(self.root, text=text, font=("Segoe UI", 10),
                         bg=self.bg, fg=self.fg, anchor="w")
        label.grid(row=row, column=0, padx=12, pady=4, sticky="w")
        return label

    def _add_path_row(self, row, label_text, var, is_dir=True):
        pad = {"padx": 12, "pady": 4}
        label = tk.Label(self.root, text=label_text, font=("Segoe UI", 10),
                         bg=self.bg, fg=self.fg, anchor="w")
        entry = tk.Entry(self.root, textvariable=var,
                         bg=self.field_bg, fg=self.fg,
                         insertbackground=self.fg,
                         relief="flat", font=("Consolas", 10))

        def browse():
            if is_dir:
                path = filedialog.askdirectory(title=label_text)
            else:
                path = filedialog.askopenfilename(
                    title=label_text,
                    filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
            if path:
                var.set(path)

        btn = tk.Button(self.root, text="Browse", command=browse,
                        bg=self.btn_bg, fg=self.fg, relief="flat",
                        font=("Segoe UI", 9), cursor="hand2")

        label.grid(row=row, column=0, **pad, sticky="w")
        entry.grid(row=row, column=1, **pad, sticky="ew")
        btn.grid(row=row, column=2, padx=(0, 12), pady=4)

        self.root.columnconfigure(1, weight=1, minsize=300)

        return (label, entry, btn)

    # Mode logic

    def _on_mode_change(self):
        mode = self.mode_var.get()
        cfg = MODES[mode]
        self.desc_label.config(text=cfg["description"])

        for field_name, widgets in self.field_rows.items():
            visible = field_name in cfg["fields"]
            for w in widgets:
                if w:
                    if visible:
                        w.grid()
                    else:
                        w.grid_remove()

        # Update filter dropdown options based on mode
        if "detection_filter" in cfg["fields"]:
            df_widgets = self.field_rows["detection_filter"]
            combo = df_widgets[1]  # the ttk.Combobox
            if mode == "Review":
                combo["values"] = REVIEW_FILTER_OPTIONS
                if self.detection_filter_var.get() not in REVIEW_FILTER_OPTIONS:
                    self.detection_filter_var.set("All")
            else:
                combo["values"] = DETECTION_FILTER_OPTIONS
                if self.detection_filter_var.get() not in DETECTION_FILTER_OPTIONS:
                    self.detection_filter_var.set("All")

        self._on_grouping_change()

    def _on_grouping_change(self):
        mode = self.mode_var.get()
        cfg = MODES[mode]
        show_pattern = (self.grouping_var.get() == "By filename pattern"
                        and "pattern" in cfg["fields"])
        for w in self.field_rows.get("pattern", ()):
            if w:
                if show_pattern:
                    w.grid()
                else:
                    w.grid_remove()

    def _validate(self) -> bool:
        mode = self.mode_var.get()
        cfg = MODES[mode]

        for field in cfg.get("required", []):
            var_name = field + "_var"
            var = getattr(self, var_name, None)
            if not var or not var.get().strip():
                messagebox.showerror("Missing Field",
                                     f"{field.replace('_', ' ').title()} is required for {mode} mode.")
                return False

        # Validate paths exist
        images_dir = self.images_dir_var.get().strip()
        if images_dir and not Path(images_dir).exists():
            messagebox.showerror("Invalid Path", f"Images directory not found:\n{images_dir}")
            return False

        labels_dir = self.labels_dir_var.get().strip()
        if labels_dir and "labels_dir" in cfg.get("required", []) and not Path(labels_dir).exists():
            messagebox.showerror("Invalid Path", f"Labels directory not found:\n{labels_dir}")
            return False

        model_path = self.model_path_var.get().strip()
        if model_path and "model_path" in cfg.get("required", []) and not Path(model_path).exists():
            messagebox.showerror("Invalid Path", f"Model weights not found:\n{model_path}")
            return False

        # Validate confidence
        if "confidence" in cfg["fields"]:
            try:
                conf = float(self.confidence_var.get())
                if not 0 < conf < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Confidence", "Confidence must be between 0 and 1.")
                return False

        # Validate imgsz
        if "imgsz" in cfg["fields"]:
            try:
                imgsz = int(self.imgsz_var.get())
                if imgsz < 64 or imgsz > 4096 or imgsz % 32 != 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Image Size",
                                     "Image size must be an integer multiple of 32 "
                                     "between 64 and 4096 (e.g. 640, 1280).")
                return False

        return True

    # Helpers

    def _make_output_dir(self) -> Path:
        """Find existing session for this dataset, or create a new one.
        
        Checks .copied_* markers to see if a previous session already has
        labels copied from the same source. Reuses the most recent match
        to avoid redundant bulk copies.
        """
        lr_dir = Path("label_reviewer")
        labels_dir = self.labels_dir_var.get().strip()

        # Try to find an existing session for this dataset
        if labels_dir and lr_dir.exists():
            labels_path = Path(labels_dir).resolve()
            candidates = []
            for d in lr_dir.iterdir():
                if not d.is_dir() or d.name == ".cache":
                    continue
                for marker in d.glob(".copied_*"):
                    try:
                        content = marker.read_text()
                        if str(labels_path) in content:
                            candidates.append(d)
                            break
                    except Exception:
                        pass
            if candidates:
                # Pick the most recently modified session
                best = max(candidates, key=lambda d: d.stat().st_mtime)
                print(f"  Reusing existing session: {best.name}")
                return best

        # No existing session found — create new one
        mode = self.mode_var.get().lower().replace(" ", "_").replace("+", "")
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        images_name = Path(self.images_dir_var.get()).name
        out = lr_dir / f"{mode}_{images_name}_{date_str}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _get_grouping_args(self) -> dict:
        grouping = GROUPING_MAP.get(self.grouping_var.get(), "auto")
        pattern = self.pattern_var.get().strip() or None
        return {
            "grouping_mode": grouping,
            "grouping_pattern": pattern,
            "group_by_video": False,
        }

    def _get_prediction_cache_key(self, model_path: str, images_dir: str,
                                   conf: float, imgsz: int) -> str:
        """Generate a cache key from model path, images dir, confidence, and imgsz."""
        key_str = f"{Path(model_path).resolve()}|{Path(images_dir).resolve()}|{conf}|{imgsz}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _run_or_cache_prediction(self, model_path: str, images_dir: str,
                                  conf: float, imgsz: int = 640,
                                  save_review: bool = False) -> Path:
        """Run prediction or reuse cached results. Supports resuming interrupted runs."""
        from .predictor import run_prediction

        cache_key = self._get_prediction_cache_key(model_path, images_dir, conf, imgsz)
        cache_dir = Path("label_reviewer") / ".cache" / cache_key
        manifest = cache_dir / "labeling_manifest.json"
        label_dir = cache_dir / "labels"

        if manifest.exists():
            try:
                with open(manifest) as f:
                    meta = json.load(f)
                cached_weights = meta.get("weights", "")
                cached_total = meta.get("total_images", 0)
                current_total = len([f for f in Path(images_dir).iterdir()
                                     if f.suffix.lower() in IMG_EXTS])
                if (Path(cached_weights).name == Path(model_path).name
                        and cached_total == current_total):
                    label_count = len(list(label_dir.glob("*.txt")))
                    self.status_var.set(f"Using cached predictions ({label_count} labels)")
                    self.root.update()
                    print(f"  [CACHE HIT] Reusing {cache_dir} ({label_count} labels)")
                    return cache_dir
            except Exception:
                pass

        # Check for partial/interrupted prediction (labels exist, no manifest)
        if label_dir.exists():
            existing_labels = len(list(label_dir.glob("*.txt")))
            if existing_labels > 0:
                current_total = len([f for f in Path(images_dir).iterdir()
                                     if f.suffix.lower() in IMG_EXTS])
                self.status_var.set(
                    f"Resuming predictions ({existing_labels}/{current_total} done)...")
                self.root.update()
                print(f"  [RESUME] Found {existing_labels} existing predictions, "
                      f"resuming from where we left off...")

        self.status_var.set("Running model predictions...")
        self.root.update()
        run_prediction(
            weights=model_path,
            source_dir=Path(images_dir),
            output_dir=cache_dir,
            conf_threshold=conf,
            image_size=imgsz,
            save_review=save_review,
        )
        return cache_dir

    # Launch handlers

    def _on_launch(self):
        if not self._validate():
            return

        self._save_last_session()
        mode = self.mode_var.get()
        self.status_var.set(f"Launching {mode}...")
        self.launch_btn.config(state="disabled")
        self.root.update()

        try:
            if mode == "Review":
                self._launch_review()
            elif mode == "Review + Model":
                self._launch_review_model()
            elif mode == "Compare Labels":
                self._launch_compare_labels()
            elif mode == "Auto-Label":
                self._launch_auto_label()
            elif mode == "Predict Only":
                self._launch_predict_only()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.launch_btn.config(state="normal")
            self.status_var.set("Ready")

    def _launch_review(self):
        """Review existing GT labels — no model involved."""
        from .core import LabelReviewer

        output_dir = Path(self.output_dir_var.get().strip()) if self.output_dir_var.get().strip() else self._make_output_dir()

        self.root.withdraw()
        reviewer = LabelReviewer(
            images_dir=Path(self.images_dir_var.get()),
            labels_dir=Path(self.labels_dir_var.get()),
            output_dir=output_dir,
            label_source="gt",
            detection_filter=self.detection_filter_var.get(),
            **self._get_grouping_args(),
        )
        reviewer.run()
        self.root.deiconify()

    def _launch_review_model(self):
        """Review GT labels with model predictions overlaid."""
        from .core import LabelReviewer

        output_dir = Path(self.output_dir_var.get().strip()) if self.output_dir_var.get().strip() else self._make_output_dir()
        model_path = self.model_path_var.get().strip()
        conf = float(self.confidence_var.get())
        imgsz = int(self.imgsz_var.get())

        # Run or reuse cached predictions
        pred_dir = self._run_or_cache_prediction(model_path,
                                                  self.images_dir_var.get(),
                                                  conf, imgsz)

        self.root.withdraw()
        reviewer = LabelReviewer(
            images_dir=Path(self.images_dir_var.get()),
            labels_dir=Path(self.labels_dir_var.get()),
            pred_labels_dir=pred_dir / "labels",
            output_dir=output_dir,
            label_source="gt",
            detection_filter=self.detection_filter_var.get(),
            **self._get_grouping_args(),
        )
        reviewer.run()
        self.root.deiconify()

    def _launch_compare_labels(self):
        """Compare two label directories side by side."""
        from .core import LabelReviewer

        output_dir = self._make_output_dir()

        self.root.withdraw()
        reviewer = LabelReviewer(
            images_dir=Path(self.images_dir_var.get()),
            labels_dir=Path(self.labels_dir_var.get()),
            pred_labels_dir=Path(self.labels_b_dir_var.get()),
            output_dir=output_dir,
            label_source="gt",
            detection_filter=self.detection_filter_var.get(),
            **self._get_grouping_args(),
        )
        reviewer.run()
        self.root.deiconify()

    def _launch_auto_label(self):
        """Run model on images, then review generated labels."""
        from .core import LabelReviewer

        conf = float(self.confidence_var.get())
        imgsz = int(self.imgsz_var.get())
        model_path = self.model_path_var.get().strip()

        # Run or reuse cached predictions
        pred_dir = self._run_or_cache_prediction(model_path,
                                                  self.images_dir_var.get(),
                                                  conf, imgsz,
                                                  save_review=True)

        self.root.withdraw()
        reviewer = LabelReviewer(
            images_dir=Path(self.images_dir_var.get()),
            labels_dir=pred_dir / "labels",
            manifest_path=pred_dir / "labeling_manifest.json",
            label_source="model",
            **self._get_grouping_args(),
        )
        reviewer.run()
        self.root.deiconify()

    def _launch_predict_only(self):
        """Run model, save labels + review images, no review window."""
        conf = float(self.confidence_var.get())
        imgsz = int(self.imgsz_var.get())
        model_path = self.model_path_var.get().strip()

        pred_dir = self._run_or_cache_prediction(model_path,
                                                  self.images_dir_var.get(),
                                                  conf, imgsz,
                                                  save_review=True)

        label_count = len(list((pred_dir / "labels").glob("*.txt")))
        messagebox.showinfo("Predict Complete",
                            f"Done!\n\n"
                            f"Labels: {label_count}\n"
                            f"Output: {pred_dir}")

    # Export to Dataset

    def _find_sessions_for_dataset(self, labels_dir: str) -> list:
        """Find all review sessions that were created from a specific labels dir.
        
        Checks .copied_* marker files to match sessions to their source dataset.
        Returns list of (session_path, marker_info) sorted newest first.
        """
        lr_dir = Path("label_reviewer")
        if not lr_dir.exists():
            return []

        labels_path = Path(labels_dir).resolve()
        matches = []

        for d in lr_dir.iterdir():
            if not d.is_dir() or d.name == ".cache":
                continue
            # Check all copy markers in this session
            for marker in d.glob(".copied_*"):
                try:
                    content = marker.read_text()
                    if str(labels_path) in content or labels_path.name in content:
                        matches.append(d)
                        break
                except Exception:
                    pass

        # Sort by modification time, newest first
        matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return matches

    def _collect_changed_stems(self, sessions: list) -> dict:
        """Collect all sessions that edited each file stem.
        
        Returns: {stem: [session_path, ...]} — all sessions that touched this stem,
                 ordered by changelog timestamp (newest last).
        """
        stem_sessions = {}  # stem -> set of session paths

        for sess in sessions:
            changelog_path = sess / "changelog.json"
            if changelog_path.exists():
                try:
                    with open(changelog_path) as f:
                        changes = json.load(f)
                    for entry in changes:
                        stem = entry.get("stem", "")
                        if stem:
                            if stem not in stem_sessions:
                                stem_sessions[stem] = set()
                            stem_sessions[stem].add(sess)
                except Exception:
                    pass

        return {stem: list(sessions) for stem, sessions in stem_sessions.items()}

    @staticmethod
    def _iou(box_a, box_b):
        """Compute IoU between two YOLO boxes (cls, xc, yc, w, h)."""
        ax1 = box_a[1] - box_a[3] / 2
        ay1 = box_a[2] - box_a[4] / 2
        ax2 = box_a[1] + box_a[3] / 2
        ay2 = box_a[2] + box_a[4] / 2
        bx1 = box_b[1] - box_b[3] / 2
        by1 = box_b[2] - box_b[4] / 2
        bx2 = box_b[1] + box_b[3] / 2
        by2 = box_b[2] + box_b[4] / 2
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_label_files(self, stem: str, session_paths: list,
                           original_dir: Path, iou_thresh: float = 0.5) -> list:
        """Merge bounding boxes for one image across all sessions.
        
        Strategy: start with the original labels, then union in boxes from
        each session. Boxes with IoU >= threshold against existing boxes are
        treated as duplicates (keep the newer one). Unique boxes are added.
        
        Returns: merged list of (cls, xc, yc, w, h) tuples.
        """
        def load_boxes(label_path):
            boxes = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            boxes.append((cls, xc, yc, w, h))
            return boxes

        # Start with the original file
        merged = load_boxes(original_dir / f"{stem}.txt")

        # Union in boxes from each session
        for sess in session_paths:
            sess_boxes = load_boxes(sess / f"{stem}.txt")
            for sbox in sess_boxes:
                # Check if this box already exists in merged (by IoU)
                is_dup = False
                best_iou = 0
                best_idx = -1
                for i, mbox in enumerate(merged):
                    iou = self._iou(sbox, mbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_iou >= iou_thresh:
                    # Replace with the session's version (it's a reviewed edit)
                    merged[best_idx] = sbox
                else:
                    # New box — add it
                    merged.append(sbox)

        return merged

    def _on_export(self):
        """Export reviewed labels back to the original dataset.
        
        Enhanced version:
        - Auto-detects ALL sessions that reviewed the same dataset
        - Merges changes across sessions (newest edit wins per file)
        - Creates backup before overwriting
        - Shows detailed summary
        """
        lr_dir = Path("label_reviewer")
        if not lr_dir.exists():
            messagebox.showerror("No Sessions",
                                 "No label reviewer sessions found.\n"
                                 "Run a review first to create reviewed labels.")
            return

        # List available sessions (exclude .cache)
        sessions = sorted([d for d in lr_dir.iterdir()
                           if d.is_dir() and d.name != ".cache"],
                          key=lambda d: d.stat().st_mtime,
                          reverse=True)

        if not sessions:
            messagebox.showerror("No Sessions", "No review sessions found.")
            return

        # Create export dialog
        export_win = tk.Toplevel(self.root)
        export_win.title("Export to Dataset")
        export_win.configure(bg=self.bg)
        export_win.resizable(False, False)
        export_win.transient(self.root)
        export_win.grab_set()

        tk.Label(export_win, text="Export Reviewed Labels",
                 font=("Segoe UI", 14, "bold"),
                 bg=self.bg, fg=self.accent).grid(row=0, column=0, columnspan=2,
                                                   padx=12, pady=(12, 4), sticky="w")

        # Mode toggle: single session or merge all
        export_mode_var = tk.StringVar(value="single")
        mode_frame = tk.Frame(export_win, bg=self.bg)
        mode_frame.grid(row=1, column=0, columnspan=2, padx=12, pady=4, sticky="w")

        tk.Radiobutton(mode_frame, text="Single session",
                       variable=export_mode_var, value="single",
                       bg=self.bg, fg=self.fg, selectcolor=self.field_bg,
                       activebackground=self.bg, activeforeground=self.fg,
                       command=lambda: on_mode_toggle()
                       ).pack(side="left", padx=(0, 15))
        tk.Radiobutton(mode_frame, text="Merge ALL sessions (for same dataset)",
                       variable=export_mode_var, value="merge_all",
                       bg=self.bg, fg="#a6e3a1", selectcolor=self.field_bg,
                       activebackground=self.bg, activeforeground="#a6e3a1",
                       command=lambda: on_mode_toggle()
                       ).pack(side="left")

        # Session dropdown (for single mode)
        session_label = tk.Label(export_win, text="Select session:",
                 font=("Segoe UI", 9),
                 bg=self.bg, fg="#a6adc8")
        session_label.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 4), sticky="w")

        session_var = tk.StringVar()
        session_names = [s.name for s in sessions]
        session_combo = ttk.Combobox(export_win, textvariable=session_var,
                                     values=session_names, state="readonly", width=45)
        if session_names:
            session_combo.current(0)
        session_combo.grid(row=3, column=0, columnspan=2, padx=12, pady=4, sticky="ew")

        # Info label
        info_var = tk.StringVar(value="")
        info_label = tk.Label(export_win, textvariable=info_var,
                              font=("Segoe UI", 9), bg=self.bg, fg="#a6adc8",
                              wraplength=500, justify="left")
        info_label.grid(row=4, column=0, columnspan=2, padx=12, pady=4, sticky="w")

        def update_info(*args):
            name = session_var.get()
            if name:
                sess_path = lr_dir / name
                label_files = list(sess_path.glob("*.txt"))
                changelog = sess_path / "changelog.json"
                changes = 0
                if changelog.exists():
                    try:
                        changes = len(json.loads(changelog.read_text()))
                    except Exception:
                        pass
                info_var.set(f"Labels: {len(label_files)}  |  Changes logged: {changes}  |  Path: {sess_path}")

        session_var.trace_add("write", update_info)
        update_info()

        def on_mode_toggle():
            if export_mode_var.get() == "merge_all":
                session_label.grid_remove()
                session_combo.grid_remove()
                # Show merge info
                dest = dest_var.get().strip()
                if dest:
                    matched = self._find_sessions_for_dataset(dest)
                    changed = self._collect_changed_stems(matched)
                    info_var.set(
                        f"Found {len(matched)} session(s) for this dataset\n"
                        f"Total unique files changed: {len(changed)}\n"
                        f"Merge strategy: box-level union (IoU ≥ 0.5 = duplicate)")
                else:
                    info_var.set("Set destination dir first, then switch to merge mode")
            else:
                session_label.grid()
                session_combo.grid()
                update_info()

        # Destination
        tk.Label(export_win, text="Destination (original labels dir):",
                 font=("Segoe UI", 10),
                 bg=self.bg, fg=self.fg).grid(row=5, column=0, padx=12, pady=(8, 4), sticky="w")

        dest_var = tk.StringVar(value=self.labels_dir_var.get())
        dest_entry = tk.Entry(export_win, textvariable=dest_var,
                              bg=self.field_bg, fg=self.fg,
                              insertbackground=self.fg,
                              relief="flat", font=("Consolas", 10))
        dest_entry.grid(row=6, column=0, padx=12, pady=4, sticky="ew")

        def browse_dest():
            path = filedialog.askdirectory(title="Select destination labels directory")
            if path:
                dest_var.set(path)
                on_mode_toggle()  # Refresh merge info

        tk.Button(export_win, text="Browse", command=browse_dest,
                  bg=self.btn_bg, fg=self.fg, relief="flat",
                  font=("Segoe UI", 9)).grid(row=6, column=1, padx=(0, 12), pady=4)

        # Backup checkbox
        backup_var = tk.BooleanVar(value=True)
        tk.Checkbutton(export_win, text="Create backup before overwriting",
                       variable=backup_var,
                       bg=self.bg, fg=self.fg, selectcolor=self.field_bg,
                       activebackground=self.bg, activeforeground=self.fg
                       ).grid(row=7, column=0, columnspan=2, padx=12, pady=4, sticky="w")

        export_win.columnconfigure(0, weight=1, minsize=350)

        def do_export():
            dest = dest_var.get().strip()
            if not dest:
                messagebox.showerror("Error", "Select a destination directory.", parent=export_win)
                return

            dest_path = Path(dest)
            if not dest_path.exists():
                messagebox.showerror("Error", f"Destination not found:\n{dest}", parent=export_win)
                return

            mode = export_mode_var.get()

            if mode == "merge_all":
                # Merge all sessions for this dataset — BOX-LEVEL merge
                matched_sessions = self._find_sessions_for_dataset(dest)
                if not matched_sessions:
                    messagebox.showwarning("No Matching Sessions",
                                           "No review sessions found for this dataset.\n"
                                           "Make sure you've reviewed labels from this directory.",
                                           parent=export_win)
                    return

                stem_map = self._collect_changed_stems(matched_sessions)
                if not stem_map:
                    messagebox.showwarning("No Changes",
                                           "No changes were logged in any session.\n"
                                           "Edit some labels in the reviewer first.",
                                           parent=export_win)
                    return

                # Confirm
                msg = (f"Box-level merge for {len(stem_map)} files from "
                       f"{len(matched_sessions)} session(s).\n\n"
                       f"Destination: {dest_path}\n\n"
                       f"For each file: union all boxes across sessions,\n"
                       f"deduplicate overlaps (IoU ≥ 0.5).\n\nProceed?")
                if not messagebox.askyesno("Confirm Merge Export", msg, parent=export_win):
                    return

                # Backup
                if backup_var.get():
                    backup_dir = dest_path.parent / f"{dest_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backed_up = 0
                    for stem in stem_map:
                        orig = dest_path / f"{stem}.txt"
                        if orig.exists():
                            shutil.copy2(orig, backup_dir / orig.name)
                            backed_up += 1
                    print(f"  Backup: {backed_up} original labels saved to {backup_dir}")

                # Box-level merge and write
                written = 0
                total_boxes_added = 0
                for stem, sess_list in stem_map.items():
                    merged_boxes = self._merge_label_files(stem, sess_list, dest_path)
                    # Write merged result
                    out_path = dest_path / f"{stem}.txt"
                    with open(out_path, "w") as f:
                        for cls, xc, yc, w, h in merged_boxes:
                            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    written += 1
                    total_boxes_added += len(merged_boxes)

                messagebox.showinfo("Merge Export Complete",
                                    f"Merged {written} label files\n"
                                    f"Total boxes: {total_boxes_added}\n"
                                    f"From {len(matched_sessions)} session(s)\n\n"
                                    f"Destination: {dest_path}",
                                    parent=export_win)
                export_win.destroy()

            else:
                # Single session export (original behavior)
                session_name = session_var.get()
                if not session_name:
                    messagebox.showerror("Error", "Select a session.", parent=export_win)
                    return

                sess_path = lr_dir / session_name
                label_files = [f for f in sess_path.glob("*.txt") if f.name != "classes.txt"]

                if not label_files:
                    messagebox.showwarning("No Labels", "No label files found in this session.",
                                           parent=export_win)
                    return

                count = len(label_files)
                existing = sum(1 for f in label_files if (dest_path / f.name).exists())
                msg = (f"Export {count} labels to:\n{dest_path}\n\n"
                       f"{existing} files will be overwritten.\n\nProceed?")
                if not messagebox.askyesno("Confirm Export", msg, parent=export_win):
                    return

                # Backup
                if backup_var.get() and existing > 0:
                    backup_dir = dest_path.parent / f"{dest_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for lf in label_files:
                        orig = dest_path / lf.name
                        if orig.exists():
                            shutil.copy2(orig, backup_dir / orig.name)
                    print(f"  Backup: {existing} original labels saved to {backup_dir}")

                copied = 0
                for lf in label_files:
                    shutil.copy2(lf, dest_path / lf.name)
                    copied += 1

                messagebox.showinfo("Export Complete",
                                    f"Exported {copied} labels to:\n{dest_path}",
                                    parent=export_win)
                export_win.destroy()

        # Export button
        tk.Button(export_win, text="Export",
                  font=("Segoe UI", 11, "bold"),
                  bg=self.export_color, fg="#1e1e2e",
                  relief="flat", cursor="hand2",
                  command=do_export).grid(row=8, column=0, columnspan=2,
                                          padx=12, pady=12, sticky="ew")

    def run(self):
        self.root.mainloop()


def launch_gui():
    """Entry point for the GUI launcher."""
    app = LabelReviewerGUI()
    app.run()
