"""
fusion_app.py — Dual-Modality Drone Detection Demo

Three detection modes:
  1. Single Model: one YOLO model, one video
  2. Paired Fusion: two synchronized videos (RGB + IR), fusion classifier
  3. Grayscale Fusion: one RGB video, IR model on grayscale, fusion classifier

Temporal logic (toggle on/off) mirrors gui/detection.py:
  - Stride-based inference (only run YOLO every Nth frame, cache results on hold frames)
  - TemporalContinuity spatial gate (detection.py verbatim)
  - N-of-M warning/alert windows with cooldowns
  - Temporal ROI propagation for missed detections
  - Rolling average confidence gate

Usage:
    python gui/fusion_app.py
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
try:
    from classifier.patch_verifier import PatchVerifier  # noqa: E402
except Exception:
    PatchVerifier = None

# Canonical fusion modules — single source of truth for feature extraction,
# temporal state, and overlay drawing. Do NOT reimplement these locally.
from fusion.features import (  # noqa: E402
    TARGET_NAMES, compute_global_features, compute_target_features,
)
from fusion.temporal import (  # noqa: E402
    TemporalContinuity, PerModalityTemporalState,
    draw_box, draw_detections, draw_temporal_overlays,
    overlay_text_big, build_overlay_lines,
)


def extract_fusion_features(rgb_gray, ir_gray, rgb_dets, ir_dets):
    """Build the 40-feature dict the classifier expects. Mirrors
    `FusionEngine.extract_features` exactly — kept at module level so the
    existing `_process_*` methods can call it without holding an engine
    reference."""
    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}
    for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [d.conf if hasattr(d, "conf") else d[4] for d in dets]
        n = len(confs)
        if n == 0:
            feats.update({f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.0,
                          f"{prefix}_mean_conf": 0.0, f"{prefix}_detected": 0})
        else:
            feats.update({f"{prefix}_n_dets": n,
                          f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
                          f"{prefix}_detected": 1})
    g_rgb = compute_global_features(rgb_gray, modality="rgb")
    g_ir  = compute_global_features(ir_gray, modality="ir")
    feats.update({f"rgb_{k}": v for k, v in g_rgb.items()})
    feats.update({f"ir_{k}": v for k, v in g_ir.items()})
    for prefix, dets, gray, gw, gh in (
        ("rgb", rgb_dets, rgb_gray, rgb_w, rgb_h),
        ("ir",  ir_dets,  ir_gray,  ir_w,  ir_h),
    ):
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best = max(dets, key=lambda d: (d.conf if hasattr(d, "conf") else d[4]))
            bb = best.box if hasattr(best, "box") else (best[0], best[1], best[2], best[3])
            tf = compute_target_features(gray, bb, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    rd, id_ = len(rgb_dets) > 0, len(ir_dets) > 0
    feats["both_detect"]     = int(rd and id_)
    feats["neither_detect"]  = int(not rd and not id_)
    feats["rgb_only_detect"] = int(rd and not id_)
    feats["ir_only_detect"]  = int(not rd and id_)
    return feats

WORKSPACE = Path(__file__).resolve().parent.parent

# ── DEFAULTS ────────────────────────────────────────────────────────
DEFAULTS = {
    "rgb_model": str(WORKSPACE / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
    "ir_model": str(WORKSPACE / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"),
    "fusion_model": str(WORKSPACE / "classifier" / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib"),
    "rgb_conf": 0.25,
    "ir_conf_real": 0.40,
    "ir_conf_gray": 0.05,
    "nms_iou": 0.45,
    "imgsz": 640,
    "gpu_device": 0,
    # Temporal settings (matching detection.py defaults)
    "infer_fps": 5,
    "warning_window_frames": 10,
    "warning_require_hits": 9,
    "alert_window_frames": 10,
    "alert_require_hits": 9,
    "alert_avg_conf_threshold": 0.0,
    "warning_cooldown_s": 3.0,
    "alert_cooldown_s": 3.0,
    "roi_ttl": 5,
    "roi_expand": 1.5,
    # Overlay toggles (matching detection.py)
    "show_troi": True,
    "show_gate": True,
    "show_source_tags": True,
    # Patch verifier (post-classifier drone-vs-aerial veto)
    "rgb_patch_weights": str(WORKSPACE / "classifier" / "runs" / "patches" / "confuser_filter4_rgb.pt"),
    "ir_patch_weights": str(WORKSPACE / "classifier" / "runs" / "patches" / "confuser_filter4_ir.pt"),
    "use_patch_verifier": True,
    "patch_threshold": 0.70,
    # Grayscale test mode — run IR filter on gray-replicate RGB (diagnostic).
    "grayscale_run_ir_filter": True,
    "grayscale_disable_filter_ood": True,
}

SETTINGS_PATH = Path(__file__).resolve().parent / "fusion_settings.json"

TRUST_LABELS = {0: "REJECT BOTH", 1: "TRUST RGB", 2: "TRUST IR", 3: "TRUST BOTH"}
TRUST_COLORS_BGR = {
    0: (0, 0, 200),     # red
    1: (0, 200, 0),     # green
    2: (200, 200, 0),   # cyan
    3: (0, 255, 0),     # bright green
}
TRUST_COLORS_HEX = {
    0: "#c83232",
    1: "#32c832",
    2: "#32c8c8",
    3: "#00ff00",
}



# (Local duplicates of feature / temporal / drawing helpers removed —
#  canonical versions imported above from fusion.features / fusion.temporal.)


# ── MAIN APP CLASS ──────────────────────────────────────────────────

class FusionApp:
    MODES = ["Single Model", "Paired Fusion", "Grayscale Fusion"]

    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection — Fusion Demo")
        self.root.geometry("1300x800")
        self.root.minsize(900, 600)

        # State
        self.rgb_model = None
        self.ir_model = None
        self.fusion_clf = None
        self.fusion_features = None
        self.rgb_verifier = None
        self.ir_verifier = None
        self.cap_left = None
        self.cap_right = None
        self.running = False
        self.paused = False
        self.writer = None
        self.lock = threading.Lock()

        # Temporal state
        self.rgb_temporal = None
        self.ir_temporal = None

        # Display cache (the anti-flicker mechanism from detection.py)
        self._cached_annotated = None
        self._cached_trust = None
        self._cached_trust_prob = None

        # Stats
        self.fps = 0.0
        self.frame_num = 0
        self.total_frames = 0
        self.playback_speed = 1.0

        # Load settings
        self.settings = dict(DEFAULTS)
        self._load_settings()

        # Tkinter variables
        self.mode_var = tk.StringVar(value=self.MODES[0])
        self.temporal_var = tk.BooleanVar(value=False)
        self.video_left_path = tk.StringVar()
        self.video_right_path = tk.StringVar()
        self.save_path = tk.StringVar()

        self._build_ui()

    # ── SETTINGS ────────────────────────────────────────────────────

    def _load_settings(self):
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH) as f:
                    saved = json.load(f)
                self.settings.update(saved)
            except Exception:
                pass

    def _save_settings(self):
        with open(SETTINGS_PATH, "w") as f:
            json.dump(self.settings, f, indent=2)

    # ── UI CONSTRUCTION ─────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.LabelFrame(self.root, text="Configuration", padding=8)
        top.pack(fill="x", padx=8, pady=(8, 4))

        row = 0
        ttk.Label(top, text="Mode:").grid(row=row, column=0, sticky="w")
        mode_menu = ttk.OptionMenu(top, self.mode_var, self.MODES[0], *self.MODES,
                                    command=self._on_mode_change)
        mode_menu.grid(row=row, column=1, sticky="w", padx=4)
        ttk.Button(top, text="Settings", command=self._open_settings).grid(
            row=row, column=3, padx=8)

        row = 1
        self.lbl_left = ttk.Label(top, text="Video:")
        self.lbl_left.grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.video_left_path, width=55).grid(
            row=row, column=1, columnspan=2, padx=4, pady=(4, 0), sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_left).grid(
            row=row, column=3, pady=(4, 0))

        row = 2
        self.lbl_right = ttk.Label(top, text="IR Video:")
        self.lbl_right.grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.entry_right = ttk.Entry(top, textvariable=self.video_right_path, width=55)
        self.entry_right.grid(row=row, column=1, columnspan=2, padx=4, pady=(4, 0), sticky="ew")
        self.btn_right = ttk.Button(top, text="Browse", command=self._browse_right)
        self.btn_right.grid(row=row, column=3, pady=(4, 0))

        row = 3
        ttk.Label(top, text="Save to:").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.save_path, width=55).grid(
            row=row, column=1, columnspan=2, padx=4, pady=(4, 0), sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_save).grid(
            row=row, column=3, pady=(4, 0))

        row = 4
        ttk.Label(top, text="YouTube:").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.yt_url_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.yt_url_var, width=55).grid(
            row=row, column=1, columnspan=2, padx=4, pady=(4, 0), sticky="ew")
        self.btn_yt = ttk.Button(top, text="Download", command=self._download_youtube)
        self.btn_yt.grid(row=row, column=3, pady=(4, 0))

        top.columnconfigure(1, weight=1)

        # Controls
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill="x", padx=8)

        self.btn_play = ttk.Button(ctrl, text="Play", command=self._play)
        self.btn_play.pack(side="left", padx=2)
        self.btn_pause = ttk.Button(ctrl, text="Pause", command=self._pause, state="disabled")
        self.btn_pause.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=2)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        # Speed controls
        ttk.Label(ctrl, text="Speed:").pack(side="left", padx=(0, 2))
        self.speed_var = tk.StringVar(value="1x")
        for spd_label, spd_val in [("1x", 1.0), ("2x", 2.0), ("4x", 4.0), ("Max", 0.0)]:
            rb = ttk.Radiobutton(ctrl, text=spd_label, variable=self.speed_var,
                                 value=spd_label,
                                 command=lambda v=spd_val: self._set_speed(v))
            rb.pack(side="left", padx=1)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Button(ctrl, text="Skip 30s ▶▶", command=self._skip_forward).pack(side="left", padx=2)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        self.temporal_check = ttk.Checkbutton(
            ctrl, text="Temporal Logic", variable=self.temporal_var)
        self.temporal_check.pack(side="left", padx=4)

        self.stats_label = ttk.Label(ctrl, text="Ready")
        self.stats_label.pack(side="right")

        # Canvas area
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas_left = tk.Canvas(canvas_frame, bg="#1a1a2e", highlightthickness=0)
        self.canvas_left.pack(side="left", fill="both", expand=True)
        self.canvas_right = tk.Canvas(canvas_frame, bg="#1e1e3a", highlightthickness=0)
        self.canvas_right.pack(side="right", fill="both", expand=True)

        # Fusion status bar
        self.fusion_frame = ttk.Frame(self.root, padding=(8, 2))
        self.fusion_frame.pack(fill="x", padx=8)
        self.fusion_label = ttk.Label(self.fusion_frame, text="", font=("Consolas", 11, "bold"))
        self.fusion_label.pack(side="left")
        self.fusion_conf = ttk.Label(self.fusion_frame, text="")
        self.fusion_conf.pack(side="left", padx=8)

        # Progress
        prog_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        prog_frame.pack(fill="x")
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(prog_frame, text="")
        self.progress_label.pack(anchor="e")

        # Status bar
        self.status = ttk.Label(self.root, text="Ready. Select mode, load video(s), and press Play.",
                                relief="sunken", anchor="w", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")

        # Canvas image references (prevent GC)
        self._photo_left = None
        self._photo_right = None
        # Canvas item IDs for in-place update (no delete+create flicker)
        self._img_id_left = None
        self._img_id_right = None

        self._on_mode_change(self.mode_var.get())

    def _on_mode_change(self, mode):
        is_paired = (mode == "Paired Fusion")
        is_fusion = mode in ("Paired Fusion", "Grayscale Fusion")

        if is_paired:
            self.lbl_left.config(text="RGB Video:")
            self.lbl_right.grid()
            self.entry_right.grid()
            self.btn_right.grid()
            self.canvas_right.pack(side="right", fill="both", expand=True)
        elif mode == "Grayscale Fusion":
            self.lbl_left.config(text="RGB Video:")
            self.lbl_right.grid_remove()
            self.entry_right.grid_remove()
            self.btn_right.grid_remove()
            self.canvas_right.pack(side="right", fill="both", expand=True)
        else:
            self.lbl_left.config(text="Video:")
            self.lbl_right.grid_remove()
            self.entry_right.grid_remove()
            self.btn_right.grid_remove()
            self.canvas_right.pack_forget()

        if is_fusion:
            self.fusion_frame.pack(fill="x", padx=8, before=self.status)
        else:
            self.fusion_frame.pack_forget()

    # ── BROWSE ──────────────────────────────────────────────────────

    def _browse_left(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All", "*.*")])
        if path:
            self.video_left_path.set(path)
            base, ext = os.path.splitext(path)
            self.save_path.set(base + "_fusion" + ext)

    def _browse_right(self):
        path = filedialog.askopenfilename(
            title="Select IR video",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All", "*.*")])
        if path:
            self.video_right_path.set(path)

    def _browse_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
        if path:
            self.save_path.set(path)

    # ── YOUTUBE DOWNLOAD ────────────────────────────────────────────

    def _download_youtube(self):
        url = self.yt_url_var.get().strip()
        if not url:
            messagebox.showwarning("YouTube", "Enter a YouTube URL first.")
            return

        self.status.config(text="Downloading YouTube video...")
        self.btn_yt.config(state="disabled")
        self.root.update_idletasks()

        def _dl_thread():
            try:
                import yt_dlp
                import re as _re

                vid_match = _re.search(
                    r'(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([a-zA-Z0-9_-]{11})', url)
                vid_id = vid_match.group(1) if vid_match else "video"

                dl_dir = Path(__file__).resolve().parent / "demo_outputs"
                dl_dir.mkdir(exist_ok=True)
                dl_path = str(dl_dir / f"yt_{vid_id}.mp4")

                # Reuse cached download
                if os.path.exists(dl_path) and os.path.getsize(dl_path) > 100_000:
                    self.root.after(0, lambda: self._yt_done(dl_path, cached=True))
                    return

                # Remove partial file
                if os.path.exists(dl_path):
                    os.remove(dl_path)

                ydl_opts = {
                    'format': 'best[height<=720]',
                    'outtmpl': dl_path,
                    'quiet': True,
                    'no_warnings': True,
                    'overwrites': True,
                    'continuedl': False,
                    'socket_timeout': 30,
                    'retries': 5,
                    'fragment_retries': 5,
                }

                # Auto-extract cookies from browser
                cookie_file = str(Path(__file__).resolve().parent / "youtube_cookies.txt")
                browser_cookie_set = False
                for browser in ['chrome', 'opera', 'edge']:
                    try:
                        test_opts = {**ydl_opts, 'cookiesfrombrowser': (browser,),
                                     'extract_flat': True}
                        with yt_dlp.YoutubeDL(test_opts) as test_ydl:
                            test_ydl.extract_info(url, download=False)
                        ydl_opts['cookiesfrombrowser'] = (browser,)
                        print(f"[YT] Using cookies from {browser}")
                        browser_cookie_set = True
                        break
                    except Exception:
                        continue

                if not browser_cookie_set and os.path.isfile(cookie_file):
                    ydl_opts['cookiefile'] = cookie_file
                    print(f"[YT] Falling back to cookie file: {cookie_file}")

                print(f"[YT] Downloading {vid_id} to {dl_path}...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                print(f"[YT] Download complete: {dl_path}")

                self.root.after(0, lambda: self._yt_done(dl_path, cached=False))

            except Exception as e:
                self.root.after(0, lambda: self._yt_error(str(e)))

        threading.Thread(target=_dl_thread, daemon=True).start()

    def _yt_done(self, path, cached=False):
        self.video_left_path.set(path)
        base, ext = os.path.splitext(path)
        self.save_path.set(base + "_fusion" + ext)
        tag = "cached" if cached else "downloaded"
        self.status.config(text=f"YouTube video {tag}: {os.path.basename(path)}")
        self.btn_yt.config(state="normal")

    def _yt_error(self, msg):
        self.status.config(text=f"YouTube error: {msg}")
        self.btn_yt.config(state="normal")
        messagebox.showerror("YouTube Error", msg)

    # ── SETTINGS DIALOG ────────────────────────────────────────────

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("600x580")
        win.transient(self.root)

        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)

        entries = {}
        labels = [
            ("rgb_model", "RGB Model Path"),
            ("ir_model", "IR Model Path"),
            ("fusion_model", "Fusion Classifier Path"),
            ("rgb_conf", "RGB Confidence"),
            ("ir_conf_real", "IR Confidence (paired + grayscale)"),
            ("nms_iou", "NMS IoU"),
            ("grayscale_run_ir_filter", "Grayscale: run IR filter"),
            ("imgsz", "Image Size"),
            ("gpu_device", "GPU Device"),
            ("infer_fps", "Temporal Infer FPS"),
            ("warning_window_frames", "Warning Window (M)"),
            ("warning_require_hits", "Warning Require Hits (N)"),
            ("alert_window_frames", "Alert Window (M)"),
            ("alert_require_hits", "Alert Require Hits (N)"),
            ("alert_avg_conf_threshold", "Alert Avg Conf Threshold"),
            ("warning_cooldown_s", "Warning Cooldown (s)"),
            ("alert_cooldown_s", "Alert Cooldown (s)"),
            ("roi_ttl", "ROI Propagation TTL"),
            ("roi_expand", "ROI Expand Factor"),
            ("show_troi", "Show TROI ROI boxes"),
            ("show_gate", "Show TC Gate boxes"),
            ("show_source_tags", "Show Source Tags (FULL/TROI)"),
        ]

        for i, (key, label) in enumerate(labels):
            ttk.Label(frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(self.settings[key]))
            e = ttk.Entry(frame, textvariable=var, width=50)
            e.grid(row=i, column=1, padx=8, pady=2, sticky="ew")
            entries[key] = var

            if key.endswith("model"):
                def make_browse(entry_var=var, file_types=None):
                    def browse():
                        ft = file_types or [("All", "*.*")]
                        p = filedialog.askopenfilename(filetypes=ft)
                        if p:
                            entry_var.set(p)
                    return browse
                ft = [("PT", "*.pt")] if key != "fusion_model" else [("Joblib", "*.joblib")]
                ttk.Button(frame, text="...", width=3,
                           command=make_browse(var, ft)).grid(row=i, column=2)

        frame.columnconfigure(1, weight=1)

        float_keys = ("rgb_conf", "ir_conf_real", "nms_iou",
                       "alert_avg_conf_threshold", "warning_cooldown_s",
                       "alert_cooldown_s", "roi_expand")
        int_keys = ("imgsz", "gpu_device", "infer_fps", "warning_window_frames",
                     "warning_require_hits", "alert_window_frames",
                     "alert_require_hits", "roi_ttl")
        bool_keys = ("show_troi", "show_gate", "show_source_tags",
                     "grayscale_run_ir_filter")

        def save():
            for key, var in entries.items():
                val = var.get()
                if key in float_keys:
                    self.settings[key] = float(val)
                elif key in int_keys:
                    self.settings[key] = int(val)
                elif key in bool_keys:
                    self.settings[key] = val.lower() in ("true", "1", "yes")
                else:
                    self.settings[key] = val
            self._save_settings()
            win.destroy()
            self.status.config(text="Settings saved.")

        ttk.Button(frame, text="Save", command=save).grid(
            row=len(labels), column=1, pady=16, sticky="e")

    # ── MODEL LOADING ───────────────────────────────────────────────

    def _load_models(self):
        mode = self.mode_var.get()
        s = self.settings

        if mode == "Single Model":
            self.status.config(text="Loading model...")
            self.root.update()
            self.rgb_model = YOLO(s["rgb_model"])
            self.ir_model = None
            self.fusion_clf = None
        else:
            self.status.config(text="Loading RGB model...")
            self.root.update()
            self.rgb_model = YOLO(s["rgb_model"])
            self.status.config(text="Loading IR model...")
            self.root.update()
            self.ir_model = YOLO(s["ir_model"])
            self.status.config(text="Loading fusion classifier...")
            self.root.update()
            bundle = joblib.load(s["fusion_model"])
            self.fusion_clf = bundle["model"]
            self.fusion_features = bundle["features"]

            self.rgb_verifier = None
            self.ir_verifier = None
            if s.get("use_patch_verifier", True) and PatchVerifier is not None:
                rgb_w = s.get("rgb_patch_weights")
                ir_w = s.get("ir_patch_weights")
                if rgb_w and os.path.isfile(rgb_w):
                    self.status.config(text="Loading RGB patch verifier...")
                    self.root.update()
                    self.rgb_verifier = PatchVerifier(rgb_w)
                if ir_w and os.path.isfile(ir_w):
                    self.status.config(text="Loading IR patch verifier...")
                    self.root.update()
                    self.ir_verifier = PatchVerifier(ir_w)

    # ── PLAYBACK ────────────────────────────────────────────────────

    def _play(self):
        if self.paused:
            self.paused = False
            self.btn_pause.config(text="Pause")
            self.status.config(text="Resumed.")
            return

        mode = self.mode_var.get()
        left_path = self.video_left_path.get().strip()

        if not left_path or not os.path.isfile(left_path):
            messagebox.showerror("Error", "Select a valid video file.")
            return

        if mode == "Paired Fusion":
            right_path = self.video_right_path.get().strip()
            if not right_path or not os.path.isfile(right_path):
                messagebox.showerror("Error", "Select a valid IR video file.")
                return

        try:
            self._load_models()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            return

        self.cap_left = cv2.VideoCapture(left_path)
        if not self.cap_left.isOpened():
            messagebox.showerror("Error", "Failed to open video.")
            return

        if mode == "Paired Fusion":
            self.cap_right = cv2.VideoCapture(self.video_right_path.get().strip())
            if not self.cap_right.isOpened():
                messagebox.showerror("Error", "Failed to open IR video.")
                return
        else:
            self.cap_right = None

        self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.progress["maximum"] = max(self.total_frames, 1)
        self.progress["value"] = 0

        # Clear display cache
        self._cached_annotated = None
        self._cached_trust = None
        self._cached_trust_prob = None
        self._img_id_left = None
        self._img_id_right = None

        # Temporal state
        s = self.settings
        fps_src = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
        self._stride = max(1, int(round(fps_src / s["infer_fps"])))
        infer_fps = s["infer_fps"]
        warn_cd_infer = int(round(s["warning_cooldown_s"] * infer_fps))
        alert_cd_infer = int(round(s["alert_cooldown_s"] * infer_fps))

        self.rgb_temporal = PerModalityTemporalState(
            stride=self._stride,
            warning_window=s["warning_window_frames"],
            warning_require=s["warning_require_hits"],
            alert_window=s["alert_window_frames"],
            alert_require=s["alert_require_hits"],
            alert_avg_conf_thresh=s["alert_avg_conf_threshold"],
            warning_cooldown_frames=warn_cd_infer,
            alert_cooldown_frames=alert_cd_infer,
            roi_ttl=s["roi_ttl"], roi_expand=s["roi_expand"],
        )
        self.ir_temporal = PerModalityTemporalState(
            stride=self._stride,
            warning_window=s["warning_window_frames"],
            warning_require=s["warning_require_hits"],
            alert_window=s["alert_window_frames"],
            alert_require=s["alert_require_hits"],
            alert_avg_conf_thresh=s["alert_avg_conf_threshold"],
            warning_cooldown_frames=warn_cd_infer,
            alert_cooldown_frames=alert_cd_infer,
            roi_ttl=s["roi_ttl"], roi_expand=s["roi_expand"],
        )

        # Video writer
        sp = self.save_path.get().strip()
        if sp:
            w = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w = w * 2 if mode != "Single Model" else w
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(sp) if os.path.dirname(sp) else ".", exist_ok=True)
            self.writer = cv2.VideoWriter(sp, fourcc, fps_src, (out_w, h))

        self.running = True
        self.paused = False
        self.btn_play.config(text="Play")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.status.config(text=f"Playing: {os.path.basename(left_path)} [{mode}]")

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.btn_pause.config(text="Resume" if self.paused else "Pause")

    def _stop(self):
        self.running = False
        self.paused = False
        self._cleanup()
        self.btn_play.config(text="Play")
        self.btn_pause.config(text="Pause", state="disabled")
        self.btn_stop.config(state="disabled")
        self.status.config(text="Stopped.")

    def _cleanup(self):
        with self.lock:
            if self.cap_left and self.cap_left.isOpened():
                self.cap_left.release()
                self.cap_left = None
            if self.cap_right and self.cap_right.isOpened():
                self.cap_right.release()
                self.cap_right = None
            if self.writer:
                self.writer.release()
                self.writer = None
    # ── SPEED / SKIP ──────────────────────────────────────────────────

    def _set_speed(self, speed):
        self.playback_speed = speed
        label = "Max" if speed == 0 else f"{speed:.0f}x"
        self.status.config(text=f"Playback speed: {label}")

    def _skip_forward(self):
        """Skip forward 30 seconds in the video."""
        if not self.running or self.cap_left is None:
            return
        with self.lock:
            fps_src = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
            skip_frames = int(fps_src * 30)  # 30 seconds
            current = int(self.cap_left.get(cv2.CAP_PROP_POS_FRAMES))
            target = min(current + skip_frames, self.total_frames - 1)
            self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, target)
            if self.cap_right and self.cap_right.isOpened():
                self.cap_right.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.frame_num = target
        self.status.config(text=f"Skipped to frame {target}/{self.total_frames}")

    # ── PROCESSING LOOP ─────────────────────────────────────────────

    def _process_loop(self):
        mode = self.mode_var.get()
        s = self.settings
        device = s["gpu_device"]
        imgsz = s["imgsz"]
        temporal_on = self.temporal_var.get()
        stride = self._stride if temporal_on else 1

        # Pacing: match source FPS
        fps_src = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
        frame_period = 1.0 / fps_src
        next_deadline = time.perf_counter()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                next_deadline = time.perf_counter()
                continue

            with self.lock:
                if self.cap_left is None or not self.cap_left.isOpened():
                    break
                ret_l, frame_left = self.cap_left.read()
                if not ret_l:
                    break
                frame_right = None
                if self.cap_right:
                    ret_r, frame_right = self.cap_right.read()
                    if not ret_r:
                        break

            self.frame_num += 1
            is_infer = (self.frame_num % stride == 0) or stride == 1

            t0 = time.perf_counter()

            if is_infer:
                # === INFERENCE FRAME: run YOLO, update temporal, cache result ===
                if mode == "Single Model":
                    annotated, trust, trust_prob = self._process_single(
                        frame_left, s, device, imgsz)
                elif mode == "Paired Fusion":
                    annotated, trust, trust_prob = self._process_paired(
                        frame_left, frame_right, s, device, imgsz)
                else:
                    annotated, trust, trust_prob = self._process_grayscale(
                        frame_left, s, device, imgsz)

                # Cache for hold frames
                self._cached_annotated = annotated
                self._cached_trust = trust
                self._cached_trust_prob = trust_prob
            else:
                # === HOLD FRAME: draw cached boxes on fresh video ===
                if mode == "Single Model":
                    annotated = self._hold_single(frame_left)
                elif mode == "Paired Fusion":
                    annotated = self._hold_paired(frame_left, frame_right)
                else:
                    annotated = self._hold_grayscale(frame_left)

                trust = self._cached_trust
                trust_prob = self._cached_trust_prob

            elapsed = time.perf_counter() - t0
            self.fps = 1.0 / max(elapsed, 1e-6)

            with self.lock:
                if self.writer:
                    self.writer.write(annotated)

            self.root.after(0, self._update_display, annotated, trust, trust_prob, mode)

            # Pace to source FPS (adjusted by playback speed)
            now = time.perf_counter()
            speed = self.playback_speed
            if speed > 0:
                adjusted_period = frame_period / speed
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                next_deadline = max(next_deadline + adjusted_period, time.perf_counter())
            else:
                # Max speed: no sleep, run as fast as possible
                next_deadline = time.perf_counter()

        self.running = False
        self.root.after(0, self._on_video_end)

    def _run_yolo(self, model, frame, conf, device, imgsz):
        """Run YOLO and return list of [x1, y1, x2, y2, conf]."""
        results = model.predict(frame, conf=conf, iou=self.settings["nms_iou"],
                                imgsz=imgsz, max_det=20, verbose=False, device=device)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                c = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, c])
        return dets

    def _run_with_roi_recovery(self, model, frame, conf, device, imgsz, temporal):
        """Run YOLO with temporal ROI propagation recovery.
        Returns (dets, sources, troi_rois) where sources is per-det 'full'/'troi'."""
        dets = self._run_yolo(model, frame, conf, device, imgsz)
        sources = ["full"] * len(dets)
        troi_rois = []

        if temporal is not None:
            if len(dets) == 0 and temporal.last_roi is not None and temporal.roi_age > 0:
                h, w = frame.shape[:2]
                roi_result = temporal.get_roi_crop(frame, w, h)
                if roi_result is not None:
                    crop, (ox, oy) = roi_result
                    # Track the ROI region for visualization
                    troi_rois.append((ox, oy, ox + crop.shape[1], oy + crop.shape[0]))
                    crop_dets = self._run_yolo(model, crop, conf * 0.8, device, imgsz)
                    if crop_dets:
                        dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                        sources = ["troi"] * len(dets)

        return dets, sources, troi_rois

    # ── OVERLAY BUILDER (matches detection.py windows_big format) ───

    @staticmethod
    def _build_overlay_lines(temporal, settings, prefix=""):
        """Build overlay lines matching detection.py's TOPLEFT_LOG_MODE='windows_big' format."""
        if temporal is None:
            return []

        w_hits = sum(temporal.win_warning)
        w_len = len(temporal.win_warning)
        w_need = settings["warning_require_hits"]
        w_win = settings["warning_window_frames"]
        w_events = temporal.count_warning_events

        a_hits = sum(temporal.win_alert)
        a_len = len(temporal.win_alert)
        a_need = settings["alert_require_hits"]
        a_win = settings["alert_window_frames"]
        a_events = temporal.count_alert_events

        # Avg conf string (only show if threshold > 0, matching detection.py)
        avg_conf_str = ""
        if float(settings.get("alert_avg_conf_threshold", 0)) > 0:
            hit_confs = [c for c, h in zip(temporal.alert_conf_window, temporal.win_alert) if h]
            avg_conf = (sum(hit_confs) / len(hit_confs)) if hit_confs else 0.0
            avg_conf_str = f"  avgConf={avg_conf:.2f}"

        lines = [
            f"{prefix}WARNING hits {w_hits}/{w_len}  need {w_need}/{w_win}  events={w_events}",
            f"{prefix}ALERT   hits {a_hits}/{a_len}  need {a_need}/{a_win}{avg_conf_str}  events={a_events}",
        ]
        return lines

    # ── INFERENCE-FRAME PROCESSORS ──────────────────────────────────

    def _process_single(self, frame, s, device, imgsz):
        temporal_on = self.temporal_var.get()
        temporal = self.rgb_temporal if temporal_on else None

        dets, sources, troi_rois = self._run_with_roi_recovery(
            self.rgb_model, frame, s["rgb_conf"], device, imgsz, temporal)

        annotated = frame.copy()
        draw_detections(annotated, dets, (0, 255, 255), sources=sources,
                        show_source_tags=s.get("show_source_tags", False))

        if temporal_on and self.rgb_temporal:
            h, w = frame.shape[:2]
            self.rgb_temporal.update(dets, w, h)
            self.rgb_temporal.last_dets = list(dets)
            self.rgb_temporal.last_dets_sources = list(sources)
            self.rgb_temporal.last_troi_rois = list(troi_rois)

            # Draw TROI and gate overlays
            draw_temporal_overlays(annotated, self.rgb_temporal, s)

            # Overlay text (detection.py windows_big format)
            lines = self._build_overlay_lines(self.rgb_temporal, s)
            overlay_text_big(annotated, lines)

        return annotated, None, None

    def _build_verifier_lines(self, orig_trust, new_trust, rgb_probs, ir_probs,
                              ir_skipped_gray, n_rgb_dets, n_ir_dets, s,
                              rgb_is_grayscale=False):
        """Return overlay lines describing confuser-filter activity for this frame."""
        thr = float(s.get("patch_threshold", 0.5))
        lines = []
        if orig_trust != new_trust:
            lines.append(
                f"CONFUSER VETO: {TRUST_LABELS[orig_trust]} -> "
                f"{TRUST_LABELS[new_trust]}"
            )
        else:
            lines.append(f"CONFUSER FILTER: pass (thr={thr:.2f})")
        # RGB side
        rgb_trust_orig = orig_trust in (1, 3)
        if not rgb_trust_orig:
            lines.append("  RGB filter: classifier did not trust")
        elif rgb_is_grayscale:
            lines.append("  RGB filter: skipped (input is grayscale)")
        elif n_rgb_dets == 0:
            lines.append("  RGB filter: no detections")
        elif self.rgb_verifier is None:
            lines.append("  RGB filter: model not loaded")
        elif not rgb_probs:
            lines.append("  RGB filter: skipped")
        else:
            mx = max(rgb_probs)
            tag = "VETO" if mx >= thr else "OK"
            lines.append(
                f"  RGB filter: max P(confuser)={mx:.2f} "
                f"on {len(rgb_probs)} box(es) [{tag}]"
            )
        # IR side
        ir_trust_orig = orig_trust in (2, 3)
        if not ir_trust_orig:
            lines.append("  IR filter: classifier did not trust")
        elif ir_skipped_gray:
            lines.append("  IR filter: skipped (grayscale, OOD)")
        elif n_ir_dets == 0:
            lines.append("  IR filter: no detections")
        elif self.ir_verifier is None:
            lines.append("  IR filter: model not loaded")
        elif not ir_probs:
            lines.append("  IR filter: skipped")
        else:
            mx = max(ir_probs)
            tag = "VETO" if mx >= thr else "OK"
            lines.append(
                f"  IR filter: max P(confuser)={mx:.2f} "
                f"on {len(ir_probs)} box(es) [{tag}]"
            )
        return lines

    def _apply_patch_veto(self, trust, rgb_dets, ir_dets, rgb_bgr, ir_bgr,
                          ir_is_real_thermal, s,
                          ir_verifier_enabled=None):
        """Post-classifier confuser-filter veto. Returns (new_trust, rgb_probs, ir_probs, vetoed).

        `ir_verifier_enabled` (optional) overrides the `ir_is_real_thermal`
        gate. Grayscale test mode sets this True to exercise the IR filter on
        gray-replicate input.
        """
        if not s.get("use_patch_verifier", True) or trust == 0:
            return trust, [], [], False
        thr = float(s.get("patch_threshold", 0.70))
        trust_rgb = trust in (1, 3)
        trust_ir = trust in (2, 3)

        def _is_gray(img, max_diff=5):
            if img is None or img.ndim != 3 or img.shape[2] != 3:
                return True
            h, w = img.shape[:2]
            step = max(1, min(h, w) // 64)
            sample = img[::step, ::step].astype(np.int16)
            b, g, r = sample[..., 0], sample[..., 1], sample[..., 2]
            return (int(np.abs(b - g).max()) <= max_diff
                    and int(np.abs(g - r).max()) <= max_diff)

        rgb_is_color = not _is_gray(rgb_bgr)
        ir_active = (bool(ir_verifier_enabled)
                     if ir_verifier_enabled is not None
                     else bool(ir_is_real_thermal))
        rgb_probs, ir_probs = [], []
        if trust_rgb and self.rgb_verifier is not None and rgb_is_color and rgb_dets:
            boxes = [d.box for d in rgb_dets]
            rgb_probs = self.rgb_verifier.predict_boxes(rgb_bgr, boxes).tolist()
        if trust_ir and self.ir_verifier is not None and ir_active and ir_dets:
            boxes = [d.box for d in ir_dets]
            ir_probs = self.ir_verifier.predict_boxes(ir_bgr, boxes).tolist()
        rgb_ok = True
        ir_ok = True
        if trust_rgb and self.rgb_verifier is not None and rgb_is_color and rgb_probs:
            rgb_ok = max(rgb_probs) < thr
        if trust_ir and self.ir_verifier is not None and ir_active and ir_probs:
            ir_ok = max(ir_probs) < thr
        new_trust_rgb = trust_rgb and rgb_ok
        new_trust_ir = trust_ir and ir_ok
        if new_trust_rgb and new_trust_ir:
            new_trust = 3
        elif new_trust_rgb:
            new_trust = 1
        elif new_trust_ir:
            new_trust = 2
        else:
            new_trust = 0
        return new_trust, rgb_probs, ir_probs, new_trust != trust

    def _process_paired(self, frame_rgb, frame_ir, s, device, imgsz):
        temporal_on = self.temporal_var.get()
        rgb_t = self.rgb_temporal if temporal_on else None
        ir_t = self.ir_temporal if temporal_on else None

        rgb_dets, rgb_src, rgb_troi = self._run_with_roi_recovery(
            self.rgb_model, frame_rgb, s["rgb_conf"], device, imgsz, rgb_t)
        ir_dets, ir_src, ir_troi = self._run_with_roi_recovery(
            self.ir_model, frame_ir, s["ir_conf_real"], device, imgsz, ir_t)

        # Fusion
        rgb_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY) if len(frame_ir.shape) == 3 else frame_ir
        feat = extract_fusion_features(rgb_gray, ir_gray, rgb_dets, ir_dets)
        X = np.array([[feat[f] for f in self.fusion_features]])
        trust = int(self.fusion_clf.predict(X)[0])
        probs = self.fusion_clf.predict_proba(X)[0]
        trust_prob = float(probs[trust])

        ir_bgr = frame_ir if len(frame_ir.shape) == 3 else cv2.cvtColor(frame_ir, cv2.COLOR_GRAY2BGR)
        orig_trust = trust
        trust, rgp, irp, _veto = self._apply_patch_veto(
            trust, rgb_dets, ir_dets, frame_rgb, ir_bgr,
            ir_is_real_thermal=True, s=s)
        verif_lines = self._build_verifier_lines(
            orig_trust, trust, rgp, irp,
            ir_skipped_gray=False, n_rgb_dets=len(rgb_dets),
            n_ir_dets=len(ir_dets), s=s)

        rgb_trusted = [trust in (1, 3)] * len(rgb_dets)
        ir_trusted = [trust in (2, 3)] * len(ir_dets)
        show_tags = s.get("show_source_tags", False)

        left = frame_rgb.copy()
        right = frame_ir.copy()
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        draw_detections(left, rgb_dets, (0, 255, 0), "RGB ", rgb_trusted, rgb_src, show_tags)
        draw_detections(right, ir_dets, (255, 200, 0), "IR ", ir_trusted, ir_src, show_tags)

        if temporal_on and self.rgb_temporal:
            draw_temporal_overlays(left, self.rgb_temporal, s)
        if temporal_on and self.ir_temporal:
            draw_temporal_overlays(right, self.ir_temporal, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))

        combined = np.hstack([left, right])

        trust_label = TRUST_LABELS[trust]
        lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%)"]
        lines += verif_lines

        if temporal_on and self.rgb_temporal and self.ir_temporal:
            # Shared confuser feed: paired/grayscale image the same scene,
            # so the stronger filter (typically IR on helicopters) informs
            # both alert chains.
            rgb_max_p = float(max(rgp)) if rgp else None
            ir_max_p  = float(max(irp)) if irp else None
            shared_p = (max(p for p in (rgb_max_p, ir_max_p) if p is not None)
                        if (rgb_max_p is not None or ir_max_p is not None)
                        else None)
            self.rgb_temporal.add_confuser_prob(shared_p)
            self.ir_temporal.add_confuser_prob(shared_p)
            thr = float(s.get("patch_threshold", 0.70)) \
                  if s.get("use_patch_verifier", True) else None
            self.rgb_temporal.update(
                rgb_dets if trust in (1, 3) else [], lw, lh,
                confuser_threshold=thr)
            self.ir_temporal.update(
                ir_dets if trust in (2, 3) else [], rw, rh,
                confuser_threshold=thr)
            self.rgb_temporal.last_dets = list(rgb_dets)
            self.rgb_temporal.last_dets_sources = list(rgb_src)
            self.rgb_temporal.last_troi_rois = list(rgb_troi)
            self.ir_temporal.last_dets = list(ir_dets)
            self.ir_temporal.last_dets_sources = list(ir_src)
            self.ir_temporal.last_troi_rois = list(ir_troi)
            self.rgb_temporal.last_trust = trust
            self.rgb_temporal.last_trust_prob = trust_prob

            lines += self._build_overlay_lines(self.rgb_temporal, s, prefix="RGB ")
            lines += self._build_overlay_lines(self.ir_temporal, s, prefix="IR  ")

        overlay_text_big(combined, lines)
        return combined, trust, trust_prob

    def _process_grayscale(self, frame_rgb, s, device, imgsz):
        temporal_on = self.temporal_var.get()
        rgb_t = self.rgb_temporal if temporal_on else None
        ir_t = self.ir_temporal if temporal_on else None

        rgb_dets, rgb_src, rgb_troi = self._run_with_roi_recovery(
            self.rgb_model, frame_rgb, s["rgb_conf"], device, imgsz, rgb_t)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])
        # Shared IR confidence (same as paired mode), user-configurable.
        ir_conf_grayscale = float(s.get("ir_conf_real", 0.40))
        ir_dets, ir_src, ir_troi = self._run_with_roi_recovery(
            self.ir_model, gray_3ch, ir_conf_grayscale, device, imgsz, ir_t)

        feat = extract_fusion_features(gray, gray, rgb_dets, ir_dets)
        X = np.array([[feat[f] for f in self.fusion_features]])
        trust = int(self.fusion_clf.predict(X)[0])
        probs = self.fusion_clf.predict_proba(X)[0]
        trust_prob = float(probs[trust])

        orig_trust = trust
        # Grayscale test mode: run IR filter on gray-replicate per user toggle.
        ir_filter_enabled = bool(s.get("grayscale_run_ir_filter", True))
        trust, rgp, irp, _veto = self._apply_patch_veto(
            trust, rgb_dets, ir_dets, frame_rgb, gray_3ch,
            ir_is_real_thermal=False, s=s,
            ir_verifier_enabled=ir_filter_enabled)
        verif_lines = self._build_verifier_lines(
            orig_trust, trust, rgp, irp,
            ir_skipped_gray=not ir_filter_enabled, n_rgb_dets=len(rgb_dets),
            n_ir_dets=len(ir_dets), s=s)

        rgb_trusted = [trust in (1, 3)] * len(rgb_dets)
        ir_trusted = [trust in (2, 3)] * len(ir_dets)
        show_tags = s.get("show_source_tags", False)

        left = frame_rgb.copy()
        right = gray_3ch.copy()
        draw_detections(left, rgb_dets, (0, 255, 0), "RGB ", rgb_trusted, rgb_src, show_tags)
        draw_detections(right, ir_dets, (255, 200, 0), "IR ", ir_trusted, ir_src, show_tags)

        if temporal_on and self.rgb_temporal:
            draw_temporal_overlays(left, self.rgb_temporal, s)
        if temporal_on and self.ir_temporal:
            draw_temporal_overlays(right, self.ir_temporal, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))

        combined = np.hstack([left, right])

        trust_label = TRUST_LABELS[trust]
        lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%) [grayscale]"]
        lines += verif_lines

        if temporal_on and self.rgb_temporal and self.ir_temporal:
            # Shared confuser feed (grayscale = same physical scene by design).
            rgb_max_p = float(max(rgp)) if rgp else None
            ir_max_p  = float(max(irp)) if irp else None
            shared_p = (max(p for p in (rgb_max_p, ir_max_p) if p is not None)
                        if (rgb_max_p is not None or ir_max_p is not None)
                        else None)
            self.rgb_temporal.add_confuser_prob(shared_p)
            self.ir_temporal.add_confuser_prob(shared_p)
            thr = float(s.get("patch_threshold", 0.70)) \
                  if s.get("use_patch_verifier", True) else None
            self.rgb_temporal.update(
                rgb_dets if trust in (1, 3) else [], lw, lh,
                confuser_threshold=thr)
            self.ir_temporal.update(
                ir_dets if trust in (2, 3) else [], rw, rh,
                confuser_threshold=thr)
            self.rgb_temporal.last_dets = list(rgb_dets)
            self.rgb_temporal.last_dets_sources = list(rgb_src)
            self.rgb_temporal.last_troi_rois = list(rgb_troi)
            self.ir_temporal.last_dets = list(ir_dets)
            self.ir_temporal.last_dets_sources = list(ir_src)
            self.ir_temporal.last_troi_rois = list(ir_troi)
            self.rgb_temporal.last_trust = trust
            self.rgb_temporal.last_trust_prob = trust_prob

            lines += self._build_overlay_lines(self.rgb_temporal, s, prefix="RGB ")
            lines += self._build_overlay_lines(self.ir_temporal, s, prefix="IR  ")

        overlay_text_big(combined, lines)
        return combined, trust, trust_prob

    # ── HOLD-FRAME RENDERERS (draw cached state on fresh video) ─────

    def _hold_single(self, frame):
        vis = frame.copy()
        s = self.settings
        if self.rgb_temporal and self.rgb_temporal.last_dets:
            draw_detections(vis, self.rgb_temporal.last_dets, (0, 255, 255),
                            sources=self.rgb_temporal.last_dets_sources,
                            show_source_tags=s.get("show_source_tags", False))
            draw_temporal_overlays(vis, self.rgb_temporal, s)
            lines = self._build_overlay_lines(self.rgb_temporal, s)
            overlay_text_big(vis, lines)
        return vis

    def _hold_paired(self, frame_rgb, frame_ir):
        s = self.settings
        show_tags = s.get("show_source_tags", False)
        left = frame_rgb.copy()
        right = frame_ir.copy() if frame_ir is not None else frame_rgb.copy()
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        trust = self.rgb_temporal.last_trust if self.rgb_temporal else self._cached_trust
        rgb_trusted = [trust in (1, 3)] * len(self.rgb_temporal.last_dets) if self.rgb_temporal else []
        ir_trusted = [trust in (2, 3)] * len(self.ir_temporal.last_dets) if self.ir_temporal else []

        if self.rgb_temporal and self.rgb_temporal.last_dets:
            draw_detections(left, self.rgb_temporal.last_dets, (0, 255, 0), "RGB ", rgb_trusted,
                            self.rgb_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(left, self.rgb_temporal, s)
        if self.ir_temporal and self.ir_temporal.last_dets:
            draw_detections(right, self.ir_temporal.last_dets, (255, 200, 0), "IR ", ir_trusted,
                            self.ir_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(right, self.ir_temporal, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))

        combined = np.hstack([left, right])

        if self.rgb_temporal and self.ir_temporal:
            trust_label = TRUST_LABELS.get(trust, "?")
            trust_prob = self.rgb_temporal.last_trust_prob or 0.0
            lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%)"]
            lines += self._build_overlay_lines(self.rgb_temporal, self.settings, prefix="RGB ")
            lines += self._build_overlay_lines(self.ir_temporal, self.settings, prefix="IR  ")
            overlay_text_big(combined, lines)

        return combined

    def _hold_grayscale(self, frame_rgb):
        s = self.settings
        show_tags = s.get("show_source_tags", False)
        left = frame_rgb.copy()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        right = cv2.merge([gray, gray, gray])

        trust = self.rgb_temporal.last_trust if self.rgb_temporal else self._cached_trust
        rgb_trusted = [trust in (1, 3)] * len(self.rgb_temporal.last_dets) if self.rgb_temporal else []
        ir_trusted = [trust in (2, 3)] * len(self.ir_temporal.last_dets) if self.ir_temporal else []

        if self.rgb_temporal and self.rgb_temporal.last_dets:
            draw_detections(left, self.rgb_temporal.last_dets, (0, 255, 0), "RGB ", rgb_trusted,
                            self.rgb_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(left, self.rgb_temporal, s)
        if self.ir_temporal and self.ir_temporal.last_dets:
            draw_detections(right, self.ir_temporal.last_dets, (255, 200, 0), "IR ", ir_trusted,
                            self.ir_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(right, self.ir_temporal, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))

        combined = np.hstack([left, right])

        if self.rgb_temporal and self.ir_temporal:
            trust_label = TRUST_LABELS.get(trust, "?")
            trust_prob = self.rgb_temporal.last_trust_prob or 0.0
            lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%) [grayscale]"]
            lines += self._build_overlay_lines(self.rgb_temporal, self.settings, prefix="RGB ")
            lines += self._build_overlay_lines(self.ir_temporal, self.settings, prefix="IR  ")
            overlay_text_big(combined, lines)

        return combined

    # ── DISPLAY UPDATE ──────────────────────────────────────────────

    def _update_display(self, annotated, trust, trust_prob, mode):
        if mode == "Single Model":
            self._show_on_canvas(self.canvas_left, annotated, "left")
        else:
            h, w = annotated.shape[:2]
            mid = w // 2
            left_img = annotated[:, :mid]
            right_img = annotated[:, mid:]

            self._show_on_canvas(self.canvas_left, left_img, "left")
            self._show_on_canvas(self.canvas_right, right_img, "right")

            if trust is not None:
                label_text = TRUST_LABELS[trust]
                color = TRUST_COLORS_HEX[trust]

                if self.temporal_var.get():
                    warn = (self.rgb_temporal and self.rgb_temporal.warning_active) or \
                           (self.ir_temporal and self.ir_temporal.warning_active)
                    alert = (self.rgb_temporal and self.rgb_temporal.alert_active) or \
                            (self.ir_temporal and self.ir_temporal.alert_active)
                    extra = ""
                    if alert:
                        extra = "  ALERT"
                        color = "#ff3333"
                    elif warn:
                        extra = "  WARNING"
                        color = "#ffaa33"
                    self.fusion_label.config(
                        text=f"FUSION: {label_text}{extra}", foreground=color)
                else:
                    self.fusion_label.config(text=f"FUSION: {label_text}", foreground=color)
                self.fusion_conf.config(
                    text=f"({trust_prob*100:.1f}%)" if trust_prob else "")

        # Stats
        self.stats_label.config(
            text=f"FPS: {self.fps:.1f}  |  Frame: {self.frame_num}/{self.total_frames}")
        self.progress["value"] = self.frame_num
        pct = self.frame_num / max(self.total_frames, 1) * 100
        self.progress_label.config(text=f"{pct:.1f}%")

    def _show_on_canvas(self, canvas, frame, side):
        """Update canvas image in-place (no delete+create = no flicker)."""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)

        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2

        if side == "left":
            self._photo_left = photo
            if self._img_id_left is not None:
                try:
                    canvas.itemconfigure(self._img_id_left, image=photo)
                    canvas.coords(self._img_id_left, x_off, y_off)
                except tk.TclError:
                    self._img_id_left = canvas.create_image(x_off, y_off, anchor="nw", image=photo)
            else:
                self._img_id_left = canvas.create_image(x_off, y_off, anchor="nw", image=photo)
        else:
            self._photo_right = photo
            if self._img_id_right is not None:
                try:
                    canvas.itemconfigure(self._img_id_right, image=photo)
                    canvas.coords(self._img_id_right, x_off, y_off)
                except tk.TclError:
                    self._img_id_right = canvas.create_image(x_off, y_off, anchor="nw", image=photo)
            else:
                self._img_id_right = canvas.create_image(x_off, y_off, anchor="nw", image=photo)

    def _on_video_end(self):
        self._cleanup()
        self.btn_play.config(text="Play")
        self.btn_pause.config(text="Pause", state="disabled")
        self.btn_stop.config(state="disabled")
        sp = self.save_path.get().strip()
        msg = f"Done. {self.frame_num} frames processed."
        if sp and os.path.exists(sp):
            msg += f"\nSaved to: {sp}"
        self.status.config(text=msg)
        messagebox.showinfo("Done", msg)


# ── ENTRY POINT ──────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    style = ttk.Style()
    for pref in ("clam", "vista", "xpnative", "alt"):
        if pref in style.theme_names():
            style.theme_use(pref)
            break
    FusionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
