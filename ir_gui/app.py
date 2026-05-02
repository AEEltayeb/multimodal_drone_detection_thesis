"""
IR Drone Detection GUI
Run a YOLOv8 model on a video file and visualize detections in real time.

Usage:
    python ir_gui/app.py

Controls:
    - Browse for video and model weights
    - Play / Pause / Stop
    - Adjust confidence threshold with slider
    - Annotated video is saved to disk when you press Stop or the video ends
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import os
from PIL import Image, ImageTk
from ultralytics import YOLO


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IR Drone Detection")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # State
        self.model = None
        self.cap = None
        self.running = False
        self.paused = False
        self.writer = None
        self.lock = threading.Lock()

        # Stats
        self.fps = 0.0
        self.frame_num = 0
        self.total_frames = 0
        self.detection_count = 0

        # Paths
        self.video_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.4)
        self.save_path = tk.StringVar()

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar: file inputs
        top = ttk.LabelFrame(self.root, text="Inputs", padding=8)
        top.pack(fill="x", padx=8, pady=(8, 4))

        # Video row
        ttk.Label(top, text="Video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.video_path, width=60).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Browse", command=self._browse_video).grid(row=0, column=2)

        # Model row
        ttk.Label(top, text="Model:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.model_path, width=60).grid(row=1, column=1, padx=4, pady=(4, 0))
        ttk.Button(top, text="Browse", command=self._browse_model).grid(row=1, column=2, pady=(4, 0))

        # Save row
        ttk.Label(top, text="Save to:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.save_path, width=60).grid(row=2, column=1, padx=4, pady=(4, 0))
        ttk.Button(top, text="Browse", command=self._browse_save).grid(row=2, column=2, pady=(4, 0))

        top.columnconfigure(1, weight=1)

        # Controls bar
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill="x", padx=8)

        self.btn_play = ttk.Button(ctrl, text="▶ Play", command=self._play)
        self.btn_play.pack(side="left", padx=2)

        self.btn_pause = ttk.Button(ctrl, text="⏸ Pause", command=self._pause, state="disabled")
        self.btn_pause.pack(side="left", padx=2)

        self.btn_stop = ttk.Button(ctrl, text="⏹ Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=2)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(ctrl, text="Confidence:").pack(side="left")
        self.conf_slider = ttk.Scale(ctrl, from_=0.05, to=0.95, variable=self.conf_threshold,
                                      orient="horizontal", length=150)
        self.conf_slider.pack(side="left", padx=4)
        self.conf_label = ttk.Label(ctrl, text="0.40")
        self.conf_label.pack(side="left")
        self.conf_threshold.trace_add("write", self._update_conf_label)

        # Stats on the right
        self.stats_label = ttk.Label(ctrl, text="FPS: --  |  Frame: 0/0  |  Detections: 0")
        self.stats_label.pack(side="right")

        # Video canvas
        self.canvas = tk.Canvas(self.root, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        # Progress bar
        prog_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        prog_frame.pack(fill="x")
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(prog_frame, text="")
        self.progress_label.pack(anchor="e")

        # Status bar
        self.status = ttk.Label(self.root, text="Ready. Load a video and model to start.", relief="sunken",
                                anchor="w", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")

        # Keep a reference to the photo so it doesn't get garbage collected
        self._photo = None

    # ── Browse callbacks ────────────────────────────────────────────

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.ts"),
                       ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            # Auto-fill save path
            base, ext = os.path.splitext(path)
            self.save_path.set(base + "_detected" + ext)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def _browse_save(self):
        path = filedialog.asksaveasfilename(
            title="Save annotated video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")]
        )
        if path:
            self.save_path.set(path)

    def _update_conf_label(self, *_):
        self.conf_label.config(text=f"{self.conf_threshold.get():.2f}")

    # ── Playback controls ──────────────────────────────────────────

    def _play(self):
        if self.paused:
            # Resume
            self.paused = False
            self.btn_pause.config(text="⏸ Pause")
            self.status.config(text="Resumed.")
            return

        vp = self.video_path.get().strip()
        mp = self.model_path.get().strip()

        if not vp or not os.path.isfile(vp):
            messagebox.showerror("Error", "Select a valid video file.")
            return
        if not mp or not os.path.isfile(mp):
            messagebox.showerror("Error", "Select a valid model weights file.")
            return

        # Load model
        self.status.config(text="Loading model...")
        self.root.update()
        try:
            self.model = YOLO(mp)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.status.config(text="Model load failed.")
            return

        # Open video
        self.cap = cv2.VideoCapture(vp)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            self.status.config(text="Video open failed.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.detection_count = 0
        self.progress["maximum"] = max(self.total_frames, 1)
        self.progress["value"] = 0

        # Setup video writer
        sp = self.save_path.get().strip()
        if sp:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(sp) if os.path.dirname(sp) else ".", exist_ok=True)
            self.writer = cv2.VideoWriter(sp, fourcc, fps, (w, h))

        self.running = True
        self.paused = False

        self.btn_play.config(text="▶ Play")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.status.config(text=f"Playing: {os.path.basename(vp)}")

        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.btn_pause.config(text="▶ Resume" if self.paused else "⏸ Pause")
        self.status.config(text="Paused." if self.paused else "Resumed.")

    def _stop(self):
        self.running = False
        self.paused = False
        self._cleanup()
        self.btn_play.config(text="▶ Play")
        self.btn_pause.config(text="⏸ Pause", state="disabled")
        self.btn_stop.config(state="disabled")
        self.status.config(text="Stopped.")

    def _cleanup(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            if self.writer:
                self.writer.release()
                sp = self.save_path.get().strip()
                self.status.config(text=f"Saved: {sp}")
                self.writer = None

    # ── Processing loop (runs in thread) ───────────────────────────

    def _process_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()

            if not ret:
                # Video ended
                self.running = False
                self.root.after(0, self._on_video_end)
                break

            self.frame_num += 1
            t0 = time.perf_counter()

            # Run detection
            conf = self.conf_threshold.get()
            results = self.model.predict(frame, conf=conf, verbose=False)

            # Draw boxes
            annotated = frame.copy()
            det_count = 0
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0])
                    det_count += 1

                    # Color: green if conf > 0.6, yellow if > 0.4, red otherwise
                    if c > 0.6:
                        color = (0, 255, 0)
                    elif c > 0.4:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"drone {c:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            self.detection_count += det_count
            elapsed = time.perf_counter() - t0
            self.fps = 1.0 / max(elapsed, 1e-6)

            # Write to disk
            with self.lock:
                if self.writer:
                    self.writer.write(annotated)

            # Update UI (schedule on main thread)
            self.root.after(0, self._update_display, annotated, det_count)

        self.root.after(0, self._cleanup)

    def _update_display(self, frame, det_count):
        # Resize frame to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR -> RGB for Tkinter
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)

        # Center on canvas
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x_off, y_off, anchor="nw", image=self._photo)

        # Update stats
        self.stats_label.config(
            text=f"FPS: {self.fps:.1f}  |  Frame: {self.frame_num}/{self.total_frames}  |  Detections this frame: {det_count}"
        )
        self.progress["value"] = self.frame_num
        pct = (self.frame_num / max(self.total_frames, 1)) * 100
        self.progress_label.config(text=f"{pct:.1f}%")

    def _on_video_end(self):
        self.btn_play.config(text="▶ Play")
        self.btn_pause.config(text="⏸ Pause", state="disabled")
        self.btn_stop.config(state="disabled")
        sp = self.save_path.get().strip()
        msg = f"Video finished. {self.frame_num} frames processed, {self.detection_count} total detections."
        if sp:
            msg += f"\nSaved to: {sp}"
        self.status.config(text=msg)
        messagebox.showinfo("Done", msg)


def main():
    root = tk.Tk()
    # Use a nicer theme if available
    style = ttk.Style()
    available = style.theme_names()
    for pref in ("clam", "vista", "xpnative", "alt"):
        if pref in available:
            style.theme_use(pref)
            break
    DetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
