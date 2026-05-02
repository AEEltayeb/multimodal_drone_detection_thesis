"""Processing engine — wraps fusion_app.py's inference logic."""
import base64
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Add workspace and ir_gui to path for imports
_WS = Path(__file__).resolve().parents[2]
_IR_GUI = Path(__file__).resolve().parents[1]
for _p in (_WS, _IR_GUI):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ultralytics import YOLO
import joblib

try:
    from classifier.patch_verifier import PatchVerifier
except Exception:
    PatchVerifier = None

from fusion.features import TARGET_NAMES, compute_global_features, compute_target_features
from fusion.temporal import (
    TemporalContinuity, PerModalityTemporalState,
    draw_box, draw_detections, draw_temporal_overlays,
    overlay_text_big, build_overlay_lines,
)

TRUST_LABELS = {0: "REJECT BOTH", 1: "TRUST RGB", 2: "TRUST IR", 3: "TRUST BOTH"}


def extract_fusion_features(rgb_gray, ir_gray, rgb_dets, ir_dets):
    """Build the 40-feature dict the classifier expects."""
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
    g_rgb = compute_global_features(rgb_gray)
    g_ir = compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in g_rgb.items()})
    feats.update({f"ir_{k}": v for k, v in g_ir.items()})
    for prefix, dets, gray, gw, gh in (
        ("rgb", rgb_dets, rgb_gray, rgb_w, rgb_h),
        ("ir", ir_dets, ir_gray, ir_w, ir_h),
    ):
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best = max(dets, key=lambda d: (d.conf if hasattr(d, "conf") else d[4]))
            bb = best.box if hasattr(best, "box") else (best[0], best[1], best[2], best[3])
            tf = compute_target_features(gray, bb, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    rd, id_ = len(rgb_dets) > 0, len(ir_dets) > 0
    feats["both_detect"] = int(rd and id_)
    feats["neither_detect"] = int(not rd and not id_)
    feats["rgb_only_detect"] = int(rd and not id_)
    feats["ir_only_detect"] = int(not rd and id_)
    return feats


class DetectionEngine:
    """Encapsulates YOLO + fusion + temporal logic."""

    def __init__(self):
        self.rgb_model = None
        self.ir_model = None
        self.fusion_clf = None
        self.fusion_features = None
        self.rgb_verifier = None
        self.ir_verifier = None
        self.rgb_temporal = None
        self.ir_temporal = None

        self.cap_left = None
        self.cap_right = None
        self.writer = None
        self.running = False
        self.paused = False
        self.lock = threading.Lock()

        self.frame_num = 0
        self.total_frames = 0
        self.fps = 0.0
        self.playback_speed = 1.0
        self._stride = 1

        # Cached state
        self._cached_trust = None
        self._cached_trust_prob = None
        self._cached_n_rgb = 0
        self._cached_n_ir = 0
        self._cached_filter_veto = False
        self._cached_rgb_max_p = 0.0
        self._cached_ir_max_p = 0.0

        # Callback: called with (base64_jpeg, stats_dict) on each frame
        self.on_frame = None

    def load_models(self, settings, mode, status_cb=None):
        s = settings
        def _status(msg):
            if status_cb: status_cb(msg)

        _status("Loading RGB model...")
        self.rgb_model = YOLO(s["rgb_model"])

        if mode != "Single Model":
            _status("Loading IR model...")
            self.ir_model = YOLO(s["ir_model"])
            _status("Loading fusion classifier...")
            bundle = joblib.load(s["fusion_model"])
            self.fusion_clf = bundle["model"]
            self.fusion_features = bundle["features"]

            self.rgb_verifier = self.ir_verifier = None
            if s.get("use_patch_verifier", True) and PatchVerifier:
                rw = s.get("rgb_patch_weights")
                iw = s.get("ir_patch_weights")
                if rw and os.path.isfile(rw):
                    _status("Loading RGB patch verifier...")
                    self.rgb_verifier = PatchVerifier(rw)
                if iw and os.path.isfile(iw):
                    _status("Loading IR patch verifier...")
                    self.ir_verifier = PatchVerifier(iw)
        _status("Models loaded")

    def start(self, left_path, right_path, mode, settings, temporal_on,
              save_path=None):
        self.cap_left = cv2.VideoCapture(left_path)
        if not self.cap_left.isOpened():
            raise ValueError(f"Cannot open {left_path}")

        if mode == "Paired Fusion":
            self.cap_right = cv2.VideoCapture(right_path)
            if not self.cap_right.isOpened():
                raise ValueError(f"Cannot open {right_path}")
        else:
            self.cap_right = None

        self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self._cached_trust = None
        self._cached_trust_prob = None

        s = settings
        fps_src = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
        self._stride = max(1, int(round(fps_src / s["infer_fps"]))) if temporal_on else 1
        infer_fps = s["infer_fps"]

        def _mk_temporal():
            return PerModalityTemporalState(
                stride=self._stride,
                warning_window=s["warning_window_frames"],
                warning_require=s["warning_require_hits"],
                alert_window=s["alert_window_frames"],
                alert_require=s["alert_require_hits"],
                alert_avg_conf_thresh=s["alert_avg_conf_threshold"],
                warning_cooldown_frames=int(round(s["warning_cooldown_s"] * infer_fps)),
                alert_cooldown_frames=int(round(s["alert_cooldown_s"] * infer_fps)),
                roi_ttl=s["roi_ttl"], roi_expand=s["roi_expand"],
            )

        if temporal_on:
            self.rgb_temporal = _mk_temporal()
            self.ir_temporal = _mk_temporal()
        else:
            self.rgb_temporal = self.ir_temporal = None

        # Video writer
        self.writer = None
        if save_path:
            w = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w = w * 2 if mode != "Single Model" else w
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            self.writer = cv2.VideoWriter(save_path, fourcc, fps_src, (out_w, h))

        self.running = True
        self.paused = False
        threading.Thread(target=self._loop, args=(mode, settings, fps_src),
                         daemon=True).start()

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused

    def skip_forward(self, seconds=30):
        if not self.cap_left: return
        with self.lock:
            fps = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
            target = min(int(self.cap_left.get(cv2.CAP_PROP_POS_FRAMES)) + int(fps * seconds),
                         self.total_frames - 1)
            self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, target)
            if self.cap_right and self.cap_right.isOpened():
                self.cap_right.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.frame_num = target

    def _loop(self, mode, s, fps_src):
        device = s["gpu_device"]
        imgsz = s["imgsz"]
        frame_period = 1.0 / fps_src

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            with self.lock:
                if not self.cap_left or not self.cap_left.isOpened():
                    break
                ret, frame_l = self.cap_left.read()
                if not ret: break
                frame_r = None
                if self.cap_right:
                    ret_r, frame_r = self.cap_right.read()
                    if not ret_r: break

            self.frame_num += 1
            is_infer = (self.frame_num % self._stride == 0) or self._stride == 1
            t0 = time.perf_counter()

            if mode == "Single Model":
                annotated = self._process_single(frame_l, s, device, imgsz, is_infer)
                trust = None
            elif mode == "Paired Fusion":
                annotated, trust = self._process_paired(frame_l, frame_r, s, device, imgsz, is_infer)
            else:
                annotated, trust = self._process_grayscale(frame_l, s, device, imgsz, is_infer)

            elapsed = time.perf_counter() - t0
            self.fps = 1.0 / max(elapsed, 1e-6)

            if self.writer:
                self.writer.write(annotated)

            # Encode to JPEG bytes (Flet 0.84 Image.src accepts bytes)
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buf.tobytes()

            if self.on_frame:
                self.on_frame(jpeg_bytes, {
                    "frame": self.frame_num, "total": self.total_frames,
                    "fps": round(self.fps, 1), "trust": trust,
                    "trust_label": TRUST_LABELS.get(trust, ""),
                    "trust_prob": self._cached_trust_prob,
                    "warning": self.rgb_temporal.warning_active if self.rgb_temporal else False,
                    "alert": (self.rgb_temporal.alert_active if self.rgb_temporal else False) or
                             (self.ir_temporal.alert_active if self.ir_temporal else False),
                    "n_rgb": self._cached_n_rgb,
                    "n_ir": self._cached_n_ir,
                    "filter_veto": self._cached_filter_veto,
                    "rgb_max_p": self._cached_rgb_max_p,
                    "ir_max_p": self._cached_ir_max_p,
                    "w_events": (self.rgb_temporal.count_warning_events if self.rgb_temporal else 0),
                    "a_events": ((self.rgb_temporal.count_alert_events if self.rgb_temporal else 0) +
                                 (self.ir_temporal.count_alert_events if self.ir_temporal else 0)),
                })

            # Pace
            speed = self.playback_speed
            if speed > 0:
                wait = frame_period / speed - (time.perf_counter() - t0)
                if wait > 0: time.sleep(wait)

        self.running = False
        self._cleanup()
        if self.on_frame:
            self.on_frame(None, {"done": True})

    def _cleanup(self):
        with self.lock:
            for cap in (self.cap_left, self.cap_right):
                if cap and cap.isOpened(): cap.release()
            self.cap_left = self.cap_right = None
            if self.writer: self.writer.release(); self.writer = None

    # ── YOLO helpers ────────────────────────────────────────
    def _run_yolo(self, model, frame, conf, s, device, imgsz):
        results = model.predict(frame, conf=conf, iou=s["nms_iou"],
                                imgsz=imgsz, max_det=20, verbose=False, device=device)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                c = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, c])
        return dets

    def _run_with_roi(self, model, frame, conf, s, device, imgsz, temporal):
        dets = self._run_yolo(model, frame, conf, s, device, imgsz)
        sources = ["full"] * len(dets)
        troi_rois = []
        if temporal and not dets and temporal.last_roi is not None and temporal.roi_age > 0:
            h, w = frame.shape[:2]
            roi_result = temporal.get_roi_crop(frame, w, h)
            if roi_result:
                crop, (ox, oy) = roi_result
                troi_rois.append((ox, oy, ox + crop.shape[1], oy + crop.shape[0]))
                crop_dets = self._run_yolo(model, crop, conf * 0.8, s, device, imgsz)
                if crop_dets:
                    dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                    sources = ["troi"] * len(dets)
        return dets, sources, troi_rois

    # ── Patch veto (same logic as fusion_app.py) ────────────
    def _apply_patch_veto(self, trust, rgb_dets, ir_dets, rgb_bgr, ir_bgr,
                          ir_is_real, s, ir_verifier_enabled=None):
        if not s.get("use_patch_verifier", True) or trust == 0:
            return trust, [], [], False
        thr = float(s.get("patch_threshold", 0.70))
        trust_rgb = trust in (1, 3)
        trust_ir = trust in (2, 3)

        def _is_gray(img, max_diff=5):
            if img is None or img.ndim != 3: return True
            step = max(1, min(img.shape[:2]) // 64)
            samp = img[::step, ::step].astype(np.int16)
            return (int(np.abs(samp[...,0]-samp[...,1]).max()) <= max_diff and
                    int(np.abs(samp[...,1]-samp[...,2]).max()) <= max_diff)

        rgb_color = not _is_gray(rgb_bgr)
        ir_active = bool(ir_verifier_enabled) if ir_verifier_enabled is not None else bool(ir_is_real)
        rgp, irp = [], []
        if trust_rgb and self.rgb_verifier and rgb_color and rgb_dets:
            boxes = [d.box if hasattr(d, 'box') else d[:4] for d in rgb_dets]
            rgp = self.rgb_verifier.predict_boxes(rgb_bgr, boxes).tolist()
        if trust_ir and self.ir_verifier and ir_active and ir_dets:
            boxes = [d.box if hasattr(d, 'box') else d[:4] for d in ir_dets]
            irp = self.ir_verifier.predict_boxes(ir_bgr, boxes).tolist()
        rgb_ok = not (trust_rgb and self.rgb_verifier and rgb_color and rgp and max(rgp) >= thr)
        ir_ok = not (trust_ir and self.ir_verifier and ir_active and irp and max(irp) >= thr)
        nr, ni = trust_rgb and rgb_ok, trust_ir and ir_ok
        new = 3 if nr and ni else 1 if nr else 2 if ni else 0
        return new, rgp, irp, new != trust

    # ── Single mode ─────────────────────────────────────────
    def _process_single(self, frame, s, device, imgsz, is_infer):
        temporal = self.rgb_temporal
        if is_infer:
            dets, src, troi = self._run_with_roi(self.rgb_model, frame, s["rgb_conf"],
                                                  s, device, imgsz, temporal)
            self._cached_n_rgb = len(dets)
            self._cached_n_ir = 0
            self._cached_filter_veto = False
            self._cached_rgb_max_p = 0.0
            self._cached_ir_max_p = 0.0
            vis = frame.copy()
            draw_detections(vis, dets, (0, 255, 255), sources=src,
                            show_source_tags=s.get("show_source_tags", False))
            if temporal:
                h, w = frame.shape[:2]
                temporal.update(dets, w, h)
                temporal.last_dets = list(dets)
                temporal.last_dets_sources = list(src)
                temporal.last_troi_rois = list(troi)
                draw_temporal_overlays(vis, temporal, s)
                overlay_text_big(vis, build_overlay_lines(temporal, s))
            return vis
        else:
            vis = frame.copy()
            if temporal and temporal.last_dets:
                draw_detections(vis, temporal.last_dets, (0, 255, 255),
                                sources=temporal.last_dets_sources,
                                show_source_tags=s.get("show_source_tags", False))
                draw_temporal_overlays(vis, temporal, s)
                overlay_text_big(vis, build_overlay_lines(temporal, s))
            return vis

    # ── Paired mode ─────────────────────────────────────────
    def _process_paired(self, frame_rgb, frame_ir, s, device, imgsz, is_infer):
        if not is_infer:
            return self._hold_dual(frame_rgb, frame_ir, s), self._cached_trust

        rgb_t, ir_t = self.rgb_temporal, self.ir_temporal
        rgb_dets, rgb_src, rgb_troi = self._run_with_roi(
            self.rgb_model, frame_rgb, s["rgb_conf"], s, device, imgsz, rgb_t)
        ir_dets, ir_src, ir_troi = self._run_with_roi(
            self.ir_model, frame_ir, s["ir_conf_real"], s, device, imgsz, ir_t)

        rgb_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY) if len(frame_ir.shape) == 3 else frame_ir
        feat = extract_fusion_features(rgb_gray, ir_gray, rgb_dets, ir_dets)
        X = np.array([[feat[f] for f in self.fusion_features]])
        trust = int(self.fusion_clf.predict(X)[0])
        probs = self.fusion_clf.predict_proba(X)[0]
        trust_prob = float(probs[trust])

        ir_bgr = frame_ir if len(frame_ir.shape) == 3 else cv2.cvtColor(frame_ir, cv2.COLOR_GRAY2BGR)
        orig_trust = trust
        trust, rgp, irp, _ = self._apply_patch_veto(
            trust, rgb_dets, ir_dets, frame_rgb, ir_bgr, True, s)

        self._cached_trust = trust
        self._cached_trust_prob = trust_prob
        self._cached_n_rgb = len(rgb_dets)
        self._cached_n_ir = len(ir_dets)
        self._cached_filter_veto = (trust != orig_trust)
        self._cached_rgb_max_p = round(max(rgp), 3) if rgp else 0.0
        self._cached_ir_max_p = round(max(irp), 3) if irp else 0.0

        show_tags = s.get("show_source_tags", False)
        left = frame_rgb.copy()
        right = frame_ir.copy()
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        draw_detections(left, rgb_dets, (0, 255, 0), "RGB ",
                        [trust in (1,3)]*len(rgb_dets), rgb_src, show_tags)
        draw_detections(right, ir_dets, (255, 200, 0), "IR ",
                        [trust in (2,3)]*len(ir_dets), ir_src, show_tags)

        if rgb_t: draw_temporal_overlays(left, rgb_t, s)
        if ir_t: draw_temporal_overlays(right, ir_t, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))

        combined = np.hstack([left, right])
        lines = [f"FUSION: {TRUST_LABELS[trust]} ({trust_prob*100:.1f}%)"]

        if rgb_t and ir_t:
            thr = float(s.get("patch_threshold", 0.70)) if s.get("use_patch_verifier") else None
            rgb_t.update(rgb_dets if trust in (1,3) else [], lw, lh, confuser_threshold=thr)
            ir_t.update(ir_dets if trust in (2,3) else [], rw, rh, confuser_threshold=thr)
            rgb_t.last_dets, rgb_t.last_dets_sources = list(rgb_dets), list(rgb_src)
            ir_t.last_dets, ir_t.last_dets_sources = list(ir_dets), list(ir_src)
            rgb_t.last_troi_rois, ir_t.last_troi_rois = list(rgb_troi), list(ir_troi)
            rgb_t.last_trust, rgb_t.last_trust_prob = trust, trust_prob
            lines += build_overlay_lines(rgb_t, s, prefix="RGB ")
            lines += build_overlay_lines(ir_t, s, prefix="IR  ")

        overlay_text_big(combined, lines)
        return combined, trust

    # ── Grayscale mode ──────────────────────────────────────
    def _process_grayscale(self, frame_rgb, s, device, imgsz, is_infer):
        if not is_infer:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.merge([gray, gray, gray])
            return self._hold_dual(frame_rgb, gray3, s), self._cached_trust

        rgb_t, ir_t = self.rgb_temporal, self.ir_temporal
        rgb_dets, rgb_src, rgb_troi = self._run_with_roi(
            self.rgb_model, frame_rgb, s["rgb_conf"], s, device, imgsz, rgb_t)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.merge([gray, gray, gray])
        ir_dets, ir_src, ir_troi = self._run_with_roi(
            self.ir_model, gray3, float(s.get("ir_conf_real", 0.40)),
            s, device, imgsz, ir_t)

        feat = extract_fusion_features(gray, gray, rgb_dets, ir_dets)
        X = np.array([[feat[f] for f in self.fusion_features]])
        trust = int(self.fusion_clf.predict(X)[0])
        probs = self.fusion_clf.predict_proba(X)[0]
        trust_prob = float(probs[trust])

        ir_filter_en = bool(s.get("grayscale_run_ir_filter", True))
        orig_trust = trust
        trust, rgp, irp, _ = self._apply_patch_veto(
            trust, rgb_dets, ir_dets, frame_rgb, gray3, False, s,
            ir_verifier_enabled=ir_filter_en)

        self._cached_trust = trust
        self._cached_trust_prob = trust_prob
        self._cached_n_rgb = len(rgb_dets)
        self._cached_n_ir = len(ir_dets)
        self._cached_filter_veto = (trust != orig_trust)
        self._cached_rgb_max_p = round(max(rgp), 3) if rgp else 0.0
        self._cached_ir_max_p = round(max(irp), 3) if irp else 0.0

        show_tags = s.get("show_source_tags", False)
        left = frame_rgb.copy()
        right = gray3.copy()
        draw_detections(left, rgb_dets, (0, 255, 0), "RGB ",
                        [trust in (1,3)]*len(rgb_dets), rgb_src, show_tags)
        draw_detections(right, ir_dets, (255, 200, 0), "IR ",
                        [trust in (2,3)]*len(ir_dets), ir_src, show_tags)

        if rgb_t: draw_temporal_overlays(left, rgb_t, s)
        if ir_t: draw_temporal_overlays(right, ir_t, s)

        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))
        combined = np.hstack([left, right])

        lines = [f"FUSION: {TRUST_LABELS[trust]} ({trust_prob*100:.1f}%) [grayscale]"]
        if rgb_t and ir_t:
            thr = float(s.get("patch_threshold", 0.70)) if s.get("use_patch_verifier") else None
            rgb_t.update(rgb_dets if trust in (1,3) else [], lw, lh, confuser_threshold=thr)
            ir_t.update(ir_dets if trust in (2,3) else [], rw, rh, confuser_threshold=thr)
            rgb_t.last_dets, rgb_t.last_dets_sources = list(rgb_dets), list(rgb_src)
            ir_t.last_dets, ir_t.last_dets_sources = list(ir_dets), list(ir_src)
            rgb_t.last_troi_rois, ir_t.last_troi_rois = list(rgb_troi), list(ir_troi)
            rgb_t.last_trust, rgb_t.last_trust_prob = trust, trust_prob
            lines += build_overlay_lines(rgb_t, s, prefix="RGB ")
            lines += build_overlay_lines(ir_t, s, prefix="IR  ")
        overlay_text_big(combined, lines)
        return combined, trust

    # ── Hold frame renderer ─────────────────────────────────
    def _hold_dual(self, frame_l, frame_r, s):
        show_tags = s.get("show_source_tags", False)
        left = frame_l.copy()
        right = frame_r.copy()
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        trust = self._cached_trust
        if self.rgb_temporal and self.rgb_temporal.last_dets:
            draw_detections(left, self.rgb_temporal.last_dets, (0, 255, 0), "RGB ",
                            [trust in (1,3)]*len(self.rgb_temporal.last_dets),
                            self.rgb_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(left, self.rgb_temporal, s)
        if self.ir_temporal and self.ir_temporal.last_dets:
            draw_detections(right, self.ir_temporal.last_dets, (255, 200, 0), "IR ",
                            [trust in (2,3)]*len(self.ir_temporal.last_dets),
                            self.ir_temporal.last_dets_sources, show_tags)
            draw_temporal_overlays(right, self.ir_temporal, s)
        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        if rh != lh:
            right = cv2.resize(right, (int(rw * lh / rh), lh))
        combined = np.hstack([left, right])
        if self.rgb_temporal and self.ir_temporal:
            tp = self.rgb_temporal.last_trust_prob or 0.0
            lines = [f"FUSION: {TRUST_LABELS.get(trust, '?')} ({tp*100:.1f}%)"]
            lines += build_overlay_lines(self.rgb_temporal, s, prefix="RGB ")
            lines += build_overlay_lines(self.ir_temporal, s, prefix="IR  ")
            overlay_text_big(combined, lines)
        return combined
