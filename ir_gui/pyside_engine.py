"""PySide6 detection engine — wraps fusion/engine.py with alert-gate temporal."""
import os, sys, threading, time
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_WS = Path(__file__).resolve().parents[1]
_IR = Path(__file__).resolve().parent
for _p in (_WS, _IR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fusion.engine import FusionEngine, Detection, TRUST_LABELS
from fusion.temporal import (
    PerModalityTemporalState, draw_detections, draw_temporal_overlays,
    overlay_text_big, build_overlay_lines,
)

def _wrap(dets):
    return [Detection(box=(d[0],d[1],d[2],d[3]), conf=d[4]) for d in dets]

def _build_suppressed_classes(s):
    """Build set of confuser class names enabled for suppression from settings."""
    cls = set()
    if s.get("suppress_helicopter", True): cls.add("helicopter")
    if s.get("suppress_airplane", True):   cls.add("airplane")
    if s.get("suppress_bird", True):       cls.add("bird")
    return cls if cls else None  # None = suppress nothing

def _run_yolo(model, frame, conf, engine):
    r = model.predict(frame, conf=conf, iou=engine.nms_iou,
                      imgsz=engine.imgsz, verbose=False, device=engine.device)[0]
    dets = []
    if r.boxes is not None:
        for i in range(len(r.boxes)):
            x1,y1,x2,y2 = r.boxes.xyxy[i].cpu().numpy()
            c = float(r.boxes.conf[i])
            dets.append([float(x1),float(y1),float(x2),float(y2),c])
    return dets

def _run_with_roi(model, frame, conf, engine, temporal):
    dets = _run_yolo(model, frame, conf, engine)
    sources = ["full"] * len(dets)
    troi = []
    if temporal and not dets and temporal.last_roi is not None and temporal.roi_age > 0:
        h, w = frame.shape[:2]
        roi_result = temporal.get_roi_crop(frame, w, h)
        if roi_result:
            crop, (ox, oy) = roi_result
            troi.append((ox, oy, ox+crop.shape[1], oy+crop.shape[0]))
            crop_dets = _run_yolo(model, crop, conf*0.8, engine)
            if crop_dets:
                dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                sources = ["troi"] * len(dets)
    return dets, sources, troi


class TalosEngine:
    """Detection engine using FusionEngine + alert-gate temporal (matches api.py)."""

    def __init__(self):
        self.fe = None  # FusionEngine
        self.rgb_temporal = None
        self.ir_temporal = None
        self.cap_left = self.cap_right = None
        self.writer = None
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.frame_num = 0
        self.total_frames = 0
        self._stride = 1
        self.on_frame = None  # callback(jpeg_bytes, stats_dict)
        # Det count cache (works even when temporal state is None)
        self._last_n_rgb = 0
        self._last_n_ir = 0
        self._last_verifier = None
        self._last_confuser_vetoed = False

    def load_engine(self, settings, mode, status_cb=None):
        """Load FusionEngine from settings dict."""
        def log(m):
            if status_cb: status_cb(m)
        log("Loading models...")
        s = settings
        # Auto-detect device: fall back to CPU if no CUDA GPU available
        device = s.get("gpu_device", 0)
        if isinstance(device, int):
            import torch
            if not torch.cuda.is_available():
                log("No CUDA GPU detected — falling back to CPU")
                device = "cpu"
        kwargs = dict(
            rgb_weights=s["rgb_model"], ir_weights=s.get("ir_model", s["rgb_model"]),
            fusion_model_path=s.get("fusion_model", ""),
            rgb_conf=float(s.get("rgb_conf", 0.25)),
            ir_conf=float(s.get("ir_conf_real", 0.40)),
            nms_iou=float(s.get("nms_iou", 0.45)),
            imgsz=int(s.get("imgsz", 640)),
            device=device,
            rgb_patch_weights=s.get("rgb_patch_weights", ""),
            ir_patch_weights=s.get("ir_patch_weights", ""),
            patch_threshold=float(s.get("patch_threshold", 0.70)),
            use_patch_verifier=bool(s.get("use_patch_verifier", True)),
            grayscale_run_ir_filter=bool(s.get("grayscale_run_ir_filter", True)),
            cascade_order=str(s.get("cascade_order", "filter_then_classifier")),
            feature_stride=int(s.get("feature_stride", 5)),
            feature_max_height=int(s.get("feature_max_height", 480)),
        )
        if mode == "Single Model":
            kwargs["ir_weights"] = kwargs["rgb_weights"]
            kwargs["use_patch_verifier"] = False
        self.fe = FusionEngine(**kwargs)
        log("Models loaded")

    def start(self, path, ir_path, mode, settings, temporal_on, save_path=None):
        self.stop()
        self.cap_left = cv2.VideoCapture(path)
        self.cap_right = cv2.VideoCapture(ir_path) if ir_path and mode == "Paired Fusion" else None
        fps = self.cap_left.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.cap_right:
            self.total_frames = min(self.total_frames, int(self.cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_num = 0
        s = settings
        infer_fps = int(s.get("infer_fps", 5))
        self._stride = max(1, int(round(fps / infer_fps)))

        # New stream → drop cached scene globals.
        if self.fe is not None and hasattr(self.fe, "feature_cache"):
            # Pick up any settings changes since load_engine.
            self.fe.feature_cache.configure(
                stride=int(s.get("feature_stride", 5)),
                max_h=int(s.get("feature_max_height", 480)))
            self.fe.feature_cache.reset()

        def _mk():
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
            self.rgb_temporal = _mk()
            self.ir_temporal = _mk()
        else:
            self.rgb_temporal = self.ir_temporal = None
            # No temporal state → no hold-frame cache; run inference every frame
            self._stride = 1

        self.writer = None
        if save_path:
            w = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w = w * 2 if mode != "Single Model" else w
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h))

        self.running = True
        self.paused = False
        threading.Thread(target=self._loop, args=(mode, settings, fps), daemon=True).start()

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

    def seek_to_frame(self, frame_num):
        """Seek both captures to the given frame number."""
        if not self.cap_left: return
        with self.lock:
            target = max(0, min(int(frame_num), self.total_frames - 1))
            self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, target)
            if self.cap_right and self.cap_right.isOpened():
                self.cap_right.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.frame_num = target

    def _loop(self, mode, s, fps_src):
        fe = self.fe
        show_tags = s.get("show_source_tags", True)
        ts = {"show_troi": True, "show_gate": True, "show_source_tags": show_tags,
              **{k: s[k] for k in s if k.startswith("warning") or k.startswith("alert") or k.startswith("roi")}}
        frame_period = 1.0 / fps_src
        last_ms = 0.0
        last_probs = [0,0,0,1]

        while self.running:
            if self.paused:
                time.sleep(0.05); continue

            t0 = time.perf_counter()

            with self.lock:
                if not self.cap_left or not self.cap_left.isOpened(): break
                ret, frame_l = self.cap_left.read()
                if not ret: break
                frame_r = None
                if self.cap_right:
                    ret_r, frame_r = self.cap_right.read()
                    if not ret_r: break

            self.frame_num += 1
            is_infer = (self.frame_num % self._stride == 0) or self._stride == 1
            rt, it = self.rgb_temporal, self.ir_temporal

            if is_infer:
                vis, trust, probs = self._infer(mode, frame_l, frame_r, s, fe, rt, it, ts)
                last_ms = (time.perf_counter() - t0) * 1000
                last_probs = probs
            else:
                vis = self._hold(mode, frame_l, frame_r, rt, it, ts)
                trust = rt.last_trust if rt and rt.last_trust is not None else 0
                probs = last_probs

            if self.writer: self.writer.write(vis)
            _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

            if self.on_frame:
                self.on_frame(buf.tobytes(), self._stats(mode, trust, probs, last_ms, rt, it, fe))

            # Realtime pacing: cap at frame_period per source frame. On slow
            # hardware (inference > frame_period) this never fires and we
            # run engine-bound; on fast hardware it caps playback at
            # realtime so it doesn't blur past.
            elapsed = time.perf_counter() - t0
            wait = frame_period - elapsed
            if wait > 0: time.sleep(wait)

        self.running = False
        self._cleanup()
        if self.on_frame: self.on_frame(None, {"done": True})

    def _infer(self, mode, frame_l, frame_r, s, fe, rt, it, ts):
        if mode == "Single Model":
            return self._infer_single(frame_l, s, fe, rt, ts)
        elif mode == "Paired Fusion":
            return self._infer_paired(frame_l, frame_r, s, fe, rt, it, ts)
        else:
            return self._infer_grayscale(frame_l, s, fe, rt, it, ts)

    def _infer_single(self, frame, s, fe, rt, ts):
        dets, src, troi = _run_with_roi(fe.rgb_model, frame, fe.rgb_conf, fe, rt)
        vis = frame.copy()
        draw_detections(vis, dets, (0,255,255), sources=src, show_source_tags=ts.get("show_source_tags",True))
        self._last_n_rgb = len(dets)
        self._last_n_ir = 0
        if rt:
            h, w = frame.shape[:2]
            rt.update(dets, w, h)
            rt.last_dets = list(dets); rt.last_dets_sources = list(src); rt.last_troi_rois = list(troi)
            draw_temporal_overlays(vis, rt, ts)
            overlay_text_big(vis, build_overlay_lines(rt, ts))
        return vis, 3 if dets else 0, [0,0,0,1] if dets else [1,0,0,0]

    def _infer_paired(self, rgb, ir, s, fe, rt, it, ts):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(2) as ex:
            f_rgb = ex.submit(_run_with_roi, fe.rgb_model, rgb, fe.rgb_conf, fe, rt)
            f_ir  = ex.submit(_run_with_roi, fe.ir_model,  ir,  fe.ir_conf, fe, it)
            rgb_dets, rgb_src, rgb_troi = f_rgb.result()
            ir_dets,  ir_src,  ir_troi  = f_ir.result()
        t_yolo = time.perf_counter()
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY) if len(ir.shape)==3 else ir
        feats = fe.extract_features(_wrap(rgb_dets), _wrap(ir_dets), rgb_gray, ir_gray)
        trust, probs = fe.classify(feats)
        t_clf = time.perf_counter()
        orig_trust = trust
        rgp, irp, _v = [], [], False
        use_history = bool(s.get("confuser_filter_history", False))
        if fe.use_patch_verifier and use_history:
            ir_bgr = ir if len(ir.shape)==3 else cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            trust, rgp, irp, _v = fe.patch_veto(trust, _wrap(rgb_dets), _wrap(ir_dets), rgb, ir_bgr,
                ir_is_real_thermal=True, suppressed_classes=_build_suppressed_classes(s))
        t_vrf = time.perf_counter()
        self._last_confuser_vetoed = (orig_trust != trust)
        trust_prob = float(probs[orig_trust])
        self._last_verifier = self._verifier_diag(fe, orig_trust, trust, rgp, irp, _v, rgb_dets, ir_dets, rgb, s)
        if rt: rt.last_verifier = self._last_verifier

        self._last_n_rgb = len(rgb_dets)
        self._last_n_ir = len(ir_dets)
        vis = self._compose_dual(rgb, ir, rgb_dets, ir_dets, rgb_src, ir_src, orig_trust, trust_prob,
                                  rt, it, rgb_troi, ir_troi, rgp, irp, ts, fe, s,
                                  tag="", ir_is_real_thermal=True)
        t_end = time.perf_counter()
        if self.frame_num % 50 == 0:
            print(f"[PAIRED f={self.frame_num}] YOLO={1000*(t_yolo-t0):.0f}ms  clf={1000*(t_clf-t_yolo):.0f}ms  vrf={1000*(t_vrf-t_clf):.0f}ms  draw={1000*(t_end-t_vrf):.0f}ms  TOTAL={1000*(t_end-t0):.0f}ms")
        return vis, orig_trust, probs

    def _infer_grayscale(self, rgb, s, fe, rt, it, ts):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.merge([gray, gray, gray])
        with ThreadPoolExecutor(2) as ex:
            f_rgb = ex.submit(_run_with_roi, fe.rgb_model, rgb,   fe.rgb_conf, fe, rt)
            f_ir  = ex.submit(_run_with_roi, fe.ir_model,  gray3, fe.ir_conf,  fe, it)
            rgb_dets, rgb_src, rgb_troi = f_rgb.result()
            ir_dets,  ir_src,  ir_troi  = f_ir.result()
        t_yolo = time.perf_counter()
        feats = fe.extract_features(_wrap(rgb_dets), _wrap(ir_dets), gray, gray)
        trust, probs = fe.classify(feats)
        t_clf = time.perf_counter()
        orig_trust = trust
        rgp, irp, _v = [], [], False
        use_history = bool(s.get("confuser_filter_history", False))
        if fe.use_patch_verifier and use_history:
            gs_run = bool(s.get("grayscale_run_ir_filter", True))
            trust, rgp, irp, _v = fe.patch_veto(trust, _wrap(rgb_dets), _wrap(ir_dets), rgb, gray3,
                ir_is_real_thermal=False, ir_verifier_enabled=gs_run, skip_ir_ood_gate=True,
                suppressed_classes=_build_suppressed_classes(s))
        t_vrf = time.perf_counter()
        self._last_confuser_vetoed = (orig_trust != trust)
        trust_prob = float(probs[orig_trust])
        self._last_verifier = self._verifier_diag(fe, orig_trust, trust, rgp, irp, _v, rgb_dets, ir_dets, rgb, s)
        if rt: rt.last_verifier = self._last_verifier

        self._last_n_rgb = len(rgb_dets)
        self._last_n_ir = len(ir_dets)
        vis = self._compose_dual(rgb, gray3, rgb_dets, ir_dets, rgb_src, ir_src, orig_trust, trust_prob,
                                  rt, it, rgb_troi, ir_troi, rgp, irp, ts, fe, s,
                                  tag=" [grayscale]", ir_is_real_thermal=False)
        t_end = time.perf_counter()
        if self.frame_num % 50 == 0:
            print(f"[GRAY f={self.frame_num}] YOLO={1000*(t_yolo-t0):.0f}ms  clf={1000*(t_clf-t_yolo):.0f}ms  vrf={1000*(t_vrf-t_clf):.0f}ms  draw={1000*(t_end-t_vrf):.0f}ms  TOTAL={1000*(t_end-t0):.0f}ms")
        return vis, orig_trust, probs

    def _compose_dual(self, rgb, ir_frame, rgb_dets, ir_dets, rgb_src, ir_src,
                       orig_trust, trust_prob, rt, it, rgb_troi, ir_troi, rgp, irp, ts, fe, s,
                       tag="", ir_is_real_thermal=True):
        show_tags = ts.get("show_source_tags", True)
        left = rgb.copy()
        right = ir_frame.copy() if len(ir_frame.shape)==3 else cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
        draw_detections(left, rgb_dets, (0,255,0), "RGB ", [orig_trust in (1,3)]*len(rgb_dets), rgb_src, show_tags)
        draw_detections(right, ir_dets, (255,200,0), "IR ", [orig_trust in (2,3)]*len(ir_dets), ir_src, show_tags)

        lh, lw = left.shape[:2]; rh, rw = right.shape[:2]
        patch_thr = float(s.get("patch_threshold", 0.70))
        use_history = bool(s.get("confuser_filter_history", False))

        if rt and it:
            if use_history:
                # ── HISTORY MODE: per-modality confuser accumulation ──
                rgb_mp = float(max(rgp)) if rgp else None
                ir_mp = float(max(irp)) if irp else None
                rt.add_confuser_prob(rgb_mp)
                it.add_confuser_prob(ir_mp)
                csm = s.get("confuser_suppress_mode", "primary_only")
                rt.update(rgb_dets if orig_trust in (1,3) else [], lw, lh,
                          confuser_threshold=patch_thr if fe.use_patch_verifier else None,
                          confuser_suppress_mode=csm)
                it.update(ir_dets if orig_trust in (2,3) else [], rw, rh,
                          confuser_threshold=patch_thr if fe.use_patch_verifier else None,
                          confuser_suppress_mode=csm)
            else:
                # ── ALERT-GATE MODE (default): no history, CNN only at alert ──
                rt.update(rgb_dets if orig_trust in (1,3) else [], lw, lh)
                it.update(ir_dets if orig_trust in (2,3) else [], rw, rh)

                # Run confuser filter ONLY when an alert is about to fire
                if fe.use_patch_verifier and (rt.alert_active or it.alert_active):
                    ir_bgr = ir_frame if len(ir_frame.shape)==3 else cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                    gs_run = bool(s.get("grayscale_run_ir_filter", True))
                    _, gate_rgp, gate_irp, _ = fe.patch_veto(
                        orig_trust, _wrap(rgb_dets), _wrap(ir_dets), rgb, ir_bgr,
                        ir_is_real_thermal=ir_is_real_thermal,
                        ir_verifier_enabled=gs_run if not ir_is_real_thermal else None,
                        skip_ir_ood_gate=not ir_is_real_thermal,
                        suppressed_classes=_build_suppressed_classes(s),
                    )
                    # Trust-aware veto: only check the verifier for trusted modality
                    rgb_confuser = bool(gate_rgp and max(gate_rgp) >= patch_thr)
                    ir_confuser = bool(gate_irp and max(gate_irp) >= patch_thr)

                    should_suppress = False
                    if orig_trust == 1:    # trust_rgb → only RGB verifier matters
                        should_suppress = rgb_confuser
                    elif orig_trust == 2:  # trust_ir → only IR verifier matters
                        should_suppress = ir_confuser
                    elif orig_trust == 3:  # trust_both → either veto wins
                        should_suppress = rgb_confuser or ir_confuser
                    # trust=0 (reject_both) → no alert possible, nothing to suppress

                    if should_suppress:
                        if rt.alert_active:
                            rt.alert_active = False
                            rt.confuser_suppressed = True
                            if rt.count_alert_events > 0:
                                rt.count_alert_events -= 1
                        if it.alert_active:
                            it.alert_active = False
                            it.confuser_suppressed = True
                            if it.count_alert_events > 0:
                                it.count_alert_events -= 1
                    # Update verifier diagnostics for UI
                    if rt:
                        rt.last_verifier = self._verifier_diag(
                            fe, orig_trust, orig_trust, gate_rgp, gate_irp, False,
                            rgb_dets, ir_dets, rgb, s)

            rt.last_dets = list(rgb_dets); rt.last_dets_sources = list(rgb_src); rt.last_troi_rois = list(rgb_troi)
            it.last_dets = list(ir_dets); it.last_dets_sources = list(ir_src); it.last_troi_rois = list(ir_troi)
            rt.last_trust = orig_trust; rt.last_trust_prob = trust_prob
            draw_temporal_overlays(left, rt, ts); draw_temporal_overlays(right, it, ts)

        if rh != lh: right = cv2.resize(right, (int(rw*lh/rh), lh))
        vis = np.hstack([left, right])
        lines = [f"FUSION: {TRUST_LABELS[orig_trust]} ({trust_prob*100:.1f}%){tag}"]
        suppressed = (rt.confuser_suppressed if rt else False) or (it.confuser_suppressed if it else False)
        if suppressed:
            parts = []
            if fe.rgb_verifier and hasattr(fe.rgb_verifier,'last_labels') and fe.rgb_verifier.last_labels:
                parts.append(f"RGB:{fe.rgb_verifier.last_labels[0]}")
            if fe.ir_verifier and hasattr(fe.ir_verifier,'last_labels') and fe.ir_verifier.last_labels:
                parts.append(f"IR:{fe.ir_verifier.last_labels[0]}")
            lines.append(f"CONFUSER GATE: alert suppressed ({' '.join(parts)})")
        else:
            lines.append("CONFUSER GATE: no alert suppressed")
        if rt and it:
            lines += build_overlay_lines(rt, ts, prefix="RGB ")
            lines += build_overlay_lines(it, ts, prefix="IR  ")
        overlay_text_big(vis, lines)
        return vis

    def _hold(self, mode, frame_l, frame_r, rt, it, ts):
        if mode == "Single Model":
            vis = frame_l.copy()
            if rt and rt.last_dets:
                draw_detections(vis, rt.last_dets, (0,255,255), sources=rt.last_dets_sources, show_source_tags=ts.get("show_source_tags",True))
                draw_temporal_overlays(vis, rt, ts)
                overlay_text_big(vis, build_overlay_lines(rt, ts))
            return vis
        trust = rt.last_trust if rt and rt.last_trust is not None else 0
        left = frame_l.copy()
        right = frame_r.copy() if frame_r is not None else cv2.cvtColor(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if len(right.shape)==2: right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        if mode == "Grayscale Fusion":
            gray = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            right = cv2.merge([gray,gray,gray])
        show_tags = ts.get("show_source_tags", True)
        if rt and rt.last_dets:
            draw_detections(left, rt.last_dets, (0,255,0), "RGB ", [trust in (1,3)]*len(rt.last_dets), rt.last_dets_sources, show_tags)
            draw_temporal_overlays(left, rt, ts)
        if it and it.last_dets:
            draw_detections(right, it.last_dets, (255,200,0), "IR ", [trust in (2,3)]*len(it.last_dets), it.last_dets_sources, show_tags)
            draw_temporal_overlays(right, it, ts)
        lh, lw = left.shape[:2]; rh, rw = right.shape[:2]
        if rh != lh: right = cv2.resize(right, (int(rw*lh/rh), lh))
        vis = np.hstack([left, right])
        if rt and it:
            tp = rt.last_trust_prob or 0.0
            tag = " [grayscale]" if mode == "Grayscale Fusion" else ""
            lines = [f"FUSION: {TRUST_LABELS.get(trust,'?')} ({tp*100:.1f}%){tag}"]
            suppressed = rt.confuser_suppressed or it.confuser_suppressed
            lines.append(f"CONFUSER GATE: {'alert suppressed' if suppressed else 'no alert suppressed'}")
            lines += build_overlay_lines(rt, ts, prefix="RGB ")
            lines += build_overlay_lines(it, ts, prefix="IR  ")
            overlay_text_big(vis, lines)
        return vis

    def _verifier_diag(self, fe, orig, new, rgp, irp, vetoed, rgb_dets, ir_dets, rgb_bgr, s):
        return {
            "active": fe.use_patch_verifier, "vetoed": bool(vetoed),
            "threshold": float(s.get("patch_threshold", 0.70)),
            "original_trust": int(orig), "original_trust_name": TRUST_LABELS.get(orig,"?"),
            "rgb_max_p": float(max(rgp)) if rgp else None,
            "ir_max_p": float(max(irp)) if irp else None,
            "rgb_n_boxes": len(rgp), "ir_n_boxes": len(irp),
            "rgb_labels": list(fe.rgb_verifier.last_labels) if fe.rgb_verifier and rgp else [],
            "ir_labels": list(fe.ir_verifier.last_labels) if fe.ir_verifier and irp else [],
        }

    @staticmethod
    def _best_center(dets):
        """Return (cx, cy, area) of the highest-confidence detection, or None.

        area is bbox width*height in pixels^2 — used downstream as a proxy for
        target range (apparent size grows as the drone approaches).
        """
        if not dets:
            return None
        best = max(dets, key=lambda d: d[4])
        cx = (best[0] + best[2]) / 2.0
        cy = (best[1] + best[3]) / 2.0
        area = max(0.0, (best[2] - best[0]) * (best[3] - best[1]))
        return (cx, cy, area)

    def _stats(self, mode, trust, probs, ms, rt, it, fe):
        # In single mode, show "single_model" instead of fusion trust labels
        if mode == "Single Model":
            trust_label = "single_model"
        else:
            trust_label = TRUST_LABELS.get(trust, "")

        # Detection centers for drone path tracking
        rgb_dets = rt.last_dets if rt else []
        ir_dets = it.last_dets if it else []
        rgb_center = self._best_center(rgb_dets)
        ir_center = self._best_center(ir_dets)
        # Frame dimensions (from last read capture)
        frame_w = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap_left else 0
        frame_h = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap_left else 0

        return {
            "frame": self.frame_num, "total": self.total_frames,
            "fps": round(1000/max(ms,1), 1), "trust": trust,
            "trust_label": trust_label,
            "trust_prob": float(probs[trust]) if trust < len(probs) else 0.0,
            "trust_probs": [round(p,4) for p in probs],
            "warning": (rt.warning_active if rt else False) or (it.warning_active if it else False),
            "alert": (rt.alert_active if rt else False) or (it.alert_active if it else False),
            "n_rgb": self._last_n_rgb,
            "n_ir": self._last_n_ir,
            "infer_ms": round(ms, 1),
            "confuser_suppressed": self._last_confuser_vetoed or (rt.confuser_suppressed if rt else False) or (it.confuser_suppressed if it else False),
            "confuser_labels": {
                "rgb": list(fe.rgb_verifier.last_labels)[:1] if fe.rgb_verifier and hasattr(fe.rgb_verifier,'last_labels') and fe.rgb_verifier.last_labels else [],
                "ir": list(fe.ir_verifier.last_labels)[:1] if fe.ir_verifier and hasattr(fe.ir_verifier,'last_labels') and fe.ir_verifier.last_labels else [],
            },
            "verifier": self._last_verifier,
            "w_events": (rt.count_warning_events if rt else 0) + (it.count_warning_events if it else 0),
            "a_events": (rt.count_alert_events if rt else 0) + (it.count_alert_events if it else 0),
            "mode": mode,
            # Drone path data
            "rgb_center": rgb_center,
            "ir_center": ir_center,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    def _cleanup(self):
        with self.lock:
            for cap in (self.cap_left, self.cap_right):
                if cap and cap.isOpened(): cap.release()
            self.cap_left = self.cap_right = None
            if self.writer: self.writer.release(); self.writer = None
