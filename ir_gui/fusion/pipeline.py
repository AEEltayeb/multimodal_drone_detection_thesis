"""
FusionPipeline — orchestrates FusionEngine + per-modality temporal state.

Both the Tk GUI (`fusion_app.py`) and the FastAPI service (`api.py`) go
through this class so that detection logic, temporal gating, grayscale
behaviour, and confuser handling stay identical.

Per-frame flow:
    YOLO (RGB + IR, with TROI recovery)
      → classifier → patch veto (modality-scoped, OOD-aware)
      → per-modality temporal state update (trust-gated)
      → per-modality confuser temporal suppression
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import FusionConfig
from .engine import Detection, FusionEngine
from .temporal import PerModalityTemporalState


TRUST_LABELS = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


@dataclass
class FrameResult:
    mode: str                       # "paired" | "grayscale" | "single"
    rgb_dets: List[list] = field(default_factory=list)   # [x1,y1,x2,y2,conf]
    ir_dets:  List[list] = field(default_factory=list)
    rgb_sources: List[str] = field(default_factory=list)
    ir_sources:  List[str] = field(default_factory=list)
    rgb_troi_rois: List[tuple] = field(default_factory=list)
    ir_troi_rois:  List[tuple] = field(default_factory=list)
    original_trust: int = 0
    trust: int = 0
    trust_prob: float = 0.0
    rgb_patch_probs: list = field(default_factory=list)
    ir_patch_probs:  list = field(default_factory=list)
    rgb_patch_labels: list = field(default_factory=list)
    ir_patch_labels:  list = field(default_factory=list)
    vetoed: bool = False
    warning_active_rgb: bool = False
    alert_active_rgb: bool = False
    warning_active_ir: bool = False
    alert_active_ir: bool = False
    infer_ms: float = 0.0


class FusionPipeline:
    def __init__(self, engine: FusionEngine, config: FusionConfig,
                 fps: float = 30.0):
        self.engine = engine
        self.config = config
        self.fps = float(fps or 30.0)
        self.rgb_temporal: Optional[PerModalityTemporalState] = None
        self.ir_temporal:  Optional[PerModalityTemporalState] = None
        self._build_temporal()

    # ── setup ────────────────────────────────────────────────────────

    def _build_temporal(self) -> None:
        c = self.config
        stride = max(1, int(round(self.fps / max(1, c.infer_fps))))
        warn_cd = int(round(c.warning_cooldown_s * c.infer_fps))
        alert_cd = int(round(c.alert_cooldown_s * c.infer_fps))
        kwargs = dict(
            stride=stride,
            warning_window=c.warning_window_frames,
            warning_require=c.warning_require_hits,
            alert_window=c.alert_window_frames,
            alert_require=c.alert_require_hits,
            alert_avg_conf_thresh=c.alert_avg_conf_threshold,
            warning_cooldown_frames=warn_cd,
            alert_cooldown_frames=alert_cd,
            roi_ttl=c.roi_ttl,
            roi_expand=c.roi_expand,
        )
        self.rgb_temporal = PerModalityTemporalState(**kwargs)
        self.ir_temporal  = PerModalityTemporalState(**kwargs)

    def set_fps(self, fps: float) -> None:
        if fps and fps > 0 and abs(fps - self.fps) > 1e-6:
            self.fps = float(fps)
            self._build_temporal()

    def reset(self) -> None:
        if self.rgb_temporal: self.rgb_temporal.reset()
        if self.ir_temporal:  self.ir_temporal.reset()

    # ── YOLO + TROI ──────────────────────────────────────────────────

    def _run_yolo(self, model, frame, conf) -> List[list]:
        results = model.predict(
            frame, conf=conf, iou=self.engine.nms_iou,
            imgsz=self.engine.imgsz, verbose=False,
            device=self.engine.device,
        )[0]
        dets: List[list] = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy()
                c = float(results.boxes.conf[i])
                dets.append([float(x1), float(y1), float(x2), float(y2), c])
        return dets

    def _run_with_roi(self, model, frame, conf, temporal) \
            -> Tuple[List[list], List[str], List[tuple]]:
        dets = self._run_yolo(model, frame, conf)
        sources = ["full"] * len(dets)
        troi_rois: List[tuple] = []
        if temporal is not None and not dets and temporal.last_roi is not None \
                and temporal.roi_age > 0:
            h, w = frame.shape[:2]
            roi_result = temporal.get_roi_crop(frame, w, h)
            if roi_result is not None:
                crop, (ox, oy) = roi_result
                troi_rois.append((ox, oy, ox + crop.shape[1], oy + crop.shape[0]))
                crop_dets = self._run_yolo(model, crop, conf * 0.8)
                if crop_dets:
                    dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                    sources = ["troi"] * len(dets)
        return dets, sources, troi_rois

    @staticmethod
    def _wrap(dets: List[list]) -> List[Detection]:
        return [Detection(box=(d[0], d[1], d[2], d[3]), conf=d[4]) for d in dets]

    # ── main entry point ────────────────────────────────────────────

    def process(self, mode: str, rgb_frame, ir_frame=None) -> FrameResult:
        """Process one frame pair. `mode` ∈ {'paired','grayscale','single'}."""
        t0 = time.perf_counter()
        c = self.config

        if mode == "paired":
            return self._process_paired(rgb_frame, ir_frame, t0)
        if mode == "grayscale":
            return self._process_grayscale(rgb_frame, t0)
        if mode == "single":
            return self._process_single(rgb_frame, t0)
        raise ValueError(f"unknown mode: {mode}")

    def _process_paired(self, rgb_frame, ir_frame, t0) -> FrameResult:
        c = self.config
        rgb_dets, rgb_src, rgb_troi = self._run_with_roi(
            self.engine.rgb_model, rgb_frame, c.rgb_conf, self.rgb_temporal)
        ir_dets, ir_src, ir_troi = self._run_with_roi(
            self.engine.ir_model, ir_frame, c.ir_conf, self.ir_temporal)

        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        ir_gray = (cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
                   if ir_frame.ndim == 3 else ir_frame)

        rgb_wrapped = self._wrap(rgb_dets)
        ir_wrapped  = self._wrap(ir_dets)

        feats = self.engine.extract_features(
            rgb_wrapped, ir_wrapped, rgb_gray, ir_gray)
        trust, probs = self.engine.classify(feats)
        orig_trust = trust
        rgp, irp, vetoed = [], [], False
        ir_bgr = (ir_frame if ir_frame.ndim == 3
                  else cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR))
        if c.use_patch_verifier:
            trust, rgp, irp, vetoed = self.engine.patch_veto(
                trust, rgb_wrapped, ir_wrapped, rgb_frame, ir_bgr,
                ir_is_real_thermal=True)

        rh, rw = rgb_frame.shape[:2]
        ih_, iw_ = ir_frame.shape[:2]
        self._update_temporal(orig_trust, rgb_dets, ir_dets, rgb_src, ir_src,
                              rgb_troi, ir_troi, probs, rgp, irp,
                              rw, rh, iw_, ih_)

        return FrameResult(
            mode="paired",
            rgb_dets=rgb_dets, ir_dets=ir_dets,
            rgb_sources=rgb_src, ir_sources=ir_src,
            rgb_troi_rois=rgb_troi, ir_troi_rois=ir_troi,
            original_trust=orig_trust, trust=trust,
            trust_prob=float(probs[orig_trust]),
            rgb_patch_probs=list(rgp), ir_patch_probs=list(irp),
            rgb_patch_labels=self._verifier_labels("rgb", rgp),
            ir_patch_labels=self._verifier_labels("ir", irp),
            vetoed=vetoed,
            warning_active_rgb=self.rgb_temporal.warning_active,
            alert_active_rgb=self.rgb_temporal.alert_active,
            warning_active_ir=self.ir_temporal.warning_active,
            alert_active_ir=self.ir_temporal.alert_active,
            infer_ms=(time.perf_counter() - t0) * 1000,
        )

    def _process_grayscale(self, rgb_frame, t0) -> FrameResult:
        """Grayscale test mode — IR model runs on gray replicate of RGB.
        IR filter enabled per config (diagnostic mode)."""
        c = self.config
        rgb_dets, rgb_src, rgb_troi = self._run_with_roi(
            self.engine.rgb_model, rgb_frame, c.rgb_conf, self.rgb_temporal)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])
        ir_dets, ir_src, ir_troi = self._run_with_roi(
            self.engine.ir_model, gray_3ch, c.ir_conf, self.ir_temporal)

        rgb_wrapped = self._wrap(rgb_dets)
        ir_wrapped  = self._wrap(ir_dets)

        feats = self.engine.extract_features(
            rgb_wrapped, ir_wrapped, gray, gray)
        trust, probs = self.engine.classify(feats)
        orig_trust = trust
        rgp, irp, vetoed = [], [], False
        if c.use_patch_verifier:
            trust, rgp, irp, vetoed = self.engine.patch_veto(
                trust, rgb_wrapped, ir_wrapped, rgb_frame, gray_3ch,
                ir_is_real_thermal=False,
                ir_verifier_enabled=c.grayscale_run_ir_filter,
                skip_ir_ood_gate=c.grayscale_disable_filter_ood)

        rh, rw = rgb_frame.shape[:2]
        self._update_temporal(orig_trust, rgb_dets, ir_dets, rgb_src, ir_src,
                              rgb_troi, ir_troi, probs, rgp, irp,
                              rw, rh, rw, rh)

        return FrameResult(
            mode="grayscale",
            rgb_dets=rgb_dets, ir_dets=ir_dets,
            rgb_sources=rgb_src, ir_sources=ir_src,
            rgb_troi_rois=rgb_troi, ir_troi_rois=ir_troi,
            original_trust=orig_trust, trust=trust,
            trust_prob=float(probs[orig_trust]),
            rgb_patch_probs=list(rgp), ir_patch_probs=list(irp),
            rgb_patch_labels=self._verifier_labels("rgb", rgp),
            ir_patch_labels=self._verifier_labels("ir", irp),
            vetoed=vetoed,
            warning_active_rgb=self.rgb_temporal.warning_active,
            alert_active_rgb=self.rgb_temporal.alert_active,
            warning_active_ir=self.ir_temporal.warning_active,
            alert_active_ir=self.ir_temporal.alert_active,
            infer_ms=(time.perf_counter() - t0) * 1000,
        )

    def _process_single(self, frame, t0, modality: str = "rgb") -> FrameResult:
        """Legacy single-model mode (no fusion). Temporal still runs."""
        c = self.config
        model = self.engine.rgb_model if modality == "rgb" else self.engine.ir_model
        conf = c.rgb_conf if modality == "rgb" else c.ir_conf
        temporal = self.rgb_temporal if modality == "rgb" else self.ir_temporal
        dets, src, troi = self._run_with_roi(model, frame, conf, temporal)
        h, w = frame.shape[:2]
        temporal.update(dets, w, h)
        temporal.last_dets = list(dets)
        temporal.last_dets_sources = list(src)
        temporal.last_troi_rois = list(troi)
        trust = 3 if dets else 0
        temporal.last_trust = trust
        temporal.last_trust_prob = 1.0 if dets else 0.0
        return FrameResult(
            mode="single",
            rgb_dets=dets if modality == "rgb" else [],
            ir_dets=dets if modality == "ir" else [],
            rgb_sources=src if modality == "rgb" else [],
            ir_sources=src if modality == "ir" else [],
            rgb_troi_rois=troi if modality == "rgb" else [],
            ir_troi_rois=troi if modality == "ir" else [],
            original_trust=trust, trust=trust, trust_prob=float(bool(dets)),
            warning_active_rgb=(modality == "rgb" and temporal.warning_active),
            alert_active_rgb=(modality == "rgb" and temporal.alert_active),
            warning_active_ir=(modality == "ir" and temporal.warning_active),
            alert_active_ir=(modality == "ir" and temporal.alert_active),
            infer_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── temporal update (shared) ────────────────────────────────────

    def _update_temporal(self, orig_trust, rgb_dets, ir_dets,
                         rgb_src, ir_src, rgb_troi, ir_troi,
                         probs, rgp, irp,
                         rgb_w, rgb_h, ir_w, ir_h) -> None:
        c = self.config
        rh = self.rgb_temporal
        ih = self.ir_temporal

        rgb_feed = rgb_dets if orig_trust in (1, 3) else []
        ir_feed  = ir_dets  if orig_trust in (2, 3) else []

        # Shared confuser feed: paired/grayscale image the same scene, so the
        # stronger filter informs both alert chains. (Single-mode only has one
        # chain; behavior is equivalent.)
        rgb_max_p = float(max(rgp)) if rgp else None
        ir_max_p  = float(max(irp)) if irp else None
        shared_p = (max(p for p in (rgb_max_p, ir_max_p) if p is not None)
                    if (rgb_max_p is not None or ir_max_p is not None)
                    else None)
        rh.add_confuser_prob(shared_p)
        ih.add_confuser_prob(shared_p)

        thr = c.patch_threshold if c.use_patch_verifier else None
        rh.update(rgb_feed, rgb_w, rgb_h, confuser_threshold=thr)
        ih.update(ir_feed,  ir_w,  ir_h,  confuser_threshold=thr)

        rh.last_dets = list(rgb_dets)
        rh.last_dets_sources = list(rgb_src)
        rh.last_troi_rois = list(rgb_troi)
        ih.last_dets = list(ir_dets)
        ih.last_dets_sources = list(ir_src)
        ih.last_troi_rois = list(ir_troi)
        rh.last_trust = orig_trust
        rh.last_trust_prob = float(probs[orig_trust])

    def _verifier_labels(self, which: str, probs: list) -> list:
        """Copy verifier label strings for this frame (best-effort)."""
        v = self.engine.rgb_verifier if which == "rgb" else self.engine.ir_verifier
        if v is None or not probs:
            return []
        return list(getattr(v, "last_labels", []))
