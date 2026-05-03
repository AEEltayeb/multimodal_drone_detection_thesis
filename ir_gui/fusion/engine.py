"""
FusionEngine — dual-YOLO inference + XGBoost trust classifier.

Three modes:
  1. Single model:   one YOLO, one video, no fusion
  2. Paired fusion:  two videos (RGB + IR), two YOLOs, fusion classifier
  3. Grayscale mode: one RGB video, RGB YOLO + IR YOLO on grayscale, fusion classifier

Optional post-classifier patch-verifier veto layer suppresses aerial
false positives (planes/helicopters/birds) that pass the 4-class trust
classifier. In grayscale mode only the RGB patch verifier runs, because
the IR verifier was trained on real thermal crops and is out-of-
distribution on gray-replicate input.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from .features import (
    TARGET_NAMES, compute_global_features, compute_target_features,
    GlobalFeatureCache,
)

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
try:
    from classifier.patch_verifier import PatchVerifier  # noqa: E402
except Exception as _exc:  # torch missing, etc.
    PatchVerifier = None
    _PATCH_IMPORT_ERROR = _exc
else:
    _PATCH_IMPORT_ERROR = None

# Trust label names
TRUST_LABELS = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
TRUST_COLORS = {
    0: (0, 0, 200),     # red
    1: (200, 100, 0),   # blue-ish
    2: (0, 140, 255),   # orange
    3: (0, 200, 0),     # green
}


@dataclass
class Detection:
    """Single YOLO detection."""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float


@dataclass
class FusionResult:
    """Result of one frame's fusion prediction."""
    trust_label: int = 0
    trust_name: str = "reject_both"
    trust_probs: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    rgb_dets: List[Detection] = field(default_factory=list)
    ir_dets: List[Detection] = field(default_factory=list)
    trusted_dets: List[Detection] = field(default_factory=list)
    suppressed_dets: List[Detection] = field(default_factory=list)
    infer_ms: float = 0.0
    features: dict = field(default_factory=dict)
    # Patch-verifier diagnostics (empty when verifier is disabled)
    rgb_patch_probs: List[float] = field(default_factory=list)
    ir_patch_probs: List[float] = field(default_factory=list)
    patch_veto: bool = False  # True if verifier overrode classifier to reject
    original_trust_label: Optional[int] = None


class FusionEngine:
    """Dual-modality fusion engine."""

    def __init__(
        self,
        rgb_weights: str,
        ir_weights: str,
        fusion_model_path: str,
        rgb_conf: float = 0.25,
        ir_conf: float = 0.40,
        nms_iou: float = 0.45,
        imgsz: int = 640,
        device: int = 0,
        rgb_patch_weights: Optional[str] = None,
        ir_patch_weights: Optional[str] = None,
        patch_threshold: float = 0.70,
        use_patch_verifier: bool = True,
        grayscale_run_ir_filter: bool = True,
        cascade_order: str = "filter_then_classifier",
        feature_stride: int = 5,
        feature_max_height: int = 480,
        # kept for back-compat; ignored (ir_conf is shared across modes)
        ir_conf_grayscale: Optional[float] = None,
    ):
        self.cascade_order = cascade_order if cascade_order in (
            "filter_then_classifier", "classifier_then_filter"
        ) else "filter_then_classifier"
        self.rgb_conf = rgb_conf
        self.ir_conf = ir_conf
        self.grayscale_run_ir_filter = bool(grayscale_run_ir_filter)
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.device = device

        # Strided cache for scene globals (shared by paired + grayscale paths).
        self.feature_cache = GlobalFeatureCache(
            stride=feature_stride, max_h=feature_max_height)

        # Load YOLO models
        print(f"[Fusion] Loading RGB YOLO: {rgb_weights}")
        self.rgb_model = YOLO(rgb_weights)
        print(f"[Fusion] Loading IR YOLO: {ir_weights}")
        self.ir_model = YOLO(ir_weights)

        # Load fusion classifier
        print(f"[Fusion] Loading fusion classifier: {fusion_model_path}")
        bundle = joblib.load(fusion_model_path)
        self.fusion_clf = bundle["model"]
        self.feature_names = bundle["features"]
        print(f"[Fusion] Ready. {len(self.feature_names)} features.")

        # Patch verifiers (optional)
        self.use_patch_verifier = use_patch_verifier
        self.patch_threshold = float(patch_threshold)
        self.rgb_verifier = None
        self.ir_verifier = None
        if use_patch_verifier and PatchVerifier is not None:
            if isinstance(device, int):
                import torch
                patch_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            else:
                patch_device = str(device)
            if rgb_patch_weights and Path(rgb_patch_weights).exists():
                print(f"[Fusion] Loading RGB patch verifier: {rgb_patch_weights}")
                self.rgb_verifier = PatchVerifier(rgb_patch_weights, patch_device)
            if ir_patch_weights and Path(ir_patch_weights).exists():
                print(f"[Fusion] Loading IR patch verifier: {ir_patch_weights}")
                self.ir_verifier = PatchVerifier(ir_patch_weights, patch_device)
            if self.rgb_verifier is None and self.ir_verifier is None:
                print("[Fusion] Patch verifier requested but no weights found; disabled.")
                self.use_patch_verifier = False
        elif use_patch_verifier:
            print(f"[Fusion] Patch verifier unavailable: {_PATCH_IMPORT_ERROR}")
            self.use_patch_verifier = False

    def _run_yolo(self, model, frame, conf):
        """Run YOLO and return list of Detection objects."""
        results = model.predict(
            frame, conf=conf, iou=self.nms_iou,
            imgsz=self.imgsz, verbose=False,
            device=self.device,
        )[0]
        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy()
                c = float(results.boxes.conf[i])
                dets.append(Detection(
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    conf=c,
                ))
        return dets

    def _det_stats(self, dets, prefix):
        """Aggregate detection-level info into frame-level features."""
        n = len(dets)
        if n == 0:
            return {
                f"{prefix}_n_dets": 0,
                f"{prefix}_max_conf": 0.0,
                f"{prefix}_mean_conf": 0.0,
                f"{prefix}_detected": 0,
            }
        confs = [d.conf for d in dets]
        return {
            f"{prefix}_n_dets": n,
            f"{prefix}_max_conf": round(max(confs), 6),
            f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
            f"{prefix}_detected": 1,
        }

    def _best_det_target(self, dets, img_gray, img_w, img_h, prefix):
        """Target features for highest-conf detection."""
        if not dets:
            return {f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES}
        best = max(dets, key=lambda d: d.conf)
        tf = compute_target_features(img_gray, best.box, img_w, img_h)
        return {f"{prefix}_best_{k}": v for k, v in tf.items()}

    def extract_features(self, rgb_dets, ir_dets, rgb_gray, ir_gray,
                          feature_mode="all"):
        """Build feature dict from detections and frames."""
        rgb_h, rgb_w = rgb_gray.shape[:2]
        ir_h, ir_w = ir_gray.shape[:2]

        feats = {}

        # Detection aggregates
        feats.update(self._det_stats(rgb_dets, "rgb"))
        feats.update(self._det_stats(ir_dets, "ir"))

        # Scene features
        # feature_mode is accepted for back-compat but ignored — the cache
        # always returns full-quality globals (recomputed every Nth call).
        rgb_global = self.feature_cache.get(rgb_gray, "rgb")
        ir_global = self.feature_cache.get(ir_gray, "ir")
        feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
        feats.update({f"ir_{k}": v for k, v in ir_global.items()})

        # Best-detection target features
        feats.update(self._best_det_target(rgb_dets, rgb_gray, rgb_w, rgb_h, "rgb"))
        feats.update(self._best_det_target(ir_dets, ir_gray, ir_w, ir_h, "ir"))

        # Agreement flags
        rgb_detected = len(rgb_dets) > 0
        ir_detected = len(ir_dets) > 0
        feats["both_detect"] = int(rgb_detected and ir_detected)
        feats["neither_detect"] = int(not rgb_detected and not ir_detected)
        feats["rgb_only_detect"] = int(rgb_detected and not ir_detected)
        feats["ir_only_detect"] = int(not rgb_detected and ir_detected)

        return feats

    def classify(self, feats):
        """Run XGBoost fusion classifier. Returns (label, probs)."""
        X = np.array([[feats[f] for f in self.feature_names]])
        label = int(self.fusion_clf.predict(X)[0])
        probs = self.fusion_clf.predict_proba(X)[0].tolist()
        return label, probs

    def _select_trusted(self, label, rgb_dets, ir_dets):
        """Pick trusted/suppressed detections based on trust label."""
        if label == 0:  # reject_both
            return [], rgb_dets + ir_dets
        elif label == 1:  # trust_rgb
            return rgb_dets, ir_dets
        elif label == 2:  # trust_ir
            return ir_dets, rgb_dets
        else:  # trust_both
            return rgb_dets + ir_dets, []

    @staticmethod
    def is_effectively_grayscale(img_bgr, max_channel_diff: int = 5) -> bool:
        """True if a 3-channel BGR image is effectively grayscale (all channels ≈ equal).

        Works for: deliberately gray-replicated frames, thermal videos loaded as BGR
        where the codec expanded a 1-channel source to 3 identical channels, or
        desaturated RGB. Returns False for real color images.

        Samples a subset of pixels for speed.
        """
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return True  # 1-channel / None → treat as gray
        h, w = img_bgr.shape[:2]
        step = max(1, min(h, w) // 64)  # ~4k sampled pixels
        sample = img_bgr[::step, ::step].astype(np.int16)
        b, g, r = sample[..., 0], sample[..., 1], sample[..., 2]
        return (int(np.abs(b - g).max()) <= max_channel_diff
                and int(np.abs(g - r).max()) <= max_channel_diff)

    def patch_veto(self, label, rgb_dets, ir_dets, rgb_bgr, ir_bgr,
                   ir_is_real_thermal: bool,
                   ir_verifier_enabled: Optional[bool] = None,
                   skip_ir_ood_gate: bool = False,
                   suppressed_classes: Optional[set] = None):
        """
        Apply confuser-filter veto layer.

        Returns (new_label, rgb_probs, ir_probs, vetoed).

        The model outputs P(confuser) — the probability that the crop
        is an airplane, helicopter, or bird.  If the highest P(confuser)
        across a modality's detections exceeds self.patch_threshold,
        that modality's trust is revoked — but only if the predicted
        class is in suppressed_classes (default: all confuser classes).

        Default is PASS: novel/unseen objects (including novel drones)
        get low P(confuser) and are accepted.

        Modality-honesty guards remain unchanged.
        """
        rgb_probs = []
        ir_probs = []
        if not self.use_patch_verifier or label == 0:
            return label, rgb_probs, ir_probs, False

        trust_rgb = label in (1, 3)
        trust_ir = label in (2, 3)

        rgb_is_color = not self.is_effectively_grayscale(rgb_bgr)
        # Caller may force IR verifier on (e.g. grayscale test mode): default
        # is the legacy "ir_is_real_thermal" gate, overridden by explicit flag.
        if ir_verifier_enabled is None:
            ir_is_thermal = bool(ir_is_real_thermal)
        else:
            ir_is_thermal = bool(ir_verifier_enabled)

        rgb_ood = []
        ir_ood = []
        if trust_rgb and self.rgb_verifier is not None and rgb_is_color and rgb_dets:
            boxes = [d.box for d in rgb_dets]
            p, o, _ = self.rgb_verifier.predict_boxes_with_ood(rgb_bgr, boxes)
            rgb_probs = p.tolist()
            rgb_ood = o.tolist()
        if trust_ir and self.ir_verifier is not None and ir_is_thermal and ir_dets:
            boxes = [d.box for d in ir_dets]
            p, o, _ = self.ir_verifier.predict_boxes_with_ood(ir_bgr, boxes)
            ir_probs = p.tolist()
            # Grayscale test mode: OOD calibration is thermal-only, so every
            # grayscale crop gets flagged OOD; disable the OOD gate so the
            # filter can actually veto.
            ir_ood = [False] * len(p) if skip_ir_ood_gate else o.tolist()

        # OOD crops do not contribute to the veto: if the verifier has never
        # seen anything like this input, it has no opinion and we default to pass.
        def _in_dist_max(probs, ood_flags):
            vals = [p for p, o in zip(probs, ood_flags) if not o]
            return max(vals) if vals else None

        def _class_is_suppressed(verifier):
            """Check if the top predicted class is in the suppressed set."""
            if suppressed_classes is None:
                return True  # no filter = suppress all confuser classes
            if verifier is None or not hasattr(verifier, 'last_labels') or not verifier.last_labels:
                return True  # no label info = default to suppress
            # last_labels[0] is e.g. "helicopter:0.85" or "pass(bird:0.12)"
            top = verifier.last_labels[0]
            cls_name = top.split(":")[0].strip()
            return cls_name in suppressed_classes

        rgb_ok = True
        ir_ok = True
        if trust_rgb and self.rgb_verifier is not None and rgb_is_color and rgb_probs:
            mx = _in_dist_max(rgb_probs, rgb_ood)
            if mx is not None and mx >= self.patch_threshold:
                rgb_ok = not _class_is_suppressed(self.rgb_verifier)
        if trust_ir and self.ir_verifier is not None and ir_is_thermal and ir_probs:
            mx = _in_dist_max(ir_probs, ir_ood)
            if mx is not None and mx >= self.patch_threshold:
                ir_ok = not _class_is_suppressed(self.ir_verifier)

        # Re-derive trust label from the surviving modalities
        new_trust_rgb = trust_rgb and rgb_ok
        new_trust_ir = trust_ir and ir_ok
        if new_trust_rgb and new_trust_ir:
            new_label = 3
        elif new_trust_rgb:
            new_label = 1
        elif new_trust_ir:
            new_label = 2
        else:
            new_label = 0
        return new_label, rgb_probs, ir_probs, new_label != label

    def _filter_dets_by_patch(self, rgb_dets, ir_dets, rgb_bgr, ir_bgr,
                              ir_is_real_thermal: bool,
                              ir_verifier_enabled: Optional[bool] = None,
                              skip_ir_ood_gate: bool = False):
        """Apply patch verifier per detection. Returns (rgb_kept, ir_kept,
        rgb_probs, ir_probs, rgb_dropped, ir_dropped). Used by filter_then_classifier."""
        rgb_probs, ir_probs = [], []
        if not self.use_patch_verifier:
            return rgb_dets, ir_dets, rgb_probs, ir_probs, [], []

        rgb_is_color = not self.is_effectively_grayscale(rgb_bgr)
        if ir_verifier_enabled is None:
            ir_is_thermal = bool(ir_is_real_thermal)
        else:
            ir_is_thermal = bool(ir_verifier_enabled)

        rgb_kept, rgb_dropped = list(rgb_dets), []
        ir_kept,  ir_dropped  = list(ir_dets),  []

        if self.rgb_verifier is not None and rgb_is_color and rgb_dets:
            boxes = [d.box for d in rgb_dets]
            p, o, _ = self.rgb_verifier.predict_boxes_with_ood(rgb_bgr, boxes)
            rgb_probs = p.tolist()
            rgb_kept, rgb_dropped = [], []
            for d, prob, ood in zip(rgb_dets, rgb_probs, o.tolist()):
                # OOD detections default to pass; in-distribution >= threshold gets vetoed
                if (not ood) and prob >= self.patch_threshold:
                    rgb_dropped.append(d)
                else:
                    rgb_kept.append(d)
        if self.ir_verifier is not None and ir_is_thermal and ir_dets:
            boxes = [d.box for d in ir_dets]
            p, o, _ = self.ir_verifier.predict_boxes_with_ood(ir_bgr, boxes)
            ir_probs = p.tolist()
            ood_flags = [False] * len(p) if skip_ir_ood_gate else o.tolist()
            ir_kept, ir_dropped = [], []
            for d, prob, ood in zip(ir_dets, ir_probs, ood_flags):
                if (not ood) and prob >= self.patch_threshold:
                    ir_dropped.append(d)
                else:
                    ir_kept.append(d)

        return rgb_kept, ir_kept, rgb_probs, ir_probs, rgb_dropped, ir_dropped

    def predict_paired(self, rgb_bgr, ir_bgr) -> FusionResult:
        """Mode 2: Paired fusion — two real frames (IR assumed real thermal)."""
        t0 = time.perf_counter()

        rgb_dets = self._run_yolo(self.rgb_model, rgb_bgr, self.rgb_conf)
        ir_dets = self._run_yolo(self.ir_model, ir_bgr, self.ir_conf)

        rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)

        if self.cascade_order == "filter_then_classifier":
            # Filter raw dets per modality, then classify on filtered features
            rgb_kept, ir_kept, rgb_patch_probs, ir_patch_probs, rgb_drp, ir_drp = \
                self._filter_dets_by_patch(rgb_dets, ir_dets, rgb_bgr, ir_bgr,
                                            ir_is_real_thermal=True)
            feats = self.extract_features(rgb_kept, ir_kept, rgb_gray, ir_gray)
            label, probs = self.classify(feats)
            original_label = label
            vetoed = bool(rgb_drp or ir_drp)
            trusted, suppressed = self._select_trusted(label, rgb_kept, ir_kept)
        else:
            # Legacy classifier_then_filter
            feats = self.extract_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
            label, probs = self.classify(feats)
            original_label = label
            label, rgb_patch_probs, ir_patch_probs, vetoed = self.patch_veto(
                label, rgb_dets, ir_dets, rgb_bgr, ir_bgr, ir_is_real_thermal=True)
            trusted, suppressed = self._select_trusted(label, rgb_dets, ir_dets)

        return FusionResult(
            trust_label=label,
            trust_name=TRUST_LABELS[label],
            trust_probs=probs,
            rgb_dets=rgb_dets,
            ir_dets=ir_dets,
            trusted_dets=trusted,
            suppressed_dets=suppressed,
            infer_ms=(time.perf_counter() - t0) * 1000,
            features=feats,
            rgb_patch_probs=rgb_patch_probs,
            ir_patch_probs=ir_patch_probs,
            patch_veto=vetoed,
            original_trust_label=original_label if vetoed else None,
        )

    def predict_grayscale(self, rgb_bgr) -> FusionResult:
        """Mode 3: Single RGB frame — IR YOLO runs on the grayscale version.
        Uses the shared `ir_conf` threshold (same as paired). The IR filter
        runs when `grayscale_run_ir_filter` is True (OOD gate disabled since
        calibration was thermal-only)."""
        t0 = time.perf_counter()

        rgb_dets = self._run_yolo(self.rgb_model, rgb_bgr, self.rgb_conf)

        # Convert to grayscale then back to 3-channel for IR YOLO
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ir_dets = self._run_yolo(self.ir_model, gray_3ch, self.ir_conf)

        rgb_gray = gray
        ir_gray = gray  # same image in grayscale mode

        if self.cascade_order == "filter_then_classifier":
            rgb_kept, ir_kept, rgb_patch_probs, ir_patch_probs, rgb_drp, ir_drp = \
                self._filter_dets_by_patch(
                    rgb_dets, ir_dets, rgb_bgr, gray_3ch,
                    ir_is_real_thermal=False,
                    ir_verifier_enabled=self.grayscale_run_ir_filter,
                    skip_ir_ood_gate=True)
            feats = self.extract_features(rgb_kept, ir_kept, rgb_gray, ir_gray)
            label, probs = self.classify(feats)
            original_label = label
            vetoed = bool(rgb_drp or ir_drp)
            trusted, suppressed = self._select_trusted(label, rgb_kept, ir_kept)
        else:
            feats = self.extract_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
            label, probs = self.classify(feats)
            original_label = label
            label, rgb_patch_probs, ir_patch_probs, vetoed = self.patch_veto(
                label, rgb_dets, ir_dets, rgb_bgr, gray_3ch,
                ir_is_real_thermal=False,
                ir_verifier_enabled=self.grayscale_run_ir_filter,
                skip_ir_ood_gate=True)
            trusted, suppressed = self._select_trusted(label, rgb_dets, ir_dets)

        return FusionResult(
            trust_label=label,
            trust_name=TRUST_LABELS[label],
            trust_probs=probs,
            rgb_dets=rgb_dets,
            ir_dets=ir_dets,
            trusted_dets=trusted,
            suppressed_dets=suppressed,
            infer_ms=(time.perf_counter() - t0) * 1000,
            features=feats,
            rgb_patch_probs=rgb_patch_probs,
            ir_patch_probs=ir_patch_probs,
            patch_veto=vetoed,
            original_trust_label=original_label if vetoed else None,
        )

    # ── Back-compat aliases for callers of the old private API ──────
    def _extract_features(self, *a, **kw): return self.extract_features(*a, **kw)
    def _classify(self, *a, **kw): return self.classify(*a, **kw)
    def _patch_veto(self, *a, **kw): return self.patch_veto(*a, **kw)
    @staticmethod
    def _is_effectively_grayscale(*a, **kw):
        return FusionEngine.is_effectively_grayscale(*a, **kw)

    def predict_single(self, frame_bgr, modality="rgb") -> FusionResult:
        """Mode 1: Single model — no fusion, just YOLO."""
        t0 = time.perf_counter()
        if modality == "rgb":
            dets = self._run_yolo(self.rgb_model, frame_bgr, self.rgb_conf)
        else:
            dets = self._run_yolo(self.ir_model, frame_bgr, self.ir_conf)

        return FusionResult(
            trust_label=3 if dets else 0,
            trust_name="single_model",
            trust_probs=[0.0, 0.0, 0.0, 1.0] if dets else [1.0, 0.0, 0.0, 0.0],
            rgb_dets=dets if modality == "rgb" else [],
            ir_dets=dets if modality == "ir" else [],
            trusted_dets=dets,
            suppressed_dets=[],
            infer_ms=(time.perf_counter() - t0) * 1000,
        )


def draw_fusion_frame(
    rgb_bgr, ir_bgr_or_gray, result: FusionResult,
    target_h: int = 480,
) -> np.ndarray:
    """
    Compose a side-by-side visualization frame for fusion modes.

    Returns a single BGR image: [RGB panel | IR panel] with detection
    boxes and a fusion status bar at the bottom.
    """
    rgb_vis = rgb_bgr.copy()
    ir_vis = ir_bgr_or_gray.copy()
    if len(ir_vis.shape) == 2:
        ir_vis = cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR)

    # Resize both to same height
    rgb_h, rgb_w = rgb_vis.shape[:2]
    ir_h, ir_w = ir_vis.shape[:2]
    rgb_scale = target_h / rgb_h
    ir_scale = target_h / ir_h
    rgb_vis = cv2.resize(rgb_vis, (int(rgb_w * rgb_scale), target_h))
    ir_vis = cv2.resize(ir_vis, (int(ir_w * ir_scale), target_h))

    # Draw detection boxes
    trust = result.trust_label
    for det in result.rgb_dets:
        is_trusted = (trust in (1, 3))
        color = (0, 220, 0) if is_trusted else (0, 0, 220)
        _draw_det(rgb_vis, det, rgb_scale, color, "RGB")
    for det in result.ir_dets:
        is_trusted = (trust in (2, 3))
        color = (0, 220, 0) if is_trusted else (0, 0, 220)
        _draw_det(ir_vis, det, ir_scale, color, "IR")

    # Labels on panels
    _put_label(rgb_vis, "RGB", (10, 30))
    _put_label(ir_vis, "IR / Thermal", (10, 30))

    # Concatenate side-by-side
    combined = np.hstack([rgb_vis, ir_vis])
    cw = combined.shape[1]

    # Fusion status bar (50px tall)
    bar_h = 50
    bar = np.zeros((bar_h, cw, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    # Trust decision text
    trust_color = TRUST_COLORS.get(trust, (200, 200, 200))
    trust_text = f"FUSION: {result.trust_name}"
    conf_text = f"({max(result.trust_probs) * 100:.1f}%)"
    cv2.putText(bar, trust_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                trust_color, 2, cv2.LINE_AA)
    cv2.putText(bar, conf_text, (15 + len(trust_text) * 17, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Detection counts
    det_text = f"RGB: {len(result.rgb_dets)} dets  |  IR: {len(result.ir_dets)} dets"
    cv2.putText(bar, det_text, (cw - 350, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Inference time
    ms_text = f"{result.infer_ms:.0f}ms"
    cv2.putText(bar, ms_text, (cw - 80, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100, 200, 100), 1, cv2.LINE_AA)

    return np.vstack([combined, bar])


def draw_single_frame(frame_bgr, result: FusionResult, target_h: int = 480) -> np.ndarray:
    """Compose visualization for single-model mode (full-width, no fusion bar)."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    scale = target_h / h
    vis = cv2.resize(vis, (int(w * scale), target_h))

    for det in result.trusted_dets:
        _draw_det(vis, det, scale, (0, 220, 220), "DET")

    return vis


def _draw_det(img, det: Detection, scale: float, color, tag: str):
    """Draw one detection box on a scaled image."""
    x1 = int(det.box[0] * scale)
    y1 = int(det.box[1] * scale)
    x2 = int(det.box[2] * scale)
    y2 = int(det.box[3] * scale)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{tag} {det.conf:.2f}"
    cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _put_label(img, text, pos):
    """Panel label with shadow."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
