"""
Feature extraction for the fusion classifier.

IMPORTANT: These functions are copied verbatim from
  classifier/reliability/fusion/build_fusion_dataset.py
to prevent train/serve skew. Any change here must be
mirrored there (and the model retrained).
"""

import cv2
import numpy as np


_TRAIN_MEANS = {
    "rgb": {"img_mean": 96.964, "img_std": 25.559, "img_dynamic_range": 71.883,
            "img_entropy": 4.943, "sky_ground_ratio": 1.288,
            "edge_density": 0.007581, "blurriness": 231.512},
    "ir":  {"img_mean": 85.290, "img_std": 50.707, "img_dynamic_range": 188.590,
            "img_entropy": 6.880, "sky_ground_ratio": 0.653,
            "edge_density": 0.025516, "blurriness": 882.696},
}


def compute_global_features(img_gray, feature_mode="all", modality="rgb", max_h=480):
    """7 scene-level features from a grayscale image.

    feature_mode:
        'all'            – full computation (downsampled for speed)
        'skip_expensive' – partial compute + mean-fill the rest (legacy)
        'detections_only'– all globals mean-filled (legacy)
    modality:
        'rgb' | 'ir' — which set of training-set means to use for fills.
        Caller must pass this explicitly; the function does not infer it.
    max_h:
        Target height for the downsample step. Lower = faster, less detail
        on small targets in Canny/Laplacian. Default 480 matches the
        training pipeline.

    Note: GlobalFeatureCache below is the production path — it caches
    full-quality globals and recomputes on stride/scene-cut. Direct
    callers (offline scripts) keep using this function.
    """
    if modality not in _TRAIN_MEANS:
        raise ValueError(f"modality must be 'rgb' or 'ir', got {modality!r}")
    _defaults = _TRAIN_MEANS[modality]

    if feature_mode == "detections_only":
        return {
            "img_mean": _defaults["img_mean"],
            "img_std": _defaults["img_std"],
            "img_dynamic_range": _defaults["img_dynamic_range"],
            "img_entropy": _defaults["img_entropy"],
            "sky_ground_ratio": _defaults["sky_ground_ratio"],
            "edge_density": _defaults["edge_density"],
            "blurriness": _defaults["blurriness"],
        }

    h, w = img_gray.shape[:2]
    # Downsample for speed — global stats don't need full resolution
    if h > max_h:
        scale = max_h / h
        img_gray = cv2.resize(img_gray, (int(w * scale), max_h), interpolation=cv2.INTER_AREA)
        h, w = img_gray.shape[:2]
    img_area = h * w
    img_f = img_gray.astype(np.float32)

    img_mean = float(img_f.mean())
    img_std = float(img_f.std())

    top_mean = float(img_f[:h // 2].mean())
    bot_mean = float(img_f[h // 2:].mean())
    sky_ground_ratio = top_mean / max(bot_mean, 1.0)

    if feature_mode == "skip_expensive":
        return {
            "img_mean": round(img_mean, 3),
            "img_std": round(img_std, 3),
            "img_dynamic_range": _defaults["img_dynamic_range"],
            "img_entropy": _defaults["img_entropy"],
            "sky_ground_ratio": round(sky_ground_ratio, 4),
            "edge_density": _defaults["edge_density"],
            "blurriness": _defaults["blurriness"],
        }

    # Full computation
    p2 = float(np.percentile(img_gray, 2))
    p98 = float(np.percentile(img_gray, 98))
    img_dynamic_range = p98 - p2

    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    img_entropy = float(-np.sum(p * np.log2(p)))

    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(edges.sum()) / (img_area * 255.0)

    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    blurriness = float(lap.var())

    return {
        "img_mean": round(img_mean, 3),
        "img_std": round(img_std, 3),
        "img_dynamic_range": round(img_dynamic_range, 3),
        "img_entropy": round(img_entropy, 4),
        "sky_ground_ratio": round(sky_ground_ratio, 4),
        "edge_density": round(edge_density, 6),
        "blurriness": round(blurriness, 3),
    }


def compute_target_features(img_gray, bbox_xyxy, img_w, img_h):
    """7 target-level features from a detection bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = pw * ph

    log_bbox_area = float(np.log(area + 1.0))
    aspect_ratio = float(pw / ph)
    pos_x = float(cx / img_w) if img_w > 0 else 0.5
    pos_y = float(cy / img_h) if img_h > 0 else 0.5
    dist_to_center = float(np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2))

    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))

    if xi2 <= xi1 or yi2 <= yi1:
        local_contrast = 0.0
        target_bg_delta = 0.0
    else:
        target = img_gray[yi1:yi2, xi1:xi2].astype(np.float32)
        target_mean = float(target.mean())
        mx, my = int(pw), int(ph)
        bx1, by1 = max(0, xi1 - mx), max(0, yi1 - my)
        bx2, by2 = min(img_w, xi2 + mx), min(img_h, yi2 + my)
        bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        bg_mean = float(bg.mean())
        bg_std = float(bg.std())
        target_bg_delta = target_mean - bg_mean
        local_contrast = target_bg_delta / bg_std if bg_std >= 1.0 else 0.0

    return {
        "log_bbox_area": round(log_bbox_area, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "pos_x": round(pos_x, 4),
        "pos_y": round(pos_y, 4),
        "dist_to_center": round(dist_to_center, 4),
        "local_contrast": round(local_contrast, 4),
        "target_bg_delta": round(target_bg_delta, 3),
    }


# Target feature names (order matters for best-det prefixing)
TARGET_NAMES = [
    "log_bbox_area", "aspect_ratio", "pos_x", "pos_y",
    "dist_to_center", "local_contrast", "target_bg_delta",
]


class GlobalFeatureCache:
    """Stride-cached scene features per modality.

    Scene globals (img_mean/std/entropy/edge_density/...) change on the
    timescale of camera motion, not target motion. Computing every frame
    is wasteful; mean-filling lies to the model. This cache recomputes
    every Nth call and reuses the last real value otherwise.

    A cheap scene-cut probe (img_mean delta vs cached) forces a
    recompute when the scene changes between stride boundaries.
    """

    def __init__(self, stride: int = 5, max_h: int = 480,
                 scene_cut_delta: float = 15.0):
        self.stride = max(1, int(stride))
        self.max_h = int(max_h)
        self.scene_cut_delta = float(scene_cut_delta)
        self._idx = {"rgb": 0, "ir": 0}
        self._cache = {"rgb": None, "ir": None}

    def reset(self):
        self._idx = {"rgb": 0, "ir": 0}
        self._cache = {"rgb": None, "ir": None}

    def configure(self, stride=None, max_h=None, scene_cut_delta=None):
        """Update params without resetting state. Reset() to clear cache."""
        if stride is not None:
            new = max(1, int(stride))
            if new != self.stride:
                self.stride = new
                self.reset()
        if max_h is not None:
            new = int(max_h)
            if new != self.max_h:
                self.max_h = new
                self.reset()
        if scene_cut_delta is not None:
            self.scene_cut_delta = float(scene_cut_delta)

    def get(self, img_gray, modality: str) -> dict:
        if modality not in ("rgb", "ir"):
            raise ValueError(modality)
        idx = self._idx[modality]
        cached = self._cache[modality]

        on_stride = (idx % self.stride == 0)
        force = (cached is None) or on_stride

        # Cheap scene-cut probe between stride boundaries.
        if (not force) and (cached is not None) and img_gray.size > 0:
            quick_mean = float(img_gray.mean())
            if abs(quick_mean - cached["img_mean"]) > self.scene_cut_delta:
                force = True

        if force:
            cached = compute_global_features(
                img_gray, "all", modality=modality, max_h=self.max_h)
            self._cache[modality] = cached

        self._idx[modality] = idx + 1
        return cached
