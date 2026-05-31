"""
mri.extract — YOLO internal-feature extraction (the "imaging" stage).

Attaches a forward-pre-hook to the Detect head, captures the FPN feature maps
the neck feeds into it, and pools per-detection ROIs into a flat feature vector:

    [ 5 metadata ] + [ per-layer ROI-pooled channels ]

The metadata block is always (conf, log_area, aspect, rel_cx, rel_cy). Each
selected layer (p3/p4/p5) contributes C * grid_h * grid_w features, where C is
the channel count of that map *as reported by the model at runtime* — so the
extractor is model-agnostic (works for any ultralytics detector, any channel
widths). The exact per-layer dimensions are recorded in FeatureSchema after the
first detection is seen.

Lifted and generalized from eval/distill_v5_p3p5_ft4.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Detect-head input is a list of maps ordered high-res -> low-res.
LAYER_INDEX = {"p3": 0, "p4": 1, "p5": 2}
LAYER_STRIDE = {"p3": 8, "p4": 16, "p5": 32}
META_NAMES = ["conf", "log_area", "aspect", "rel_cx", "rel_cy"]
META_DIM = len(META_NAMES)


class DetectInputHook:
    """Captures the feature maps fed into YOLO's Detect head.

    Stores the raw list (any length) so models with a different number of
    pyramid levels still work; named accessors map p3/p4/p5 to indices 0/1/2.
    """

    def __init__(self):
        self.maps: list[torch.Tensor] | None = None

    def clear(self):
        self.maps = None

    def _hook(self, module, args):
        x = args[0]  # list of feature maps from the neck
        self.maps = [t.detach() for t in x]

    def get(self, layer: str) -> torch.Tensor | None:
        idx = LAYER_INDEX[layer]
        if self.maps is None or idx >= len(self.maps):
            return None
        return self.maps[idx]

    def register(self, model):
        """Attach to the last module (Detect head) of an ultralytics model."""
        detect_mod = model.model.model[-1]
        return detect_mod.register_forward_pre_hook(self._hook)


def roi_pool(feature_map: torch.Tensor, box_xyxy, img_shape,
             out_h: int = 1, out_w: int = 1) -> np.ndarray:
    """Adaptive-average-pool a box region from a feature map.

    Returns a flat numpy array of length C * out_h * out_w.
    """
    _, C, H, W = feature_map.shape
    ih, iw = img_shape
    x1, y1, x2, y2 = box_xyxy
    fx1 = max(0, int(x1 / iw * W))
    fy1 = max(0, int(y1 / ih * H))
    fx2 = min(W, max(fx1 + 1, int(np.ceil(x2 / iw * W))))
    fy2 = min(H, max(fy1 + 1, int(np.ceil(y2 / ih * H))))
    crop = feature_map[0, :, fy1:fy2, fx1:fx2]
    pooled = nn.functional.adaptive_avg_pool2d(crop.unsqueeze(0), (out_h, out_w))
    return pooled.squeeze(0).flatten().cpu().numpy()


def extract_box_metadata(box_xyxy, conf, img_shape) -> np.ndarray:
    """5 detection-geometry features: conf, log_area, aspect, rel_cx, rel_cy."""
    x1, y1, x2, y2 = box_xyxy
    ih, iw = img_shape
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh
    cx = (x1 + x2) / 2.0 / max(iw, 1)
    cy = (y1 + y2) / 2.0 / max(ih, 1)
    return np.array([
        float(conf),
        float(np.log(max(area, 1.0))),
        float(bw / max(bh, 1)),
        float(cx),
        float(cy),
    ], dtype=np.float32)


@dataclass
class FeatureSchema:
    """Describes a flat feature vector's layout. Populated lazily once the first
    feature maps are seen (so per-layer channel counts come from the model)."""
    layers: tuple[str, ...]                       # e.g. ("p3", "p5")
    grids: dict[str, tuple[int, int]]             # layer -> (h, w)
    layer_dims: dict[str, int] = field(default_factory=dict)  # layer -> flat dim
    meta_dim: int = META_DIM

    @property
    def total_dim(self) -> int:
        return self.meta_dim + sum(self.layer_dims.get(l, 0) for l in self.layers)

    def column_label(self, idx: int) -> str:
        """Human label for a flat-vector column index (for plot ticks)."""
        if idx < self.meta_dim:
            return f"meta:{META_NAMES[idx]}"
        off = idx - self.meta_dim
        for layer in self.layers:
            d = self.layer_dims.get(layer, 0)
            if off < d:
                gh, gw = self.grids[layer]
                cells = gh * gw
                # roi_pool flattens (C, gh, gw) row-major => CHANNEL-major:
                # flat index = channel * cells + cell.
                chan = off // cells if cells else off
                cell = off % cells if cells else 0
                return f"{layer} ch={chan}" if cells == 1 else \
                       f"{layer} ch={chan} cell={cell}"
            off -= d
        return f"idx{idx}"

    def locate(self, idx: int):
        """Map a flat column index to (layer, channel, cell). layer='meta' for
        metadata (channel/cell None); layer=None if out of range."""
        if idx < self.meta_dim:
            return ("meta", None, None)
        off = idx - self.meta_dim
        for layer in self.layers:
            d = self.layer_dims.get(layer, 0)
            if off < d:
                gh, gw = self.grids[layer]
                cells = gh * gw
                # channel-major flatten: flat = channel*cells + cell
                return (layer, off // cells if cells else off,
                        off % cells if cells else 0)
            off -= d
        return (None, None, None)

    def layer_slices(self) -> dict[str, slice]:
        """Map each block name ('meta' + each layer) to its column slice."""
        out = {"meta": slice(0, self.meta_dim)}
        start = self.meta_dim
        for layer in self.layers:
            d = self.layer_dims.get(layer, 0)
            out[layer] = slice(start, start + d)
            start += d
        return out

    def to_dict(self) -> dict:
        return {
            "layers": list(self.layers),
            "grids": {k: list(v) for k, v in self.grids.items()},
            "layer_dims": dict(self.layer_dims),
            "meta_dim": self.meta_dim,
            "total_dim": self.total_dim,
            "metadata_order": META_NAMES,
        }


class FeatureExtractor:
    """Wraps a loaded YOLO model + hook and turns detections into feature rows."""

    DEFAULT_GRIDS = {"p3": (2, 2), "p4": (1, 1), "p5": (1, 1)}

    def __init__(self, model, layers=("p3", "p5"),
                 grids: dict[str, tuple[int, int]] | None = None):
        self.model = model
        self.layers = tuple(layers)
        self.grids = {l: (grids or {}).get(l, self.DEFAULT_GRIDS[l])
                      for l in self.layers}
        self.hook = DetectInputHook()
        self._handle = self.hook.register(model)
        self.schema = FeatureSchema(layers=self.layers, grids=self.grids)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _learn_dims(self):
        """Fill schema.layer_dims from the currently captured maps."""
        for layer in self.layers:
            fmap = self.hook.get(layer)
            if fmap is not None and layer not in self.schema.layer_dims:
                C = fmap.shape[1]
                gh, gw = self.grids[layer]
                self.schema.layer_dims[layer] = C * gh * gw

    def extract_one(self, box_xyxy, conf, img_shape) -> np.ndarray:
        """Feature vector for a single detection (hook must hold this image)."""
        self._learn_dims()
        parts = [extract_box_metadata(box_xyxy, conf, img_shape)]
        for layer in self.layers:
            fmap = self.hook.get(layer)
            gh, gw = self.grids[layer]
            if fmap is not None:
                parts.append(roi_pool(fmap, box_xyxy, img_shape, gh, gw))
            else:
                parts.append(np.zeros(self.schema.layer_dims.get(layer, 0),
                                      dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)
