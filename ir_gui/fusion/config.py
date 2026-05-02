"""
FusionConfig — single source of truth for fusion-pipeline settings.

Both `fusion_app.py` (desktop Tk GUI) and `api.py` (FastAPI web service)
load, mutate, and persist this config. Keys are flat so they can be
serialised to JSON without transformation.

Defaults match the operating points validated by
`classifier/eval_six_configs.py`:
    rgb_conf=0.25, ir_conf=0.40, patch_threshold=0.70,
    nms_iou=0.45, imgsz=640.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class FusionConfig:
    # Model weights
    rgb_model: str = ""
    ir_model: str = ""
    fusion_model: str = ""
    rgb_patch_weights: Optional[str] = None
    ir_patch_weights: Optional[str] = None

    # Detector
    rgb_conf: float = 0.25
    ir_conf: float = 0.40
    nms_iou: float = 0.45
    imgsz: int = 640
    device: int = 0

    # Filter
    use_patch_verifier: bool = True
    patch_threshold: float = 0.70
    grayscale_run_ir_filter: bool = True
    grayscale_disable_filter_ood: bool = True

    # Temporal
    infer_fps: int = 5
    warning_window_frames: int = 10
    warning_require_hits: int = 9
    alert_window_frames: int = 10
    alert_require_hits: int = 9
    alert_avg_conf_threshold: float = 0.30
    warning_cooldown_s: float = 3.0
    alert_cooldown_s: float = 3.0
    roi_ttl: int = 5
    roi_expand: float = 1.5

    # Overlays
    show_troi: bool = True
    show_gate: bool = True
    show_source_tags: bool = True
    simple_mode: bool = False

    # Misc (back-compat keys occasionally referenced by legacy code)
    ir_conf_gray: float = field(default=0.40, metadata={"back_compat": True})

    # ── IO ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FusionConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: Path | str) -> "FusionConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception as exc:
            print(f"[FusionConfig] load failed ({exc}); using defaults")
            return cls()

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def update(self, **kwargs) -> None:
        """Apply a dict of overrides in place; silently ignores unknown keys."""
        known = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if k in known:
                setattr(self, k, v)

    # Convenience: ir_conf_gray is a legacy mirror of ir_conf. Always
    # kept in sync so callers still reading it get the shared value.
    def __post_init__(self):
        self.ir_conf_gray = self.ir_conf
