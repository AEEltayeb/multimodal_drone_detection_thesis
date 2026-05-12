"""
datasets.py — Unified dataset loading for evaluation.

Handles Anti-UAV (paired), Svanström (paired), and YouTube (video) datasets.
Reads dataset configuration from eval/config.yaml.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import cv2
import yaml


EVAL_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    """Load eval config.yaml."""
    if config_path is None:
        config_path = EVAL_DIR / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    """Resolve a path relative to eval/ directory."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (EVAL_DIR / pp).resolve()


# ── YOLO label reading ───────────────────────────────────────────

def read_yolo_labels(path: Path, img_w: int, img_h: int,
                     drone_classes: set[int] | None = None) -> list[tuple]:
    """Read YOLO-format labels → list of (x1, y1, x2, y2) boxes.
    Only reads classes in *drone_classes* (default: {0}).
    Pass drone_classes=set() to get no boxes (negatives-only mode)."""
    if drone_classes is None:
        drone_classes = {0}
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if cls_id not in drone_classes:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append((
            (cx - bw / 2) * img_w,
            (cy - bh / 2) * img_h,
            (cx + bw / 2) * img_w,
            (cy + bh / 2) * img_h,
        ))
    return boxes


def img_from_label(label_path: Path, img_dir: Path) -> Path | None:
    """Find image file matching a label stem in images dir."""
    stem = label_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── Category detection ───────────────────────────────────────────

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")


def detect_category(key: str) -> str:
    """Detect category from Svanström-style frame key."""
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


# ── Paired dataset (Anti-UAV, Svanström) ─────────────────────────

class PairedDataset:
    """Loads a paired RGB+IR dataset with YOLO-format labels."""

    def __init__(self, ds_config: dict):
        self.root = Path(ds_config["root"])
        self.rgb_img_dir = self.root / ds_config.get("rgb_images", "RGB/images")
        self.rgb_lbl_dir = self.root / ds_config.get("rgb_labels", "RGB/labels")
        self.ir_img_dir = self.root / ds_config.get("ir_images", "IR/images")
        self.ir_lbl_dir = self.root / ds_config.get("ir_labels", "IR/labels")
        self.categories = ds_config.get("categories", ["DRONE"])
        self.rgb_suffix = ds_config.get("rgb_stem_suffix", "")
        self.ir_suffix = ds_config.get("ir_stem_suffix", "")

    def list_stems(self) -> list[str]:
        """List all label stems from the RGB labels directory."""
        stems = []
        if self.rgb_lbl_dir.exists():
            stems = sorted(p.stem for p in self.rgb_lbl_dir.glob("*.txt"))
        return stems

    def load_frame(self, stem: str) -> dict | None:
        """Load a single frame pair by stem.

        Returns dict with keys: rgb_img, ir_img, rgb_gt, ir_gt, rgb_path,
        ir_path, rgb_w, rgb_h, ir_w, ir_h, stem, category
        """
        rgb_lbl = self.rgb_lbl_dir / f"{stem}.txt"
        ir_stem = stem
        if self.rgb_suffix and self.ir_suffix:
            ir_stem = stem.replace(self.rgb_suffix, self.ir_suffix)
        ir_lbl = self.ir_lbl_dir / f"{ir_stem}.txt"

        rgb_path = img_from_label(rgb_lbl, self.rgb_img_dir)
        ir_path = img_from_label(Path(ir_lbl), self.ir_img_dir)
        if rgb_path is None or ir_path is None:
            return None

        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            return None

        rh, rw = rgb_img.shape[:2]
        ih, iw = ir_img.shape[:2]

        return {
            "rgb_img": rgb_img,
            "ir_img": ir_img,
            "rgb_gt": read_yolo_labels(rgb_lbl, rw, rh),
            "ir_gt": read_yolo_labels(ir_lbl, iw, ih),
            "rgb_path": rgb_path,
            "ir_path": ir_path,
            "rgb_w": rw, "rgb_h": rh,
            "ir_w": iw, "ir_h": ih,
            "stem": stem,
            "category": detect_category(stem),
        }


# ── Cached detection dataset ─────────────────────────────────────

class CachedDetectionDataset:
    """Loads pre-cached YOLO detections from JSON (from cache_inference.py
    or legacy run_inference.py)."""

    def __init__(self, cache_path: Path, rgb_img_dir: Path, ir_img_dir: Path):
        self.cache_path = cache_path
        self.rgb_img_dir = rgb_img_dir
        self.ir_img_dir = ir_img_dir
        self._data: dict | None = None

    def _load(self):
        if self._data is None:
            print(f"  Loading cache: {self.cache_path.name}...")
            self._data = json.loads(self.cache_path.read_text())

    def keys(self, stride: int = 1, limit: int | None = None) -> list[str]:
        self._load()
        keys = sorted(self._data.keys())
        if stride > 1:
            keys = keys[::stride]
        if limit:
            keys = keys[:limit]
        return keys

    def get_entry(self, key: str) -> dict:
        """Get cached entry with detections and image paths."""
        self._load()
        return self._data[key]

    def get_dets(self, key: str, conf_thr: float = 0.0):
        """Extract detections above confidence threshold.

        Returns (rgb_dets, ir_dets) as lists of ((x1,y1,x2,y2), conf).
        """
        entry = self.get_entry(key)
        rgb_all = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["rgb_dets"]]
        ir_all = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["ir_dets"]]
        rgb = [d for d in rgb_all if d[1] >= conf_thr]
        ir = [d for d in ir_all if d[1] >= conf_thr]
        return rgb, ir, rgb_all, ir_all


# ── Video dataset (YouTube OOD) ──────────────────────────────────

class VideoDataset:
    """Loads YouTube-style OOD video files with category labels."""

    def __init__(self, ds_config: dict):
        self.root = resolve_path(ds_config["root"])
        self.videos = ds_config.get("videos", {})
        self.drone_quality = ds_config.get("drone_quality", {})

    def available_videos(self) -> list[dict]:
        """Return list of available videos with metadata."""
        result = []
        for fname, cat in sorted(self.videos.items(), key=lambda x: (x[1], x[0])):
            fpath = self.root / fname
            if fpath.exists():
                result.append({
                    "filename": fname,
                    "category": cat,
                    "path": fpath,
                    "quality": self.drone_quality.get(fname, ""),
                })
        return result

    def iter_frames(
        self, video_path: Path, stride: int = 1, max_frames: int = 0
    ) -> Iterator[tuple[int, any]]:
        """Yield (frame_idx, frame) from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        idx = 0
        yielded = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if stride > 1 and (idx % stride != 0):
                continue
            if max_frames > 0 and yielded >= max_frames:
                break
            yield idx, frame
            yielded += 1
        cap.release()


# ── Single-modality image dataset (for eval_model.py) ────────────

class ImageDataset:
    """Loads a YOLO-format image dataset (images/ + labels/ dirs)."""

    def __init__(self, images_dir: Path, labels_dir: Path | None = None):
        self.images_dir = Path(images_dir)
        if labels_dir is None:
            # YOLO convention: .../images/split → .../labels/split
            # e.g. dataset/images/test → dataset/labels/test
            parts = self.images_dir.parts
            if "images" in parts:
                idx = len(parts) - 1 - list(reversed(parts)).index("images")
                candidate = Path(*parts[:idx]) / "labels" / Path(*parts[idx+1:])
                if candidate.exists():
                    self.labels_dir = candidate
                else:
                    self.labels_dir = self.images_dir.parent / "labels"
            else:
                self.labels_dir = self.images_dir.parent / "labels"
        else:
            self.labels_dir = Path(labels_dir)

    def list_images(self) -> list[Path]:
        """List all image files."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(p for p in self.images_dir.iterdir()
                      if p.suffix.lower() in exts)

    def load_frame(self, img_path: Path) -> dict | None:
        """Load image + GT labels."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        gt = read_yolo_labels(lbl_path, w, h)
        return {
            "img": img,
            "gt": gt,
            "path": img_path,
            "w": w, "h": h,
            "stem": img_path.stem,
            "category": detect_category(img_path.stem),
        }
