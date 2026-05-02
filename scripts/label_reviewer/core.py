"""
core.py — LabelReviewer class: interactive OpenCV-based label review.

Extracted from review_labels.py with smart grouping support.
"""
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def compute_iou_yolo(box_a, box_b):
    """Compute IoU between two YOLO boxes (cls, xc, yc, w, h, ...)."""
    ax1 = box_a[1] - box_a[3] / 2
    ay1 = box_a[2] - box_a[4] / 2
    ax2 = box_a[1] + box_a[3] / 2
    ay2 = box_a[2] + box_a[4] / 2
    bx1 = box_b[1] - box_b[3] / 2
    by1 = box_b[2] - box_b[4] / 2
    bx2 = box_b[1] + box_b[3] / 2
    by2 = box_b[2] + box_b[4] / 2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_mismatch(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Check if GT and predictions disagree on a frame.

    A mismatch occurs when:
    - GT has boxes but no matching prediction (missed detection)
    - Prediction exists but no GT box (false alarm)
    - Both have boxes but best IoU < threshold
    """
    has_gt = len(gt_boxes) > 0
    has_pred = len(pred_boxes) > 0

    # One has boxes, the other doesn't
    if has_gt != has_pred:
        return True

    # Neither has boxes — match (both agree: no drone)
    if not has_gt and not has_pred:
        return False

    # Both have boxes — check IoU matching
    matched_gt = set()
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou_yolo(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)

    # Mismatch if any GT unmatched or any pred unmatched
    unmatched_gt = len(gt_boxes) - len(matched_gt)
    unmatched_pred = len(pred_boxes) - len(matched_gt)
    return unmatched_gt > 0 or unmatched_pred > 0


def classify_frame(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Classify a frame as TP, FP, FN, TN, or mismatch.

    Returns one of: 'TP', 'FP', 'FN', 'TN', 'mismatch'
    """
    has_gt = len(gt_boxes) > 0
    has_pred = len(pred_boxes) > 0

    if not has_gt and not has_pred:
        return "TN"
    if not has_gt and has_pred:
        return "FP"
    if has_gt and not has_pred:
        return "FN"

    # Both have boxes — check IoU matching
    matched_gt = set()
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou_yolo(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)

    unmatched_gt = len(gt_boxes) - len(matched_gt)
    unmatched_pred = len(pred_boxes) - len(matched_gt)

    if unmatched_gt == 0 and unmatched_pred == 0:
        return "TP"
    return "mismatch"

# Colors (BGR)
# GT confidence tiers (blue family)
GT_COLORS = {
    "high":   (255, 200, 0),    # Bright blue
    "medium": (255, 150, 0),    # Medium blue
    "low":    (200, 100, 0),    # Dark blue
    "manual": (255, 255, 0),    # Cyan — manually added
}
# Model prediction color (always purple/magenta)
MODEL_COLOR = (255, 0, 255)    # Magenta/purple
# Secondary prediction overlay (for compare mode pred layer)
PRED_OVERLAY_COLOR = (255, 0, 255)  # Magenta

# Legacy alias for backward compat
COLORS = GT_COLORS
PRED_COLOR = PRED_OVERLAY_COLOR

WINDOW_NAME = "Label Reviewer"


# Smart Grouping

def auto_detect_grouping(filenames: list) -> callable:
    """Scan filenames and return a grouping function, or None."""
    if not filenames:
        return None

    sample = filenames[:min(20, len(filenames))]
    stems = [Path(f).stem for f in sample]

    # Try common patterns
    patterns = [
        (r'^(.+)_f(\d+)$', "prefix_frame"),         # VideoName_f000123
        (r'^(.+?)_(\d{4,})$', "prefix_numeric"),     # Something_0001
        (r'^(\d{3})(.+)$', "3digit_prefix"),          # 001SomeName
        (r'^(.+?)[-_]img(\d+)', "prefix_img"),        # 8_img025
    ]

    for pattern, name in patterns:
        matches = sum(1 for s in stems if re.match(pattern, s))
        if matches >= len(stems) * 0.7:  # 70% match threshold
            regex = re.compile(pattern)
            def make_fn(rgx):
                def fn(filename):
                    m = rgx.match(Path(filename).stem)
                    return m.group(1) if m else Path(filename).stem
                return fn
            return make_fn(regex)

    return None


def make_grouping_fn(mode: str, custom_pattern: str = None) -> callable:
    """Create a grouping function based on mode selection.

    Args:
        mode: One of 'auto', 'pattern', 'folder', 'none'
        custom_pattern: Regex with capture group (for 'pattern' mode)

    Returns:
        A function(filename) -> group_name, or None for no grouping.
    """
    if mode == "none":
        return None
    elif mode == "folder":
        return lambda filename: Path(filename).parent.name
    elif mode == "pattern" and custom_pattern:
        regex = re.compile(custom_pattern)
        def pattern_fn(filename):
            m = regex.match(Path(filename).stem)
            return m.group(1) if m else Path(filename).stem
        return pattern_fn
    elif mode == "auto":
        return "auto"  # Sentinel — resolved in LabelReviewer.__init__
    return None


class LabelReviewer:
    def __init__(self, images_dir: Path, labels_dir: Path,
                 manifest_path: Path = None, start_index: int = 0,
                 filter_tier: str = None, group_by_video: bool = True,
                 filter_file: Path = None, pred_labels_dir: Path = None,
                 split_manifest_path: Path = None,
                 filter_polarity: str = None,
                 filter_source: str = None,
                 output_dir: Path = None,
                 grouping_mode: str = None,
                 grouping_pattern: str = None,
                 label_source: str = "gt",
                 detection_filter: str = None):
        """Initialize the label reviewer.

        Args:
            label_source: 'gt' = labels are ground truth (blue),
                          'model' = labels are model predictions (purple).
                          Controls box colors, label text, and status bar.
            detection_filter: 'FP', 'FN', 'TP', 'mismatch', or None.
                              Pre-filters images to only show matching category.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.pred_labels_dir = pred_labels_dir
        self.label_source = label_source  # 'gt' or 'model'

        # ── Output directory: lazy copy (per-dataset, once) ──
        self.output_dir = output_dir
        self.changelog_file = None
        self._lazy_copied_stems = set()  # tracks which labels were copied on-demand
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.changelog_file = output_dir / "changelog.json"
            # Per-dataset marker: hash of (images_dir, labels_dir) to detect
            # if we already copied labels for THIS dataset in a prior session
            ds_hash = hashlib.md5(
                f"{images_dir.resolve()}|{labels_dir.resolve()}".encode()
            ).hexdigest()[:12]
            self._copy_marker = output_dir / f".copied_{ds_hash}"
            if self._copy_marker.exists():
                print(f"  Output directory already initialized (marker found). Skipping bulk copy.")
                print(f"  (Labels not in output_dir will be read from original labels_dir on demand)")
            else:
                # First time for this dataset.
                # Write marker FIRST so interrupted copies can resume.
                # load_labels() falls back to labels_dir for un-copied files.
                self._copy_marker.write_text(
                    f"src={labels_dir}\nstatus=lazy\n"
                )
                import time as _time
                t0 = _time.time()
                print(f"  First time reviewing this dataset. Scanning {labels_dir.name}...")
                src_labels = list(labels_dir.glob("*.txt"))
                total = len(src_labels)

                # Check how many already exist (from a previous interrupted copy)
                existing = sum(1 for lf in src_labels
                               if lf.name != "classes.txt" and (output_dir / lf.name).exists())
                if existing > 0:
                    print(f"  Resuming: {existing:,}/{total:,} labels already in output dir")

                count = 0
                skipped = 0
                print(f"  Copying {total:,} labels to {output_dir.name}...")
                for i, lf in enumerate(src_labels):
                    if lf.name != "classes.txt":
                        dest = output_dir / lf.name
                        if not dest.exists():
                            shutil.copy2(lf, dest)
                            count += 1
                        else:
                            skipped += 1
                    if (i + 1) % 5000 == 0:
                        print(f"    Progress: {i+1}/{total} (copied={count}, skipped={skipped})")
                elapsed = _time.time() - t0
                print(f"  Done: copied {count}, skipped {skipped} in {elapsed:.1f}s")
                # Update marker with final stats
                self._copy_marker.write_text(
                    f"copied={count} skipped={skipped} src={labels_dir}\nstatus=complete\n"
                )

        self.manifest = None
        self.filter_tier = filter_tier
        self.show_preds = pred_labels_dir is not None
        self.detection_filter = detection_filter

        # ── Smart Grouping ──
        if grouping_mode is not None:
            self._grouping_fn_raw = make_grouping_fn(grouping_mode, grouping_pattern)
        elif group_by_video:
            self._grouping_fn_raw = lambda f: _legacy_parse_video_name(f)[0]
        else:
            self._grouping_fn_raw = None

        self.group_by_video = self._grouping_fn_raw is not None

        # Load labeling manifest for confidence info (optional)
        if manifest_path and manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
            self.manifest = {e["stem"]: e for e in data["frames"]}

        # Load split manifest for polarity/source info (optional)
        self.stem_info = {}
        if split_manifest_path and split_manifest_path.exists():
            with open(split_manifest_path) as f:
                sm = json.load(f)
            for vid_id, vid_data in sm.get("videos", {}).items():
                polarity = vid_data.get("polarity", "UNKNOWN")
                source = vid_data.get("dataset_source", "unknown")
                for frame in vid_data.get("images", []):
                    stem = Path(frame["filename"]).stem
                    self.stem_info[stem] = {
                        "polarity": polarity,
                        "source": source,
                        "video_id": vid_id,
                    }
            print(f"  Loaded split manifest: {len(self.stem_info)} frames indexed")

        # ── Early cache init (needed by detection filter below) ──
        self._label_cache = {}  # {stem: [(cls, xc, yc, w, h), ...]}
        self._pred_cache = {}   # {stem: [(cls, xc, yc, w, h, conf), ...]}

        # Collect image files
        import time as _time
        t0 = _time.time()
        print(f"  Scanning images in {images_dir.name}...")
        all_images = sorted([f for f in images_dir.iterdir()
                             if f.suffix.lower() in IMG_EXTS])
        elapsed = _time.time() - t0
        print(f"  Found {len(all_images):,} images ({elapsed:.1f}s)")

        # Filter by tier if requested
        if filter_tier and self.manifest:
            all_images = [f for f in all_images
                          if self.manifest.get(f.stem, {}).get("tier") == filter_tier]

        # Filter by polarity from split manifest
        if filter_polarity and self.stem_info:
            fp = filter_polarity.upper()
            all_images = [f for f in all_images
                          if self.stem_info.get(f.stem, {}).get("polarity") == fp]
            print(f"  Filtered to {len(all_images)} frames with polarity={fp}")

        # Filter by source from split manifest
        if filter_source and self.stem_info:
            all_images = [f for f in all_images
                          if self.stem_info.get(f.stem, {}).get("source") == filter_source]
            print(f"  Filtered to {len(all_images)} frames from source={filter_source}")

        # Filter by stem list file
        if filter_file and filter_file.exists():
            stems = set(line.strip() for line in filter_file.read_text().splitlines() if line.strip())
            all_images = [f for f in all_images if f.stem in stems]
            print(f"  Filtered to {len(all_images)} frames from {filter_file.name}")

        # Filter by label content (Negatives/Positives) — works without pred_labels_dir
        if detection_filter and detection_filter in ("Negatives Only", "Positives Only"):
            import time as _dt
            import os as _os
            t0_df = _dt.time()
            want_empty = (detection_filter == "Negatives Only")
            filter_name = "negatives" if want_empty else "positives"

            # Determine which labels dir to scan
            scan_dir = self.output_dir if self.output_dir else labels_dir

            # --- Disk cache: try to reuse previous scan ---
            cache_key = hashlib.md5(
                f"{scan_dir.resolve()}|{detection_filter}".encode()
            ).hexdigest()[:12]
            cache_dir = Path("label_reviewer") / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f".filter_cache_{cache_key}.json"

            cached_stems = None
            if cache_file.exists():
                try:
                    with open(cache_file) as _cf:
                        cache_data = json.load(_cf)
                    cache_ts = cache_data.get("timestamp", 0)
                    # Invalidate if changelog has newer edits
                    changelog_path = (self.output_dir / "changelog.json") if self.output_dir else None
                    changelog_mtime = 0
                    if changelog_path and changelog_path.exists():
                        changelog_mtime = changelog_path.stat().st_mtime
                    if changelog_mtime <= cache_ts:
                        cached_stems = set(cache_data.get("stems", []))
                        print(f"  [CACHE HIT] Reusing {filter_name} filter ({len(cached_stems)} stems)")
                    else:
                        print(f"  [CACHE STALE] Changelog modified since last filter — rescanning")
                except Exception:
                    pass

            if cached_stems is not None:
                # Use cached result
                all_images = [f for f in all_images if f.stem in cached_stems]
            else:
                # Fast scan: use os.scandir to check file sizes (0 bytes = negative)
                print(f"  Scanning {scan_dir.name} with os.scandir for {filter_name}...")
                label_sizes = {}  # {stem: file_size_bytes}
                try:
                    with _os.scandir(str(scan_dir)) as entries:
                        for entry in entries:
                            if entry.name.endswith('.txt') and entry.name != 'classes.txt':
                                stem = entry.name[:-4]  # strip .txt
                                label_sizes[stem] = entry.stat().st_size
                except Exception as e:
                    print(f"  Warning: scandir failed ({e}), falling back to per-file check")
                    label_sizes = None

                if label_sizes is not None:
                    # Fast path: filter using file size
                    filtered = []
                    matching_stems = []
                    for img in all_images:
                        size = label_sizes.get(img.stem, -1)  # -1 = no label file
                        is_empty = (size <= 0)  # 0 bytes or missing
                        if want_empty == is_empty:
                            filtered.append(img)
                            matching_stems.append(img.stem)
                    all_images = filtered

                    # Save to disk cache
                    try:
                        import time as _tt
                        with open(cache_file, "w") as _cf:
                            json.dump({
                                "filter": detection_filter,
                                "stems": matching_stems,
                                "timestamp": _tt.time(),
                                "scan_dir": str(scan_dir),
                            }, _cf)
                    except Exception:
                        pass
                else:
                    # Fallback: slow per-file check
                    filtered = []
                    for img in all_images:
                        gt = self.load_labels(img)
                        is_empty = len(gt) == 0
                        if want_empty == is_empty:
                            filtered.append(img)
                    all_images = filtered

            elapsed_df = _dt.time() - t0_df
            print(f"  Filtered to {len(all_images)} {filter_name} frames ({elapsed_df:.1f}s)")

        # Filter by detection category (FP/FN/TP) — requires pred_labels_dir
        if detection_filter and pred_labels_dir and detection_filter not in ("All", "Negatives Only", "Positives Only"):
            import time as _dt
            t0_df = _dt.time()
            filt = detection_filter.upper().replace(" ONLY", "")
            print(f"  Classifying frames for {filt} filter...")
            filtered = []
            for img in all_images:
                gt = self.load_labels(img)
                pred = self.load_pred_labels(img)
                cat = classify_frame(gt, pred)
                if filt == "ALL MISMATCHES":
                    if cat in ("FP", "FN", "mismatch"):
                        filtered.append(img)
                elif cat == filt or (filt == "MISMATCH" and cat == "mismatch"):
                    filtered.append(img)
            elapsed_df = _dt.time() - t0_df
            print(f"  Filtered to {len(filtered)} {filt} frames out of {len(all_images)} ({elapsed_df:.1f}s)")
            all_images = filtered

        # Resolve auto-detect grouping
        if self._grouping_fn_raw == "auto":
            detected = auto_detect_grouping([img.name for img in all_images])
            if detected:
                self._grouping_fn = detected
                self.group_by_video = True
                print(f"  Auto-detected grouping pattern")
            else:
                self._grouping_fn = None
                self.group_by_video = False
                print(f"  No grouping pattern detected — using alphabetical order")
        elif callable(self._grouping_fn_raw):
            self._grouping_fn = self._grouping_fn_raw
        else:
            self._grouping_fn = None

        # Sort by group then frame number
        if self.group_by_video and self._grouping_fn:
            all_images.sort(key=lambda f: (self._grouping_fn(f.name), f.stem))

        self.images = all_images
        self.index = min(start_index, len(self.images) - 1) if self.images else 0
        self.boxes = []
        self.modified = False
        self.mode = "view"
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.reviewed_count = 0

        # Zoom state
        self.zoom = 1.0
        self.zoom_cx = 0.5
        self.zoom_cy = 0.5
        self.zoom_crop = None

        # Line thickness: cycle with 'b'
        self.thickness_levels = [1, 2, 3]
        self.thickness_idx = 1
        self.line_thickness = 2

        # ── V3: In-memory label cache ──
        # Cache labels once loaded; write-through on save
        self._label_cache = {}  # {stem: [(cls, xc, yc, w, h), ...]}
        self._pred_cache = {}   # {stem: [(cls, xc, yc, w, h, conf), ...]}

        # ── V3: Pre-compute group indices ──
        self.video_names = []
        self.video_start_idx = {}
        self._group_of_idx = []       # group name for each image index
        self._group_frame_counts = {} # {group_name: count}
        self._group_indices = {}      # {group_name: [idx, idx, ...]}

        if self.group_by_video and self._grouping_fn:
            print(f"  Building group index...")
            for i, img in enumerate(self.images):
                vname = self._grouping_fn(img.name)
                self._group_of_idx.append(vname)
                if vname not in self.video_start_idx:
                    self.video_names.append(vname)
                    self.video_start_idx[vname] = i
                    self._group_indices[vname] = []
                    self._group_frame_counts[vname] = 0
                self._group_indices[vname].append(i)
                self._group_frame_counts[vname] += 1
            print(f"  {len(self.video_names)} groups indexed")
        else:
            # No grouping — fill with empty sentinel
            self._group_of_idx = [None] * len(self.images)

        # ── V3: Build stem-to-index lookup for search ──
        self._stem_to_idx = {img.stem: i for i, img in enumerate(self.images)}

        # ── V3: Auto-detect dataset sources by filename prefix ──
        self._sources = []          # [(prefix, count, first_idx), ...]
        self._source_of_idx = []    # source prefix for each image index
        self._show_info = False     # toggle for info overlay
        self._show_help = False     # toggle for help overlay
        self._build_source_index()

        # ── V3: Thumbnail grid state ──
        self._thumbnail_mode = False
        self._thumb_page = 0         # current page in thumbnail grid
        self._thumb_cols = 10
        self._thumb_rows = 5
        self._thumb_per_page = self._thumb_cols * self._thumb_rows  # 50
        self._thumb_size = 120       # px per thumbnail
        self._thumb_rects = []       # [(x1, y1, x2, y2, img_index), ...] for click detection

        # ── Jump-to input state ──
        self._input_mode = None   # None, 'goto', or 'search'
        self._input_buffer = ""

        # ── Undo history (per-frame) ──
        self._undo_stack = []     # list of (boxes_copy, preds_copy)
        self._undo_max = 50       # max undo steps per frame

        # Progress tracking
        progress_dir = output_dir if output_dir else labels_dir
        self.progress_file = progress_dir / ".review_progress.json"
        self.progress = self._load_progress()
        print(f"  Ready. {len(self.images):,} images loaded.")

    def _get_group_name(self, filename: str) -> str:
        """Get group name for a filename using the active grouping function."""
        if self._grouping_fn:
            return self._grouping_fn(filename)
        return filename

    def _detect_prefix(self, filename: str) -> str:
        """Extract source prefix from filename."""
        stem = Path(filename).stem
        # Known prefix patterns (check longer first)
        known = [
            'dv4_may22_', 'dv4_gv2_', 'dv4_rob_', 'dv4_so_',
            'dv4_dd_', 'dv4_bird_', 'auv_', 'sv_',
        ]
        for prefix in known:
            if stem.startswith(prefix):
                return prefix.rstrip('_')

        # Fallback: strip last _segment(s) until we get a shared prefix
        # e.g. goldV2_c100 -> goldV2, goldV2_01079 -> goldV2
        parts = stem.split('_')
        if len(parts) >= 2:
            return parts[0]  # Use first segment as source
        return stem

    def _build_source_index(self):
        """Build source composition index from filename prefixes."""
        print(f"  Building source index...")
        counts = {}      # {prefix: count}
        first_idx = {}   # {prefix: first image index}

        for i, img in enumerate(self.images):
            prefix = self._detect_prefix(img.name)
            self._source_of_idx.append(prefix)
            if prefix not in counts:
                counts[prefix] = 0
                first_idx[prefix] = i
            counts[prefix] += 1

        # Sort by first appearance (preserves natural order)
        self._sources = [
            (prefix, counts[prefix], first_idx[prefix])
            for prefix in sorted(first_idx.keys(), key=lambda p: first_idx[p])
        ]

        print(f"  {len(self._sources)} sources detected:")
        for j, (prefix, count, fidx) in enumerate(self._sources):
            key_hint = f"  (press {j+1})" if j < 9 else ""
            print(f"    [{j+1}] {prefix}: {count:,} images{key_hint}")

    def _draw_info_overlay(self, vis: np.ndarray) -> np.ndarray:
        """Draw semi-transparent info overlay showing dataset composition."""
        h, w = vis.shape[:2]
        overlay = vis.copy()

        # Background panel
        panel_w = min(500, w - 40)
        panel_h = min(55 + len(self._sources) * 28 + 50, h - 40)
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.92, vis, 0.08, 0, vis)

        # Border
        cv2.rectangle(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), (89, 180, 250), 2)

        # Title
        cv2.putText(vis, "DATASET INFO  (press i to close)",
                    (x0 + 15, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (89, 180, 250), 2)

        # Column headers
        cy = y0 + 55
        cv2.putText(vis, "Key", (x0 + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(vis, "Source", (x0 + 55, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(vis, "Count", (x0 + 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(vis, "Range", (x0 + 280, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cy += 5
        cv2.line(vis, (x0 + 15, cy), (x0 + panel_w - 15, cy), (80, 80, 80), 1)
        cy += 18

        # Source rows
        cur_prefix = self._source_of_idx[self.index] if self.index < len(self._source_of_idx) else ""
        total = len(self.images)

        for j, (prefix, count, fidx) in enumerate(self._sources):
            is_current = (prefix == cur_prefix)
            color = (0, 255, 200) if is_current else (200, 200, 200)
            marker = "> " if is_current else "  "

            key_str = str(j + 1) if j < 9 else "-"
            pct = count / total * 100 if total > 0 else 0

            cv2.putText(vis, key_str, (x0 + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (89, 180, 250), 1)
            cv2.putText(vis, f"{marker}{prefix}", (x0 + 55, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(vis, f"{count:,} ({pct:.0f}%)", (x0 + 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(vis, f"#{fidx+1}-#{fidx+count}", (x0 + 280, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
            cy += 28

        # Footer
        cy += 5
        cv2.line(vis, (x0 + 15, cy), (x0 + panel_w - 15, cy), (80, 80, 80), 1)
        cy += 20
        cv2.putText(vis, f"Total: {total:,} images  |  Press 1-9 to jump to source",
                    (x0 + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        return vis

    def _draw_thumbnail_grid(self) -> np.ndarray:
        """Render a grid of thumbnail previews for the current page."""
        cols, rows = self._thumb_cols, self._thumb_rows
        ts = self._thumb_size
        pad = 4
        cell = ts + pad

        total_pages = max(1, (len(self.images) + self._thumb_per_page - 1) // self._thumb_per_page)
        self._thumb_page = max(0, min(self._thumb_page, total_pages - 1))
        start_idx = self._thumb_page * self._thumb_per_page
        end_idx = min(start_idx + self._thumb_per_page, len(self.images))

        # Canvas
        canvas_w = cols * cell + pad
        canvas_h = rows * cell + pad + 40 + 30  # +header +footer
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        # Header
        header = f"THUMBNAILS  Page {self._thumb_page + 1}/{total_pages}  |  [{start_idx+1}-{end_idx}] of {len(self.images):,}  |  t/ESC:close  Arrow:page"
        cv2.putText(canvas, header, (pad, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (89, 180, 250), 1)

        self._thumb_rects = []
        reviewed_set = set(self.progress.get("reviewed", []))

        for slot, img_idx in enumerate(range(start_idx, end_idx)):
            row = slot // cols
            col = slot % cols
            x = pad + col * cell
            y = 40 + row * cell

            img_path = self.images[img_idx]

            # Load and resize thumbnail
            thumb = cv2.imread(str(img_path))
            if thumb is None:
                thumb = np.full((ts, ts, 3), 50, dtype=np.uint8)
                cv2.putText(thumb, "ERR", (ts//3, ts//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                thumb = cv2.resize(thumb, (ts, ts), interpolation=cv2.INTER_AREA)

            # Draw labels on thumbnail
            boxes = self.load_labels(img_path)
            th, tw = thumb.shape[:2]
            for cls, xc, yc, bw, bh in boxes:
                bx1 = int((xc - bw/2) * tw)
                by1 = int((yc - bh/2) * th)
                bx2 = int((xc + bw/2) * tw)
                by2 = int((yc + bh/2) * th)
                cv2.rectangle(thumb, (bx1, by1), (bx2, by2), (255, 200, 0), 1)

            # Place on canvas
            canvas[y:y+ts, x:x+ts] = thumb

            # Border color: cyan=current, yellow=reviewed, green=has labels, red=empty
            is_reviewed = img_path.stem in reviewed_set
            is_current = (img_idx == self.index)
            if is_current:
                border_color = (250, 250, 0)  # cyan
            elif is_reviewed:
                border_color = (0, 200, 200)  # yellow
            elif len(boxes) > 0:
                border_color = (0, 180, 0)    # green
            else:
                border_color = (0, 0, 180)    # red

            thickness = 3 if is_current else 1
            cv2.rectangle(canvas, (x-1, y-1), (x+ts, y+ts), border_color, thickness)

            # Frame number badge
            label_str = f"{img_idx+1}"
            cv2.putText(canvas, label_str, (x+2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            # Label count badge (top-right)
            if boxes:
                badge = str(len(boxes))
                cv2.rectangle(canvas, (x+ts-20, y), (x+ts, y+14), (0, 140, 0), -1)
                cv2.putText(canvas, badge, (x+ts-18, y+11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            self._thumb_rects.append((x, y, x+ts, y+ts, img_idx))

        # Footer
        fy = canvas_h - 20
        cv2.putText(canvas, "Click to jump  |  Green=labels  Red=empty  Yellow=reviewed  Cyan=current",
                    (pad, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        return canvas

    def _thumb_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on thumbnail grid."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for rx1, ry1, rx2, ry2, img_idx in self._thumb_rects:
                if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                    self._thumbnail_mode = False
                    self.index = img_idx
                    break

    def _draw_help_overlay(self, vis: np.ndarray) -> np.ndarray:
        """Draw help overlay showing all hotkeys."""
        h, w = vis.shape[:2]
        overlay = vis.copy()

        entries = [
            ("", ""),
            ("-- Navigation --", ""),
            ("< >  (Shift+,/.)", "Skip 5 frames"),
            ("PgUp / PgDn", "Skip 50 frames"),
            ("v / V", "Next / Prev group"),
            ("g", "Go to frame number"),
            ("/", "Search by filename"),
            ("1-9", "Jump to source dataset"),
            ("G", "Jump to last frame"),
            ("", ""),
            ("-- Editing --", ""),
            ("a", "Add mode (draw box)"),
            ("d", "Delete mode (click box)"),
            ("n", "Clear all labels"),
            ("D (Shift+D)", "Delete image + label from dataset"),
            ("R (Shift+R)", "Delete RANGE of images + labels"),
            ("Ctrl+Z", "Undo last edit"),
            ("s", "Save + next frame"),
            ("r", "Reload from disk"),
            ("", ""),
            ("-- View --", ""),
            ("t", "Thumbnail grid"),
            ("i", "Dataset info panel"),
            ("b", "Cycle line thickness"),
            ("scroll", "Zoom in/out"),
            ("ESC", "Back to view mode"),
            ("q", "Quit"),
        ]

        if self.show_preds:
            entries += [
                ("", ""),
                ("-- Predictions --", ""),
                ("w", "Promote single pred"),
                ("W", "Promote all preds"),
                ("x", "Swap: replace GT with preds"),
                ("m / M", "Next / Prev mismatch"),
            ]

        # Dynamic height based on entry count
        n_lines = 0
        n_gaps = 0
        for key, desc in entries:
            if not key and not desc:
                n_gaps += 1
            else:
                n_lines += 1
        panel_h = min(40 + n_lines * 16 + n_gaps * 5 + 10, h - 20)
        panel_w = min(480, w - 40)
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.92, vis, 0.08, 0, vis)
        cv2.rectangle(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), (89, 180, 250), 2)

        cv2.putText(vis, "HELP  (press ? to close)",
                    (x0 + 15, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (89, 180, 250), 2)

        cy = y0 + 48
        for key, desc in entries:
            if cy > y0 + panel_h - 10:
                break  # Don't draw outside panel
            if not key and not desc:
                cy += 5
                continue
            if key.startswith("--"):
                cv2.putText(vis, key.strip("- "), (x0 + 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (89, 180, 250), 1)
                cy += 18
                continue
            cv2.putText(vis, key, (x0 + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(vis, desc, (x0 + 180, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
            cy += 16

        return vis

    def _load_progress(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {"reviewed": [], "last_index": 0}

    def _save_progress(self):
        self.progress["last_index"] = self.index
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def _log_change(self, stem: str, boxes_before: int, boxes_after: int):
        """Append a change entry to changelog.json."""
        if not self.changelog_file:
            return
        changelog = []
        if self.changelog_file.exists():
            try:
                with open(self.changelog_file) as f:
                    changelog = json.load(f)
            except (json.JSONDecodeError, Exception):
                changelog = []

        if boxes_after > boxes_before:
            action = "add_box"
        elif boxes_after < boxes_before:
            action = "delete_box"
        else:
            action = "modify"

        entry = {
            "stem": stem,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "boxes_before": boxes_before,
            "boxes_after": boxes_after,
        }
        changelog.append(entry)

        with open(self.changelog_file, "w") as f:
            json.dump(changelog, f, indent=2)

    def load_labels(self, img_path: Path) -> list:
        """Load YOLO labels for an image. Uses in-memory cache."""
        stem = img_path.stem

        # Check cache first
        if stem in self._label_cache:
            return list(self._label_cache[stem])  # return copy

        # Read from disk
        read_dir = self.output_dir if self.output_dir else self.labels_dir
        label_path = read_dir / f"{stem}.txt"
        if not label_path.exists() and self.output_dir:
            label_path = self.labels_dir / f"{stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((cls, xc, yc, w, h))

        # Store in cache
        self._label_cache[stem] = list(boxes)
        return boxes

    def load_pred_labels(self, img_path: Path) -> list:
        """Load prediction labels (YOLO format with optional confidence column)."""
        if not self.pred_labels_dir:
            return []
        stem = img_path.stem

        # Check cache
        if stem in self._pred_cache:
            return list(self._pred_cache[stem])

        pred_path = self.pred_labels_dir / f"{stem}.txt"
        preds = []
        if pred_path.exists():
            with open(pred_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        conf = float(parts[5]) if len(parts) >= 6 else -1
                        preds.append((cls, xc, yc, w, h, conf))

        self._pred_cache[stem] = list(preds)
        return preds

    def _push_undo(self):
        """Save current box+pred state to the undo stack."""
        snapshot = (list(self.boxes), list(self.preds) if self.show_preds else [])
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._undo_max:
            self._undo_stack.pop(0)

    def _undo(self):
        """Restore the previous box+pred state from the undo stack."""
        if not self._undo_stack:
            return False
        self.boxes, preds = self._undo_stack.pop()
        if self.show_preds:
            self.preds = preds
            self.preds_modified = True
        self.modified = True
        return True

    def save_labels(self, img_path: Path, boxes: list):
        """Save YOLO labels for an image. Write-through to disk + cache."""
        write_dir = self.output_dir if self.output_dir else self.labels_dir
        label_path = write_dir / f"{img_path.stem}.txt"

        # Get old count from cache (not disk) for changelog
        old_boxes = self._label_cache.get(img_path.stem, [])
        boxes_before = len(old_boxes)

        with open(label_path, "w") as f:
            for cls, xc, yc, w, h in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # Update cache
        self._label_cache[img_path.stem] = list(boxes)

        if self.changelog_file and len(boxes) != boxes_before:
            self._log_change(img_path.stem, boxes_before, len(boxes))

    def _delete_single_image_files(self, stem: str, img_path: Path):
        """Delete the files for a single image (image + labels). Internal helper.

        Returns list of paths that were deleted.
        """
        files_to_delete = [img_path]

        # Label in output dir (working copy)
        if self.output_dir:
            out_label = self.output_dir / f"{stem}.txt"
            if out_label.exists():
                files_to_delete.append(out_label)

        # Label in original labels dir
        orig_label = self.labels_dir / f"{stem}.txt"
        if orig_label.exists():
            files_to_delete.append(orig_label)

        # Pred label if applicable
        if self.pred_labels_dir:
            pred_label = self.pred_labels_dir / f"{stem}.txt"
            if pred_label.exists():
                files_to_delete.append(pred_label)

        # Delete the files
        for f in files_to_delete:
            f.unlink()

        # Clear caches
        self._label_cache.pop(stem, None)
        self._pred_cache.pop(stem, None)

        return files_to_delete

    def delete_image_and_label(self, img_path: Path):
        """Delete the image and its label file from the dataset.

        Removes the source image, its label in the output/labels dir,
        and the original label. Logs the deletion to the changelog.
        Returns the next (img_path, img) to display, or None if no images left.
        """
        stem = img_path.stem
        files_to_delete = self._delete_single_image_files(stem, img_path)

        # Log to changelog
        if self.changelog_file:
            changelog = []
            if self.changelog_file.exists():
                try:
                    with open(self.changelog_file) as f:
                        changelog = json.load(f)
                except (json.JSONDecodeError, Exception):
                    changelog = []
            changelog.append({
                "stem": stem,
                "action": "delete_image",
                "timestamp": datetime.now().isoformat(),
                "deleted_files": [str(f) for f in files_to_delete],
            })
            with open(self.changelog_file, "w") as f:
                json.dump(changelog, f, indent=2)

        # Remove from images list
        self.images.pop(self.index)

        if not self.images:
            return None

        # Adjust index
        if self.index >= len(self.images):
            self.index = len(self.images) - 1

        # Load the next image
        new_path = self.images[self.index]
        new_img = cv2.imread(str(new_path))
        self.boxes = self.load_labels(new_path)
        self.preds = self.load_pred_labels(new_path) if self.show_preds else []
        self.modified = False
        self.preds_modified = False
        self.mode = "view"
        self.zoom = 1.0
        self._undo_stack.clear()
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, new_img)
        return new_path, new_img

    def _resolve_range_input(self, range_str: str):
        """Parse range input and return (start_idx, end_idx) inclusive, or None.

        Supports:
          - Frame numbers: '100-200' (1-indexed)
          - Filenames: 'filename_a-filename_b' (matches stems shown in info panel)
        """
        range_str = range_str.strip()
        if not range_str or '-' not in range_str:
            return None

        # Try to split as frame numbers first (both sides are purely numeric)
        # Handle possible negative frame numbers by splitting on the last '-'
        # that separates two parts
        parts = range_str.split('-')

        # Try interpreting as frame numbers
        if len(parts) == 2:
            try:
                start = int(parts[0].strip()) - 1  # 1-indexed → 0-indexed
                end = int(parts[1].strip()) - 1
                if start < 0:
                    start = 0
                if end >= len(self.images):
                    end = len(self.images) - 1
                if start > end:
                    start, end = end, start  # swap if reversed
                return (start, end)
            except ValueError:
                pass

        # Try interpreting as filenames (find first match for each part)
        # Split only on the first '-' to support filenames containing dashes
        first_dash = range_str.index('-')
        name_a = range_str[:first_dash].strip().lower()
        name_b = range_str[first_dash + 1:].strip().lower()

        if not name_a or not name_b:
            return None

        idx_a = None
        idx_b = None
        for i, img in enumerate(self.images):
            stem_lower = img.stem.lower()
            if idx_a is None and name_a in stem_lower:
                idx_a = i
            if name_b in stem_lower:
                idx_b = i  # keep last match for end

        if idx_a is not None and idx_b is not None:
            if idx_a > idx_b:
                idx_a, idx_b = idx_b, idx_a
            return (idx_a, idx_b)

        return None

    def delete_range(self, start_idx: int, end_idx: int):
        """Delete all images + labels from start_idx to end_idx (inclusive).

        Returns the next (img_path, img) to display, or None if no images left.
        """
        # Clamp indices
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self.images) - 1)
        if start_idx > end_idx:
            return self.images[self.index], cv2.imread(str(self.images[self.index]))

        count = end_idx - start_idx + 1
        all_deleted_files = []
        deleted_stems = []

        # Delete files for each image in the range (iterate in reverse to
        # not invalidate indices, but we'll bulk-remove from list after)
        for idx in range(start_idx, end_idx + 1):
            img_path = self.images[idx]
            stem = img_path.stem
            deleted = self._delete_single_image_files(stem, img_path)
            all_deleted_files.extend(deleted)
            deleted_stems.append(stem)

        # Log bulk deletion to changelog
        if self.changelog_file:
            changelog = []
            if self.changelog_file.exists():
                try:
                    with open(self.changelog_file) as f:
                        changelog = json.load(f)
                except (json.JSONDecodeError, Exception):
                    changelog = []
            changelog.append({
                "stems": deleted_stems,
                "action": "delete_range",
                "timestamp": datetime.now().isoformat(),
                "count": count,
                "range": f"{start_idx + 1}-{end_idx + 1}",
                "deleted_files": [str(f) for f in all_deleted_files],
            })
            with open(self.changelog_file, "w") as f:
                json.dump(changelog, f, indent=2)

        # Remove from images list (bulk)
        del self.images[start_idx:end_idx + 1]

        print(f"  Deleted {count} images + labels (frames {start_idx+1}-{end_idx+1})")

        if not self.images:
            return None

        # Adjust index
        self.index = min(start_idx, len(self.images) - 1)

        # Load the next image
        new_path = self.images[self.index]
        new_img = cv2.imread(str(new_path))
        self.boxes = self.load_labels(new_path)
        self.preds = self.load_pred_labels(new_path) if self.show_preds else []
        self.modified = False
        self.preds_modified = False
        self.mode = "view"
        self.zoom = 1.0
        self._undo_stack.clear()
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, new_img)
        return new_path, new_img

    def save_pred_labels(self, img_path: Path):
        """Save current prediction labels back to pred_labels_dir."""
        if not self.pred_labels_dir:
            return
        pred_path = self.pred_labels_dir / f"{img_path.stem}.txt"
        with open(pred_path, "w") as f:
            for pred in self.preds:
                cls, xc, yc, w, h = pred[:5]
                conf = pred[5] if len(pred) > 5 else -1
                if conf >= 0:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
                else:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def promote_pred(self, pred_idx: int):
        """Promote a prediction to GT (move from preds to boxes)."""
        if 0 <= pred_idx < len(self.preds):
            self._push_undo()
            pred = self.preds.pop(pred_idx)
            cls, xc, yc, w, h = pred[:5]
            self.boxes.append((cls, xc, yc, w, h))
            self.modified = True
            self.preds_modified = True

    def promote_all_preds(self):
        """Promote all predictions on this frame to GT."""
        self._push_undo()
        while self.preds:
            pred = self.preds.pop(0)
            cls, xc, yc, w, h = pred[:5]
            self.boxes.append((cls, xc, yc, w, h))
        self.modified = True
        self.preds_modified = True

    def swap_labels(self):
        """Replace all GT boxes with predictions (clear GT, promote all preds)."""
        self._push_undo()
        self.boxes = []
        while self.preds:
            pred = self.preds.pop(0)
            cls, xc, yc, w, h = pred[:5]
            self.boxes.append((cls, xc, yc, w, h))
        self.modified = True
        self.preds_modified = True

    def get_confidence(self, stem: str, box_idx: int) -> float:
        """Get confidence for a detection from manifest."""
        if self.manifest and stem in self.manifest:
            confs = self.manifest[stem].get("confidences", [])
            if box_idx < len(confs):
                return confs[box_idx]
        return -1

    def get_tier(self, conf: float) -> str:
        if conf < 0:
            return "manual"
        if conf >= 0.80:
            return "high"
        if conf >= 0.50:
            return "medium"
        return "low"

    def draw_frame(self, img: np.ndarray, img_path: Path) -> np.ndarray:
        """Draw current frame with boxes and UI bars (non-overlapping)."""
        vis = img.copy()
        h, w = vis.shape[:2]

        # Determine primary box color based on label source
        is_model_labels = (self.label_source == "model")
        box_tag = "PRED" if is_model_labels else "GT"

        # Draw secondary prediction overlay (below primary boxes)
        if self.show_preds and self.preds:
            for pi, pred in enumerate(self.preds):
                cls, xc, yc, bw, bh = pred[:5]
                pconf = pred[5] if len(pred) > 5 else -1

                px1 = int((xc - bw / 2) * w)
                py1 = int((yc - bh / 2) * h)
                px2 = int((xc + bw / 2) * w)
                py2 = int((yc + bh / 2) * h)

                thickness = 2 if self.mode == "promote" else 1
                cv2.rectangle(vis, (px1, py1), (px2, py2), PRED_OVERLAY_COLOR, thickness)
                cv2.rectangle(vis, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), PRED_OVERLAY_COLOR, 1)

                plabel = f"PRED#{pi}"
                if pconf >= 0:
                    plabel += f" {pconf:.2f}"
                cv2.putText(vis, plabel, (px1, py2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_OVERLAY_COLOR, 1)

        # Draw primary boxes (GT or model depending on label_source)
        for i, (cls, xc, yc, bw, bh) in enumerate(self.boxes):
            if is_model_labels:
                color = MODEL_COLOR
            else:
                conf = self.get_confidence(img_path.stem, i)
                tier = self.get_tier(conf)
                color = GT_COLORS.get(tier, (255, 200, 0))

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            thickness = self.line_thickness + 1 if self.mode == "delete" else self.line_thickness
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            if is_model_labels:
                conf = self.get_confidence(img_path.stem, i)
                if conf >= 0:
                    label = f"PRED#{i} {conf:.2f}"
                else:
                    label = f"PRED#{i}"
            else:
                conf = self.get_confidence(img_path.stem, i)
                if conf >= 0:
                    label = f"GT#{i} {conf:.2f}"
                else:
                    label = f"GT#{i}"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw in-progress box (add mode)
        if self.drawing and self.draw_start and self.draw_end:
            cv2.rectangle(vis, self.draw_start, self.draw_end,
                          GT_COLORS["manual"], 2)

        # Apply zoom
        if self.zoom > 1.0:
            zh = int(h / self.zoom)
            zw = int(w / self.zoom)
            cx = int(self.zoom_cx * w)
            cy = int(self.zoom_cy * h)
            x1z = max(0, cx - zw // 2)
            y1z = max(0, cy - zh // 2)
            x2z = min(w, x1z + zw)
            y2z = min(h, y1z + zh)
            if x2z - x1z < zw:
                x1z = max(0, x2z - zw)
            if y2z - y1z < zh:
                y1z = max(0, y2z - zh)
            self.zoom_crop = (x1z, y1z, x2z, y2z)
            crop = vis[y1z:y2z, x1z:x2z]
            vis = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            self.zoom_crop = None

        # ── Build canvas: top bar (50px) + image + bottom bar (25px) ──
        bar_top = 50
        bar_bot = 25
        canvas = np.full((h + bar_top + bar_bot, w, 3), 35, dtype=np.uint8)
        canvas[bar_top:bar_top + h, 0:w] = vis
        self._bar_top_h = bar_top  # stored for mouse offset

        # ── Top Status Bar (2 rows) ──
        status_color = (0, 255, 0) if not self.modified else (0, 165, 255)
        reviewed = "[R]" if img_path.stem in self.progress.get("reviewed", []) else "[ ]"
        mode_colors = {
            "view": (0, 255, 0), "add": (0, 200, 255),
            "delete": (0, 0, 255), "promote": (255, 180, 0)
        }
        mode_col = mode_colors.get(self.mode, (200, 200, 200))

        # Row 1: position + mode
        row1 = f"{self.index + 1}/{len(self.images)}  {reviewed}  Mode: {self.mode.upper()}"
        cv2.putText(canvas, row1, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_col, 1)

        # Source tag (right-aligned on row 1)
        if self.index < len(self._source_of_idx):
            src_tag = f"src: {self._source_of_idx[self.index]}"
            text_w = cv2.getTextSize(src_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            cv2.putText(canvas, src_tag, (w - text_w - 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 200, 255), 1)

        # Frame classification badge (FP/FN/TP/TN)
        if self.show_preds:
            frame_cat = classify_frame(self.boxes, self.preds)
            cat_colors = {
                "TP": (0, 200, 0), "TN": (150, 150, 150),
                "FP": (0, 0, 255), "FN": (0, 165, 255),
                "mismatch": (0, 100, 255),
            }
            cat_col = cat_colors.get(frame_cat, (200, 200, 200))
            cat_label = frame_cat
            if self.detection_filter and self.detection_filter != "All":
                cat_label += f" [{self.detection_filter}]"
            cat_tw = cv2.getTextSize(cat_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
            cat_x = w - cat_tw - 10
            cv2.putText(canvas, cat_label, (cat_x, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cat_col, 2)

        # Row 2: counts + group
        count_info = f"{len(self.boxes)} {box_tag}"
        if self.show_preds:
            count_info += f"  |  {len(self.preds)} PRED"

        group_info = ""
        if self.group_by_video and self.video_names and self.index < len(self._group_of_idx):
            vname = self._group_of_idx[self.index]
            if vname and vname in self._group_frame_counts:
                vid_idx = self.video_names.index(vname)
                vid_count = self._group_frame_counts[vname]
                group_idxs = self._group_indices[vname]
                vid_pos = 0
                for gi in group_idxs:
                    vid_pos += 1
                    if gi == self.index:
                        break
                group_info = f"  |  Grp {vid_idx+1}/{len(self.video_names)}: {vname} [{vid_pos}/{vid_count}]"

        row2 = f"{count_info}{group_info}"
        cv2.putText(canvas, row2, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        # ── Bottom Hotkey Bar (context-sensitive) ──
        bot_y = bar_top + h + 17
        if self._input_mode:
            if self._input_mode == "goto":
                prompt = "Go to #: "
            elif self._input_mode == "search":
                prompt = "Search: "
            elif self._input_mode == "range_delete":
                prompt = "Delete range (e.g. 100-200 or name_a-name_b): "
            elif self._input_mode == "range_confirm":
                prompt = f"DELETE {self._range_delete_count} images? (y=yes, n=cancel): "
            else:
                prompt = ""
            keys = f"{prompt}{self._input_buffer}_  (Enter=go, Esc=cancel)"
        elif self.mode == "add":
            keys = "DRAW MODE: Click+drag to draw box | ESC: back to view"
        elif self.mode == "delete":
            keys = "DELETE MODE: Click a box to remove | ESC: back to view"
        elif self.mode == "promote":
            keys = "PROMOTE MODE: Click a prediction to promote to GT | ESC: back to view"
        else:
            keys = "<>:skip5  PgUp/Dn:skip50  v/V:group  a:add  d:del  D:del img  R:del range  s:save  t:thumbs  i:info  ?:help  q:quit"
        cv2.putText(canvas, keys, (10, bot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Info overlay (toggle with 'i')
        if self._show_info:
            canvas = self._draw_info_overlay(canvas)

        # Help overlay (toggle with '?')
        if self._show_help:
            canvas = self._draw_help_overlay(canvas)

        return canvas

    def _autosave(self, img_path: Path):
        """Auto-save labels and pred labels if modified."""
        if self.modified:
            self.save_labels(img_path, self.boxes)
            self.modified = False
        if self.preds_modified:
            self.save_pred_labels(img_path)
            self.preds_modified = False

    def _navigate_to(self, new_index: int, img_path: Path):
        """Navigate to a new frame index, auto-saving current."""
        self._autosave(img_path)
        if img_path.stem not in self.progress["reviewed"]:
            self.progress["reviewed"].append(img_path.stem)
            self.reviewed_count += 1
        self._save_progress()
        self.index = new_index
        self._undo_stack.clear()  # reset undo for new frame
        new_path = self.images[self.index]
        new_img = cv2.imread(str(new_path))
        self.boxes = self.load_labels(new_path)
        self.preds = self.load_pred_labels(new_path) if self.show_preds else []
        self.modified = False
        self.preds_modified = False
        self.mode = "view"
        self.zoom = 1.0
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, new_img)
        return new_path, new_img

    def _screen_to_img(self, sx, sy, w, h):
        """Convert screen (zoomed) coordinates to original image coordinates.
        Accounts for the top bar offset."""
        # Subtract top bar offset from screen y
        bar_top = getattr(self, '_bar_top_h', 0)
        sy = sy - bar_top
        # Clamp to image area
        sy = max(0, min(sy, h - 1))
        sx = max(0, min(sx, w - 1))
        if self.zoom_crop is not None:
            cx1, cy1, cx2, cy2 = self.zoom_crop
            cw = cx2 - cx1
            ch = cy2 - cy1
            ox = cx1 + (sx / w) * cw
            oy = cy1 + (sy / h) * ch
            return ox, oy
        return float(sx), float(sy)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for add/delete/zoom."""
        img = param
        h, w = img.shape[:2]

        # Mouse wheel zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            ox, oy = self._screen_to_img(x, y, w, h)
            self.zoom_cx = ox / w
            self.zoom_cy = oy / h
            if flags > 0:
                self.zoom = min(self.zoom * 1.3, 8.0)
            else:
                self.zoom = max(self.zoom / 1.3, 1.0)
            return

        if self.mode == "add":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.draw_start = (x, y)
                self.draw_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.draw_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                self.draw_end = (x, y)
                ox1, oy1 = self._screen_to_img(self.draw_start[0], self.draw_start[1], w, h)
                ox2, oy2 = self._screen_to_img(self.draw_end[0], self.draw_end[1], w, h)
                if abs(ox2 - ox1) > 3 and abs(oy2 - oy1) > 3:
                    self._push_undo()
                    xc = ((ox1 + ox2) / 2) / w
                    yc = ((oy1 + oy2) / 2) / h
                    bw = abs(ox2 - ox1) / w
                    bh = abs(oy2 - oy1) / h
                    self.boxes.append((0, xc, yc, bw, bh))
                    self.modified = True

        elif self.mode == "delete":
            if event == cv2.EVENT_LBUTTONDOWN:
                ox, oy = self._screen_to_img(x, y, w, h)
                best_area = float("inf")
                best_idx = -1
                best_source = None

                for i, (cls, xc, yc, bw, bh) in enumerate(self.boxes):
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    if x1 <= ox <= x2 and y1 <= oy <= y2:
                        area = bw * bh
                        if area < best_area:
                            best_area = area
                            best_idx = i
                            best_source = "gt"

                if self.show_preds:
                    for i, pred in enumerate(self.preds):
                        xc, yc, bw, bh = pred[1], pred[2], pred[3], pred[4]
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        if x1 <= ox <= x2 and y1 <= oy <= y2:
                            area = bw * bh
                            if area < best_area:
                                best_area = area
                                best_idx = i
                                best_source = "pred"

                if best_idx >= 0:
                    self._push_undo()
                    if best_source == "gt":
                        self.boxes.pop(best_idx)
                        self.modified = True
                    elif best_source == "pred":
                        self.preds.pop(best_idx)
                        self.preds_modified = True

        elif self.mode == "promote":
            if event == cv2.EVENT_LBUTTONDOWN and self.preds:
                min_dist = float("inf")
                min_idx = -1
                for i, pred in enumerate(self.preds):
                    xc, yc = pred[1], pred[2]
                    cx = xc * w
                    cy = yc * h
                    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                if min_idx >= 0 and min_dist < 200:
                    self.promote_pred(min_idx)

    def run(self):
        """Main review loop."""
        if not self.images:
            print("No images found!")
            return

        # Resume from last position
        if self.progress.get("last_index", 0) > 0:
            self.index = min(self.progress["last_index"], len(self.images) - 1)
            print(f"  Resuming from frame {self.index + 1}/{len(self.images)}")
            print(f"  Already reviewed: {len(self.progress.get('reviewed', []))} frames")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        img_path = self.images[self.index]
        img = cv2.imread(str(img_path))
        self.boxes = self.load_labels(img_path)
        self.preds = self.load_pred_labels(img_path) if self.show_preds else []
        self.modified = False
        self.preds_modified = False

        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, img)

        print(f"\n  Review started. {len(self.images)} frames to review.")
        print(f"  Controls: arrows=nav, a=add, d=del, s=save+next, q=quit\n")

        while True:
            # ── Thumbnail grid mode ──
            if self._thumbnail_mode:
                grid = self._draw_thumbnail_grid()
                cv2.imshow(WINDOW_NAME, grid)
                cv2.setMouseCallback(WINDOW_NAME, self._thumb_mouse_callback)
                key = cv2.waitKeyEx(30)

                if key == ord('t') or key == 27:  # t or ESC = close grid
                    self._thumbnail_mode = False
                    # Reload current frame
                    img_path = self.images[self.index]
                    img = cv2.imread(str(img_path))
                    self.boxes = self.load_labels(img_path)
                    self.preds = self.load_pred_labels(img_path) if self.show_preds else []
                    self.modified = False
                    self.preds_modified = False
                    cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, img)
                elif key == 2555904 or key == ord('.'):  # Right = next page
                    self._thumb_page += 1
                elif key == 2424832 or key == ord(','):  # Left = prev page
                    self._thumb_page = max(0, self._thumb_page - 1)
                elif key == 2228224:  # PageDown
                    self._thumb_page += 5
                elif key == 2162688:  # PageUp
                    self._thumb_page = max(0, self._thumb_page - 5)
                elif key == ord('q'):
                    self._save_progress()
                    break

                # Check if user clicked a thumbnail (index changed)
                if not self._thumbnail_mode:
                    img_path = self.images[self.index]
                    img = cv2.imread(str(img_path))
                    self.boxes = self.load_labels(img_path)
                    self.preds = self.load_pred_labels(img_path) if self.show_preds else []
                    self.modified = False
                    self.preds_modified = False
                    cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback, img)

                continue  # Skip normal review loop
            vis = self.draw_frame(img, img_path)
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKeyEx(30)

            # ── V3: Input mode (goto / search / range_delete) ──
            if self._input_mode:
                if key == 27:  # ESC — cancel input
                    self._input_mode = None
                    self._input_buffer = ""
                elif key == 13:  # Enter — confirm
                    if self._input_mode == "goto":
                        try:
                            target = int(self._input_buffer) - 1  # 1-indexed input
                            target = max(0, min(target, len(self.images) - 1))
                            img_path, img = self._navigate_to(target, img_path)
                        except ValueError:
                            pass
                        self._input_mode = None
                        self._input_buffer = ""
                    elif self._input_mode == "search":
                        query = self._input_buffer.lower()
                        # Search from current position forward
                        for j in range(self.index + 1, len(self.images)):
                            if query in self.images[j].stem.lower():
                                img_path, img = self._navigate_to(j, img_path)
                                break
                        else:
                            # Wrap around from start
                            for j in range(0, self.index):
                                if query in self.images[j].stem.lower():
                                    img_path, img = self._navigate_to(j, img_path)
                                    break
                        self._input_mode = None
                        self._input_buffer = ""
                    elif self._input_mode == "range_delete":
                        result = self._resolve_range_input(self._input_buffer)
                        if result:
                            start_idx, end_idx = result
                            count = end_idx - start_idx + 1
                            # Move to confirmation step
                            self._range_delete_start = start_idx
                            self._range_delete_end = end_idx
                            self._range_delete_count = count
                            self._input_mode = "range_confirm"
                            self._input_buffer = ""
                            # Show preview: navigate to start of range
                            if self.index != start_idx:
                                img_path, img = self._navigate_to(start_idx, img_path)
                        else:
                            print("  Invalid range. Use frame#-frame# or filename-filename")
                            self._input_mode = None
                            self._input_buffer = ""
                    elif self._input_mode == "range_confirm":
                        answer = self._input_buffer.strip().lower()
                        if answer == 'y' or answer == 'yes':
                            result = self.delete_range(
                                self._range_delete_start,
                                self._range_delete_end
                            )
                            if result is None:
                                print("  All images deleted. Exiting.")
                                break
                            img_path, img = result
                        else:
                            print("  Range delete cancelled.")
                        self._input_mode = None
                        self._input_buffer = ""
                elif key == 8:  # Backspace
                    self._input_buffer = self._input_buffer[:-1]
                elif 32 <= key <= 126:  # Printable ASCII
                    self._input_buffer += chr(key)
                continue  # Skip normal hotkeys while in input mode

            if key == ord('q') or key == 113:
                if self.modified:
                    self.save_labels(img_path, self.boxes)
                if self.preds_modified:
                    self.save_pred_labels(img_path)
                self._save_progress()
                break

            elif key == 27:  # ESC — return to view mode
                self.mode = "view"
                self.drawing = False
                self._show_info = False
                self._show_help = False

            elif key == ord('a'):
                self.mode = "add"

            elif key == ord('t'):  # Toggle thumbnail grid
                self._thumbnail_mode = True
                self._thumb_page = self.index // self._thumb_per_page

            elif key == ord('d'):
                self.mode = "delete"

            elif key == 27:  # ESC
                self.mode = "view"
                self.drawing = False

            elif key == ord('s'):
                self.save_labels(img_path, self.boxes)
                if img_path.stem not in self.progress["reviewed"]:
                    self.progress["reviewed"].append(img_path.stem)
                self.modified = False
                self.reviewed_count += 1
                self._save_progress()
                if self.index < len(self.images) - 1:
                    img_path, img = self._navigate_to(self.index + 1, img_path)

            elif key == ord('w') and self.show_preds:
                self.mode = "promote"

            elif key == ord('W') and self.show_preds:
                if self.preds:
                    self.promote_all_preds()

            elif key == ord('x') and self.show_preds:
                # Swap GT with one prediction at a time
                if self.preds:
                    self._push_undo()
                    pred = self.preds.pop(0)
                    cls, xc, yc, w, h = pred[:5]
                    self.boxes = [(cls, xc, yc, w, h)]  # replace GT
                    self.modified = True
                    self.preds_modified = True

            elif key == ord('n'):
                self._push_undo()
                self.boxes = []
                self.modified = True
                if self.show_preds:
                    self.preds = []
                    self.preds_modified = True

            elif key == ord('D'):  # Shift+D = delete image + label from dataset
                result = self.delete_image_and_label(img_path)
                if result is None:
                    print("  All images deleted. Exiting.")
                    break
                img_path, img = result

            elif key == ord('R'):  # Shift+R = delete range of images + labels
                self._input_mode = "range_delete"
                self._input_buffer = ""
                self._range_delete_start = 0
                self._range_delete_end = 0
                self._range_delete_count = 0

            # ── Ctrl+Z: Undo ──
            elif key == 26:  # Ctrl+Z
                self._undo()

            elif key == ord('b'):
                self.thickness_idx = (self.thickness_idx + 1) % len(self.thickness_levels)
                self.line_thickness = self.thickness_levels[self.thickness_idx]

            elif key == ord('r'):
                # Reload from disk (invalidate cache for this stem)
                stem = img_path.stem
                self._label_cache.pop(stem, None)
                self._pred_cache.pop(stem, None)
                self.boxes = self.load_labels(img_path)
                self.preds = self.load_pred_labels(img_path) if self.show_preds else []
                self.modified = False
                self.preds_modified = False

            elif key == ord('h'):
                for j in range(self.index + 1, len(self.images)):
                    check_path = self.images[j]
                    check_boxes = self.load_labels(check_path)
                    if not check_boxes:
                        img_path, img = self._navigate_to(j, img_path)
                        break

            elif key == ord('m') and self.show_preds:
                for j in range(self.index + 1, len(self.images)):
                    check_path = self.images[j]
                    check_gt = self.load_labels(check_path)
                    check_pred = self.load_pred_labels(check_path)
                    if is_mismatch(check_gt, check_pred):
                        img_path, img = self._navigate_to(j, img_path)
                        break

            elif key == ord('M') and self.show_preds:
                for j in range(self.index - 1, -1, -1):
                    check_path = self.images[j]
                    check_gt = self.load_labels(check_path)
                    check_pred = self.load_pred_labels(check_path)
                    if is_mismatch(check_gt, check_pred):
                        img_path, img = self._navigate_to(j, img_path)
                        break

            # ── V3: Jump-to-index ──
            elif key == ord('g'):
                self._input_mode = "goto"
                self._input_buffer = ""

            # ── V3: Search by filename ──
            elif key == ord('/'):
                self._input_mode = "search"
                self._input_buffer = ""

            # ── V3: Info panel toggle ──
            elif key == ord('i'):
                self._show_info = not self._show_info
                self._show_help = False

            # ── V3: Help overlay toggle ──
            elif key == ord('?'):
                self._show_help = not self._show_help
                self._show_info = False

            # ── V3: Number keys 1-9 → jump to source ──
            elif ord('1') <= key <= ord('9'):
                src_idx = key - ord('1')  # 0-based
                if src_idx < len(self._sources):
                    _, _, target = self._sources[src_idx]
                    img_path, img = self._navigate_to(target, img_path)
                    self._show_info = False

            # ── V3: Jump to start / end ──
            elif key == ord('G'):
                if self.index < len(self.images) - 1:
                    img_path, img = self._navigate_to(len(self.images) - 1, img_path)

            elif key == ord('v'):  # Next group
                if self.group_by_video and self.video_names:
                    cur_vname = self._group_of_idx[self.index]
                    cur_vid_idx = self.video_names.index(cur_vname)
                    if cur_vid_idx < len(self.video_names) - 1:
                        next_vid = self.video_names[cur_vid_idx + 1]
                        img_path, img = self._navigate_to(self.video_start_idx[next_vid], img_path)

            elif key == ord('V'):  # Previous group
                if self.group_by_video and self.video_names:
                    cur_vname = self._group_of_idx[self.index]
                    cur_vid_idx = self.video_names.index(cur_vname)
                    if cur_vid_idx > 0:
                        prev_vid = self.video_names[cur_vid_idx - 1]
                        img_path, img = self._navigate_to(self.video_start_idx[prev_vid], img_path)

            # ── V3: Fast skip — PageUp/PageDown = 50 frames ──
            elif key == 2162688 or key == 73:  # PageUp
                target = max(0, self.index - 50)
                if target != self.index:
                    img_path, img = self._navigate_to(target, img_path)

            elif key == 2228224 or key == 81:  # PageDown
                target = min(len(self.images) - 1, self.index + 50)
                if target != self.index:
                    img_path, img = self._navigate_to(target, img_path)

            elif key == 2555904 or key == 83 or key == ord('.'):  # Right arrow
                if self.index < len(self.images) - 1:
                    img_path, img = self._navigate_to(self.index + 1, img_path)

            elif key == 2424832 or key == 81 or key == ord(','):  # Left arrow
                if self.index > 0:
                    img_path, img = self._navigate_to(self.index - 1, img_path)

            # ── V3: Shift+arrow = skip 5 frames ──
            elif key == ord('>'):  # Shift+.
                target = min(len(self.images) - 1, self.index + 5)
                if target != self.index:
                    img_path, img = self._navigate_to(target, img_path)

            elif key == ord('<'):  # Shift+,
                target = max(0, self.index - 5)
                if target != self.index:
                    img_path, img = self._navigate_to(target, img_path)

        cv2.destroyAllWindows()
        print(f"\n  Review complete. Reviewed {self.reviewed_count} frames this session.")
        print(f"  Total reviewed: {len(self.progress.get('reviewed', []))}/{len(self.images)}")


# Legacy compatibility function
def _legacy_parse_video_name(filename: str) -> tuple:
    """Extract video name and frame number from 'VideoName_f000123.png'."""
    match = re.match(r'^(.+)_f(\d+)$', Path(filename).stem)
    if match:
        return match.group(1), int(match.group(2))
    return Path(filename).stem, 0
