"""
render_example_images.py — composite example PNGs for the dashboard.

For each dataset emits ONE PNG with multi-panel layout showing both a
*happy* path (pipeline does the right thing) and a *sad* path (pipeline
misses or slips), where each scenario exists.

Color legend (every panel):
  - cyan dashed  : ground-truth drone box
  - yellow solid : raw detector box (selcom for RGB / ir_v3b for IR)
  - lime solid   : boxes that survive the full pipeline alert gate
                    (classifier kept + patch verifier passed)
  - red dashed   : raw boxes that the classifier/patch rejected

Caches used (consistent with eval_1000_results.md):
  antiuav, svanstrom: selcom_1280 @ imgsz=960
       eval/results/detector_eval/selcom_1280_960imgsz_detections.json (flat,
       multi-dataset cache keyed by stem)
       eval/results/detector_eval/ir_v3b_detections.json (paired IR)
  drone_video clips: docs/.../cache/video_*_selcom_960_sz960.json
                     docs/.../cache/video_*_ir_grayscale_sz640.json
  rgb_test:         docs/.../cache/rgb_test_selcom_960_sz960.json + ir_grayscale_sz640.json
  ir_test:          docs/.../cache/ir_test_selcom_960_sz960.json + ir_native_sz640.json

Outputs: docs/analysis/full_pipeline_ablations/plots/*.png
Run after the eval scripts have populated caches:
  python eval/render_example_images.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from datasets import read_yolo_labels  # noqa: E402
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES  # noqa: E402

CACHE = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
DETEVAL_CACHE = REPO / "eval" / "results" / "detector_eval"
PLOTS = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
PATCH_RGB_PATH = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
PATCH_IR_PATH  = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"
SOFTVETO_TAU = 0.95
PATCH_THR = 0.70
MAX_SCAN = 600   # candidate stems to scan per category looking for happy+sad


# ── Cache loaders (handle both file formats) ─────────────────────────

def _load_singlepass_cache(path: Path) -> dict:
    """docs/.../cache/<>.json format: {'dets': {stem: [(x1,y1,x2,y2,conf)...]}}"""
    if not path.exists(): return {}
    return json.loads(path.read_text(encoding="utf-8")).get("dets", {})


def _load_flat_cache(path: Path) -> dict:
    """eval_detector.py JSON format: flat {stem: [[x1,y1,x2,y2,conf]...]}"""
    if not path.exists(): return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_dets(raw) -> list:
    if not raw: return []
    return [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]


# ── Classifier features + soft-veto helper ──────────────────────────

def _build_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    feats = {}
    for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": float(max(confs)),
                          f"{prefix}_mean_conf": float(np.mean(confs))})
    feats.update({f"rgb_{k}": v for k, v in compute_global_features(rgb_gray).items()})
    feats.update({f"ir_{k}": v for k, v in compute_global_features(ir_gray).items()})
    rh, rw = rgb_gray.shape[:2]; ih, iw = ir_gray.shape[:2]
    for prefix, dets, gray, gw, gh in (
        ("rgb", rgb_dets, rgb_gray, rw, rh),
        ("ir",  ir_dets,  ir_gray,  iw, ih),
    ):
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    return np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)


def _softveto_label(rgb_dets, ir_dets, probs, tau=SOFTVETO_TAU):
    p_reject = float(probs[0]); argmax = int(np.argmax(probs))
    if rgb_dets:
        return 0 if p_reject >= tau else 1
    if argmax in (2, 3) and ir_dets: return argmax
    return 0


def _route(label, rgb_dets, ir_dets):
    if label == 0: return []
    if label == 1: return rgb_dets
    if label == 2: return ir_dets
    return rgb_dets + ir_dets


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


# ── Pipeline applied to a single frame ───────────────────────────────

def apply_pipeline(rgb_dets, ir_dets, rgb_gray, ir_gray,
                   classifier, feat_cols, mode: str,
                   img_for_patch, patch_verifier):
    """Returns (raw, final_after_alert_gate, dropped) lists of (box, conf)."""
    x = _build_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols)
    try:
        probs = classifier.predict_proba(x)[0]
        argmax = int(np.argmax(probs))
    except Exception:
        probs = np.array([0.0, 0.0, 0.0, 1.0]); argmax = 3
    if mode == "softveto":
        label = _softveto_label(rgb_dets, ir_dets, probs)
    else:
        label = argmax
    kept_clf = _route(label, rgb_dets, ir_dets)
    if kept_clf:
        probs_p = patch_verifier.predict_boxes(img_for_patch, [b for b, _ in kept_clf])
        final = [d for d, p in zip(kept_clf, probs_p) if p < PATCH_THR]
    else:
        final = []
    final_ids = {id(d) for d in final}
    dropped = [d for d in (rgb_dets + ir_dets) if id(d) not in final_ids]
    return rgb_dets + ir_dets, final, dropped


# ── Drawing ─────────────────────────────────────────────────────────

def _draw_box(ax, box, color, ls='-', lw=2):
    x1, y1, x2, y2 = box
    r = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=lw,
                   edgecolor=color, facecolor='none', linestyle=ls)
    ax.add_patch(r)


def render_panel(ax, img_bgr, gts, raw_dets, final_dets, dropped_dets, title: str):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    for g in gts: _draw_box(ax, g, color='cyan', ls='--', lw=2)
    for d, _c in raw_dets: _draw_box(ax, d, color='yellow', ls='-', lw=1.5)
    for d, _c in dropped_dets: _draw_box(ax, d, color='red', ls='--', lw=1.5)
    for d, _c in final_dets: _draw_box(ax, d, color='lime', ls='-', lw=2.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def find_image(img_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists(): return p
    return None


# ── Outcome classification per stem ─────────────────────────────────

def classify_outcome(gts, raw_dets, final_dets, has_gt: bool) -> str:
    """Return one of: drone_happy, drone_sad, conf_happy, conf_sad, uninteresting."""
    if has_gt:
        # drone-positive frame
        for g in gts:
            for d, _c in final_dets:
                if _iou(g, d) >= 0.3:
                    return "drone_happy"
        # final didn't cover any GT
        if any(_iou(g, d) >= 0.3 for g in gts for d, _c in raw_dets):
            return "drone_sad"   # raw saw it, pipeline killed it
        # raw also missed it → uninteresting (just hard frame)
        return "uninteresting"
    # drone-negative (confuser) frame
    if not raw_dets:
        return "uninteresting"
    if not final_dets:
        return "conf_happy"     # pipeline killed all FPs
    return "conf_sad"           # at least one FP slipped through


# ── Frame search: pick happy and sad per category ────────────────────

def find_happy_sad(stems: list, rgb_cache: dict, ir_cache: dict,
                    img_dir: Path, lbl_dir: Path,
                    classifier, feat_cols, mode: str,
                    patch_verifier_alert, ir_image_loader=None,
                    pair_ir_gray=None, max_scan=MAX_SCAN):
    """Iterate stems, run pipeline, find one happy and one sad.

    `ir_image_loader(stem) -> Path or None`: where to fetch the IR image to use
        as classifier IR feature source. If None, falls back to grayscale of RGB.
    Returns dict {"happy": {...}, "sad": {...}} with at most one each.
    """
    found = {}
    for stem in stems[:max_scan]:
        if "happy" in found and "sad" in found:
            break
        rgb_dets = _to_dets(rgb_cache.get(stem, []))
        ir_dets = _to_dets(ir_cache.get(stem, []))
        if not rgb_dets and not ir_dets:
            continue
        img_path = find_image(img_dir, stem)
        if img_path is None: continue
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        rgb_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if ir_image_loader:
            ir_path = ir_image_loader(stem)
            ir_img = cv2.imread(str(ir_path)) if ir_path else None
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if ir_img is not None else rgb_gray
        else:
            ir_gray = rgb_gray
        lbl_path = lbl_dir / f"{stem}.txt"
        gts = (read_yolo_labels(lbl_path, w, h, drone_classes={0})
               if (lbl_path.exists() and lbl_path.stat().st_size > 0) else [])
        has_gt = bool(gts)
        raw, final, dropped = apply_pipeline(
            rgb_dets, ir_dets, rgb_gray, ir_gray,
            classifier, feat_cols, mode, img, patch_verifier_alert)
        # Dedup dropped from final
        final_ids = {id(d) for d in final}
        dropped = [d for d in dropped if id(d) not in final_ids]
        outcome = classify_outcome(gts, raw, final, has_gt)
        if outcome in ("drone_happy", "conf_happy") and "happy" not in found:
            found["happy"] = {"stem": stem, "img_path": img_path, "gts": gts,
                               "raw": raw, "final": final, "dropped": dropped}
        elif outcome in ("drone_sad", "conf_sad") and "sad" not in found:
            found["sad"] = {"stem": stem, "img_path": img_path, "gts": gts,
                             "raw": raw, "final": final, "dropped": dropped}
    return found


# ── Composite rendering ──────────────────────────────────────────────

def render_composite(name: str, panels: list[dict], mode: str):
    """panels: list of {"title": str, "img": np.ndarray BGR, "gts": list, "raw": list,
                         "final": list, "dropped": list}"""
    n = len(panels)
    if n == 0:
        print(f"  [{name}] no panels — skipping"); return
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2), squeeze=False)
    axes = axes.flatten()
    for i, p in enumerate(panels):
        ax = axes[i]
        render_panel(ax, p["img"], p["gts"], p["raw"], p["final"], p["dropped"], p["title"])
    for j in range(n, len(axes)):
        axes[j].axis('off')
    legend_handles = [
        plt.Line2D([0], [0], color='cyan', linestyle='--', lw=2, label='GT'),
        plt.Line2D([0], [0], color='yellow', lw=1.5, label='Raw detector'),
        plt.Line2D([0], [0], color='lime', lw=2.5, label='After alert gate'),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Dropped'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{name} — example frames ({mode} classifier + alert gate)", fontsize=11)
    plt.tight_layout()
    out = PLOTS / f"{name}_examples.png"
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")


# ── Paired (Anti-UAV / Svanström) ───────────────────────────────────

def _make_panel(found_entry, title_suffix, prefix=""):
    if not found_entry: return None
    return {"title": f"{prefix}{title_suffix}",
            "img": cv2.imread(str(found_entry["img_path"])),
            "gts": found_entry["gts"],
            "raw": [d for d in found_entry["raw"]
                    if id(d) not in {id(f) for f in found_entry["final"]}
                    and id(d) not in {id(f) for f in found_entry["dropped"]}],
            "final": found_entry["final"], "dropped": found_entry["dropped"]}


def render_paired(name: str, info: dict, classifier, feat_cols, patch_rgb, patch_ir):
    """Anti-UAV / Svanström: real RGB+IR, classifier=argmax.

    Panels per side:
      - RGB drone happy + sad (when present)
      - IR drone happy + sad
      - Svanström: RGB bird, airplane, helicopter confusers (happy/sad each)
    """
    # Use the per-dataset singlepass cache (selcom_1280 at imgsz=1280). The flat
    # eval-detector cache at 960 only contains antiuav stems on disk; the
    # per-dataset cache covers both antiuav and svanstrom and is keyed by stem.
    rgb_cache = _load_singlepass_cache(CACHE / f"{name}_selcom_1280_sz1280.json")
    # IR cache: docs/.../cache/<>_ir_model_sz640.json — keyed by RGB stem
    ir_cache = _load_singlepass_cache(CACHE / f"{name}_ir_model_sz640.json")

    img_dir = info["rgb_img"]; lbl_dir = info["rgb_lbl"]
    ir_img_dir = info["ir_img"]; ir_lbl_dir = info["ir_lbl"]
    rgb_suf = info["rgb_suffix"]; ir_suf = info["ir_suffix"]

    panels = []

    # Per-dataset cache: every stem belongs to this dataset, skip the existence pre-filter.
    dataset_stems = sorted(rgb_cache.keys())
    # Within those, find drone (GT non-empty) vs confusers (GT empty + raw det in RGB)

    drone_stems, conf_stems = [], {}
    for s in dataset_stems:
        lbl = lbl_dir / f"{s}.txt"
        has_gt = lbl.exists() and lbl.stat().st_size > 0
        if has_gt:
            drone_stems.append(s)
        else:
            # Categorise by keyword for svanstrom only
            cat = "other"
            stem_u = s.upper()
            if "BIRD" in stem_u: cat = "bird"
            elif "AIRPLANE" in stem_u: cat = "airplane"
            elif "HELICOPTER" in stem_u or "HELI" in stem_u: cat = "helicopter"
            conf_stems.setdefault(cat, []).append(s)

    def ir_loader(stem):
        ir_stem = stem.replace(rgb_suf, ir_suf)
        return find_image(ir_img_dir, ir_stem)

    # RGB drone happy + sad
    found = find_happy_sad(drone_stems, rgb_cache, ir_cache, img_dir, lbl_dir,
                            classifier, feat_cols, "argmax", patch_rgb, ir_loader)
    if found.get("happy"):
        panels.append(_make_panel(found["happy"], f"{name} RGB — drone (happy)"))
    if found.get("sad"):
        panels.append(_make_panel(found["sad"], f"{name} RGB — drone (sad: pipeline dropped)"))

    # IR drone happy + sad (use IR image, IR dets, IR labels)
    # Note: ir_cache is keyed by RGB stem; IR image/label use IR stem.
    ir_panels_done = 0
    for category, found_key in (("happy", "happy"), ("sad", "sad")):
        # For IR-side panel scan, iterate the same drone_stems but evaluate IR-detector only.
        # We don't need the classifier for an IR-only display panel.
        # Pick the first stem with: IR raw dets exist + cyan GT exists in IR.
        chosen = None
        for s in drone_stems:
            ir_dets = _to_dets(ir_cache.get(s, []))
            if not ir_dets: continue
            ir_stem = s.replace(rgb_suf, ir_suf)
            ir_img_path = find_image(ir_img_dir, ir_stem)
            ir_lbl_path = ir_lbl_dir / f"{ir_stem}.txt"
            if ir_img_path is None: continue
            ir_img = cv2.imread(str(ir_img_path))
            if ir_img is None: continue
            ih, iw = ir_img.shape[:2]
            ir_gts = (read_yolo_labels(ir_lbl_path, iw, ih, drone_classes={0})
                      if ir_lbl_path.exists() and ir_lbl_path.stat().st_size > 0 else [])
            # Apply patch verifier (ir_filter) at the alert boundary
            probs_p = patch_ir.predict_boxes(ir_img, [b for b, _ in ir_dets])
            final = [d for d, p in zip(ir_dets, probs_p) if p < PATCH_THR]
            dropped = [d for d in ir_dets if id(d) not in {id(f) for f in final}]
            covered = any(_iou(g, d) >= 0.3 for g in ir_gts for d, _ in final)
            raw_covered = any(_iou(g, d) >= 0.3 for g in ir_gts for d, _ in ir_dets)
            if found_key == "happy" and covered and ir_panels_done == 0:
                chosen = (ir_img_path, ir_gts, ir_dets, final, dropped); break
            if found_key == "sad" and raw_covered and not covered and ir_panels_done == 1:
                chosen = (ir_img_path, ir_gts, ir_dets, final, dropped); break
        if chosen:
            ir_img_path, ir_gts, raw_ir, final_ir, dropped_ir = chosen
            raw_for_panel = [d for d in raw_ir
                              if id(d) not in {id(f) for f in final_ir}
                              and id(d) not in {id(f) for f in dropped_ir}]
            panels.append({"title": f"{name} IR — drone ({found_key})",
                            "img": cv2.imread(str(ir_img_path)),
                            "gts": ir_gts, "raw": raw_for_panel,
                            "final": final_ir, "dropped": dropped_ir})
            ir_panels_done += 1

    # Confusers for Svanström only
    if name == "svanstrom":
        for cat in ("bird", "airplane", "helicopter"):
            stems = conf_stems.get(cat, [])
            if not stems: continue
            found = find_happy_sad(stems, rgb_cache, ir_cache, img_dir, lbl_dir,
                                    classifier, feat_cols, "argmax", patch_rgb, ir_loader)
            if found.get("happy"):
                panels.append(_make_panel(found["happy"], f"svanstrom RGB — {cat} (happy)"))
            if found.get("sad"):
                panels.append(_make_panel(found["sad"], f"svanstrom RGB — {cat} (slipped)"))

    render_composite(name, panels, "argmax")


# ── RGB test (RGB-only mixed) + IR test (IR-primary mixed) ──────────

def render_test_split(name: str, info: dict, classifier, feat_cols, patch_rgb, patch_ir):
    rgb_cache = _load_singlepass_cache(CACHE / info["rgb_cache"])
    ir_cache = _load_singlepass_cache(CACHE / info["ir_cache"])
    img_dir = info["rgb_img"]; lbl_dir = info["rgb_lbl"]
    mode = info["mode"]
    panels = []

    all_stems = list(rgb_cache.keys())

    drone_stems, conf_stems = [], {}
    for s in all_stems:
        lbl = lbl_dir / f"{s}.txt"
        has_gt = lbl.exists() and lbl.stat().st_size > 0
        sl = s.lower()
        if has_gt:
            drone_stems.append(s)
        else:
            cat = ("bird" if "bird" in sl else
                   "airplane" if "airplane" in sl else
                   "helicopter" if "helicopter" in sl else None)
            if cat: conf_stems.setdefault(cat, []).append(s)

    patch_for_alert = patch_ir if name == "ir_test" else patch_rgb

    found = find_happy_sad(drone_stems, rgb_cache, ir_cache, img_dir, lbl_dir,
                            classifier, feat_cols, mode, patch_for_alert)
    if found.get("happy"):
        panels.append(_make_panel(found["happy"], f"{name} — drone (happy)"))
    if found.get("sad"):
        panels.append(_make_panel(found["sad"], f"{name} — drone (sad)"))

    for cat in ("bird", "airplane", "helicopter"):
        stems = conf_stems.get(cat, [])
        if not stems: continue
        found = find_happy_sad(stems, rgb_cache, ir_cache, img_dir, lbl_dir,
                                classifier, feat_cols, mode, patch_for_alert)
        if found.get("happy"):
            panels.append(_make_panel(found["happy"], f"{name} — {cat} (suppressed)"))
        if found.get("sad"):
            panels.append(_make_panel(found["sad"], f"{name} — {cat} (slipped)"))

    # IR-grayscale view for rgb_test only (ir_test image is already IR)
    if name == "rgb_test":
        # Take an existing panel's stem and render the grayscale version overlayed with
        # the ir_grayscale detector boxes (cached in ir_cache).
        for p in panels[:1]:  # use first panel's stem
            # NOTE: we need stem from the original found entry; encode via title isn't ideal.
            pass
        # Simpler: pick a drone_happy stem and render its grayscale variant.
        if drone_stems:
            stem = drone_stems[0]
            img_path = find_image(img_dir, stem)
            if img_path is not None:
                img = cv2.imread(str(img_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ir_dets = _to_dets(ir_cache.get(stem, []))
                # Apply patch (rgb_filter on grayscale-RGB image)
                if ir_dets:
                    probs_p = patch_rgb.predict_boxes(gray3, [b for b, _ in ir_dets])
                    final = [d for d, p_ in zip(ir_dets, probs_p) if p_ < PATCH_THR]
                else:
                    final = []
                dropped = [d for d in ir_dets if id(d) not in {id(f) for f in final}]
                h, w = img.shape[:2]
                gts = read_yolo_labels(lbl_dir / f"{stem}.txt", w, h, drone_classes={0})
                raw_for_panel = [d for d in ir_dets
                                  if id(d) not in {id(f) for f in final}
                                  and id(d) not in {id(f) for f in dropped}]
                panels.append({"title": "rgb_test — IR-grayscale view (what the IR branch sees)",
                                "img": gray3, "gts": gts, "raw": raw_for_panel,
                                "final": final, "dropped": dropped})

    render_composite(name, panels, mode)


# ── Drone-video composite ───────────────────────────────────────────

def render_drone_video(classifier, feat_cols, patch_rgb, patch_ir):
    ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
    panels = []

    # Drone clip happy + sad — use drone_and_bird_sky_and_trees_short
    clip = "drone_and_bird_sky_and_trees_short"
    img_dir = ROOT / "drone" / clip / "images" / "test"
    lbl_dir = ROOT / "drone" / clip / "labels" / "test"
    rgb_cache = _load_singlepass_cache(CACHE / f"video_drone_{clip}_selcom_960_sz960.json")
    ir_cache = _load_singlepass_cache(CACHE / f"video_drone_{clip}_ir_grayscale_sz640.json")
    stems = sorted(rgb_cache.keys())
    found = find_happy_sad(stems, rgb_cache, ir_cache, img_dir, lbl_dir,
                            classifier, feat_cols, "softveto", patch_rgb)
    if found.get("happy"):
        panels.append(_make_panel(found["happy"], f"drone clip — happy ({clip[:30]})"))
    if found.get("sad"):
        panels.append(_make_panel(found["sad"], f"drone clip — sad ({clip[:30]})"))

    # Confuser categories: birds, airplanes, helicopters
    CONF_CLIPS = {
        "birds": "birds_in_slow_motion_flying_various_sizes_compilation",
        "airplanes": "airplanes_compilation",
        "helicopters": "helicopter_compilation",
    }
    for cat, clip_name in CONF_CLIPS.items():
        c_img_dir = ROOT / cat / clip_name / "images" / "test"
        c_lbl_dir = ROOT / cat / clip_name / "labels" / "test"
        c_rgb_cache = _load_singlepass_cache(
            CACHE / f"video_{cat}_{clip_name}_selcom_960_sz960.json")
        c_ir_cache = _load_singlepass_cache(
            CACHE / f"video_{cat}_{clip_name}_ir_grayscale_sz640.json")
        c_stems = sorted(c_rgb_cache.keys())
        found = find_happy_sad(c_stems, c_rgb_cache, c_ir_cache, c_img_dir, c_lbl_dir,
                                classifier, feat_cols, "softveto", patch_rgb)
        if found.get("happy"):
            panels.append(_make_panel(found["happy"], f"{cat} — suppressed (happy)"))
        if found.get("sad"):
            panels.append(_make_panel(found["sad"], f"{cat} — slipped (sad)"))

    # IR-grayscale view from the drone clip
    if panels:
        # Use the drone clip's first stem with raw dets to show what ir_grayscale sees.
        stems_with_ir = [s for s in stems if _to_dets(ir_cache.get(s, []))]
        if stems_with_ir:
            stem = stems_with_ir[0]
            img_path = find_image(img_dir, stem)
            if img_path is not None:
                img = cv2.imread(str(img_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ir_dets = _to_dets(ir_cache.get(stem, []))
                if ir_dets:
                    probs_p = patch_rgb.predict_boxes(gray3, [b for b, _ in ir_dets])
                    final = [d for d, p_ in zip(ir_dets, probs_p) if p_ < PATCH_THR]
                else:
                    final = []
                dropped = [d for d in ir_dets if id(d) not in {id(f) for f in final}]
                h, w = img.shape[:2]
                gts = (read_yolo_labels(lbl_dir / f"{stem}.txt", w, h, drone_classes={0})
                       if (lbl_dir / f"{stem}.txt").exists() else [])
                raw_for_panel = [d for d in ir_dets
                                  if id(d) not in {id(f) for f in final}
                                  and id(d) not in {id(f) for f in dropped}]
                panels.append({"title": "drone clip — IR-grayscale view",
                                "img": gray3, "gts": gts, "raw": raw_for_panel,
                                "final": final, "dropped": dropped})

    render_composite("drone_video", panels, "softveto")


# ── Main ─────────────────────────────────────────────────────────────

DATASETS_INFO = {
    "antiuav": {
        "rgb_img":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "rgb_lbl":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_img":   Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "ir_lbl":   Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
    },
    "svanstrom": {
        "rgb_img":  Path("G:/drone/svanstrom_paired/RGB/images"),
        "rgb_lbl":  Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_img":   Path("G:/drone/svanstrom_paired/IR/images"),
        "ir_lbl":   Path("G:/drone/svanstrom_paired/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
    },
    "rgb_test": {
        "rgb_cache": "rgb_test_selcom_960_sz960.json",
        "ir_cache":  "rgb_test_ir_grayscale_sz640.json",
        "rgb_img":  Path("G:/drone/dataset/dataset/images/test"),
        "rgb_lbl":  Path("G:/drone/dataset/dataset/labels/test"),
        "mode": "softveto",
    },
    "ir_test": {
        "rgb_cache": "ir_test_selcom_960_sz960.json",
        "ir_cache":  "ir_test_ir_native_sz640.json",
        "rgb_img":  Path("G:/drone/IR_dset_final/test/images"),
        "rgb_lbl":  Path("G:/drone/IR_dset_final/test/labels"),
        "mode": "argmax",
    },
}


def main():
    obj = joblib.load(str(CLASSIFIER_PATH))
    classifier = obj["model"]
    feat_cols = obj.get("features") or obj.get("feat_cols") or []
    from patch_verifier import PatchVerifier
    patch_rgb = PatchVerifier(str(PATCH_RGB_PATH))
    patch_ir = PatchVerifier(str(PATCH_IR_PATH))
    print(f"Loaded classifier ({len(feat_cols)} features), patch RGB+IR")

    for name in ("antiuav", "svanstrom"):
        print(f"\nBuilding panels for {name}...")
        render_paired(name, DATASETS_INFO[name], classifier, feat_cols, patch_rgb, patch_ir)

    for name in ("rgb_test", "ir_test"):
        print(f"\nBuilding panels for {name}...")
        render_test_split(name, DATASETS_INFO[name], classifier, feat_cols, patch_rgb, patch_ir)

    print("\nBuilding drone_video panels...")
    render_drone_video(classifier, feat_cols, patch_rgb, patch_ir)


if __name__ == "__main__":
    main()
