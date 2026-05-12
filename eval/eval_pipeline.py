"""
eval_pipeline.py — Unified pipeline evaluation.

Evaluates any combination of: YOLO detection, fusion classifier,
confuser filter, across all datasets (Anti-UAV, Svanström, YouTube).

Usage:
    python eval/eval_pipeline.py --dataset antiuav
    python eval/eval_pipeline.py --dataset svanstrom --stride 3
    python eval/eval_pipeline.py --dataset youtube_ir --stride 3
    python eval/eval_pipeline.py --dataset both --plot
    python eval/eval_pipeline.py --yolo-only --dataset antiuav
    python eval/eval_pipeline.py --classifier-only --dataset antiuav
    python eval/eval_pipeline.py --filter-only --dataset antiuav
    python eval/eval_pipeline.py --preset ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(REPO / "ir_gui"))

from metrics import (
    score_detections, score_detections_detailed, compute_prf,
    size_distribution, classify_size,
)
from datasets import (
    load_config, resolve_path, read_yolo_labels, img_from_label,
    detect_category, CachedDetectionDataset, VideoDataset, SVAN_CATS,
)
from run_manifest import write_manifest, cache_identity_tag
from reporting import (
    print_metrics_table, print_fp_by_category, print_size_distribution,
    save_metrics_csv, save_fp_category_csv, save_json, save_jsonl,
    plot_metrics_bars, plot_confusion_matrices, plot_pr_curves,
    plot_size_distribution, plot_youtube_summary,
)

# Lazy imports (heavy)
_clf_bundle = None
_patch_rgb = None
_patch_ir = None

CONFIGS_ALL = [
    "ir_only", "rgb_only", "classifier",
    "ir_filter", "rgb_filter",
    "classifier_then_filter", "filter_then_classifier",
]
RULES = ("iou", "iop")
CATEGORIES = [*SVAN_CATS, "OTHER"]


def _load_classifier(cfg):
    global _clf_bundle
    if _clf_bundle is None:
        import joblib
        path = resolve_path(cfg["classifier_path"])
        print(f"  Loading classifier: {path.name}")
        _clf_bundle = joblib.load(str(path))
    return _clf_bundle


def _load_patch_verifiers(cfg):
    global _patch_rgb, _patch_ir
    if _patch_rgb is None:
        sys.path.insert(0, str(REPO / "classifier"))
        from patch_verifier import PatchVerifier
        rgb_path = resolve_path(cfg["patch_rgb_weights"])
        ir_path = resolve_path(cfg["patch_ir_weights"])
        _patch_rgb = PatchVerifier(str(rgb_path))
        _patch_ir = PatchVerifier(str(ir_path))
        print(f"  Loaded patch verifiers: RGB={rgb_path.name}  IR={ir_path.name}")
    return _patch_rgb, _patch_ir


def reset_patch_verifiers():
    """Drop cached verifiers so the next _load_patch_verifiers re-reads from cfg."""
    global _patch_rgb, _patch_ir
    _patch_rgb = None
    _patch_ir = None


# ── Feature builder (mirrors fusion/engine.py) ───────────────────

def build_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
    """Build classifier feature dict from detections + grayscale frames."""
    from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES
    feats = {}
    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        n = len(confs)
        if n == 0:
            feats.update({f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.0,
                          f"{prefix}_mean_conf": 0.0, f"{prefix}_detected": 0})
        else:
            feats.update({f"{prefix}_n_dets": n,
                          f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
                          f"{prefix}_detected": 1})
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
    g_rgb = compute_global_features(rgb_gray)
    g_ir = compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in g_rgb.items()})
    feats.update({f"ir_{k}": v for k, v in g_ir.items()})
    for prefix, dets, gray, gw, gh in [
        ("rgb", rgb_dets, rgb_gray, rw, rh),
        ("ir", ir_dets, ir_gray, iw, ih),
    ]:
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            from fusion.features import compute_target_features
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    rd, id_ = len(rgb_dets) > 0, len(ir_dets) > 0
    feats["both_detect"] = int(rd and id_)
    feats["neither_detect"] = int(not rd and not id_)
    feats["rgb_only_detect"] = int(rd and not id_)
    feats["ir_only_detect"] = int(not rd and id_)
    return feats


# ── Config dispatch ──────────────────────────────────────────────

def apply_configs(rgb_raw, ir_raw, rgb_flt, ir_flt, clf_raw, clf_flt):
    """Return dict: config_name -> (kept_rgb, kept_ir)."""
    out = {
        "ir_only": ([], ir_raw),
        "rgb_only": (rgb_raw, []),
        "ir_filter": ([], ir_flt),
        "rgb_filter": (rgb_flt, []),
    }
    # classifier on raw dets
    kr = rgb_raw if clf_raw in (1, 3) else []
    ki = ir_raw if clf_raw in (2, 3) else []
    out["classifier"] = (kr, ki)
    # classifier -> filter
    kr = rgb_flt if clf_raw in (1, 3) else []
    ki = ir_flt if clf_raw in (2, 3) else []
    out["classifier_then_filter"] = (kr, ki)
    # filter -> classifier
    kr = rgb_flt if clf_flt in (1, 3) else []
    ki = ir_flt if clf_flt in (2, 3) else []
    out["filter_then_classifier"] = (kr, ki)
    return out


# GT scope per config (dual-modality scoring — counts both RGB and IR GT for
# every classifier frame; penalizes a silent modality even if the other detected
# the same physical drone).
GT_SCOPE = {
    "ir_only": ("ir",),
    "rgb_only": ("rgb",),
    "ir_filter": ("ir",),
    "rgb_filter": ("rgb",),
    "classifier": ("rgb", "ir"),
    "classifier_then_filter": ("rgb", "ir"),
    "filter_then_classifier": ("rgb", "ir"),
}


def trust_aware_scope(c_name, clf_raw_label, clf_flt_label):
    """Per-frame GT scope following the classifier's trust decision.
    Used when --scoring=trust_aware. Non-classifier configs unchanged.
    Labels: 0=reject_both, 1=trust_rgb, 2=trust_ir, 3=trust_both.
    """
    if c_name not in ("classifier", "classifier_then_filter", "filter_then_classifier"):
        return GT_SCOPE[c_name]
    label = clf_flt_label if c_name == "filter_then_classifier" else clf_raw_label
    if label == 1: return ("rgb",)
    if label == 2: return ("ir",)
    if label == 3: return ("rgb", "ir")
    # reject_both: count both modalities as missed (system explicitly gave up)
    return ("rgb", "ir")


# ── Paired dataset evaluation ────────────────────────────────────

def evaluate_paired(ds_name, cfg, args, active_configs):
    """Evaluate on Anti-UAV or Svanström using cached detections."""
    ds_cfg = cfg["datasets"][ds_name]
    cache_cfg = cfg["cache"]

    # Find cache file. Resolution order:
    #   1. tagged eval-side cache (if --cache-tag given)
    #   2. auto-tagged eval-side cache (weights+imgsz+stride hash)
    #   3. plain eval-side cache
    #   4. tagged legacy cache (rare)
    #   5. plain legacy cache
    # Skip tiny/empty files (< 100 bytes) — stubs from aborted runs.
    cache_key = ds_name
    legacy_key = f"legacy_{ds_name}"
    eval_path = resolve_path(cache_cfg.get(cache_key, ""))
    legacy_path = resolve_path(cache_cfg.get(legacy_key, ""))

    def _tagged(p, tag):
        return p.with_name(p.stem + f"_{tag}.json") if tag else None

    auto_tag = cache_identity_tag(
        rgb_weights=resolve_path(cfg["rgb_weights"]),
        ir_weights=resolve_path(cfg["ir_weights"]),
        imgsz=args.imgsz,
        stride=1,  # cache files are stride-1 by convention; pipeline applies stride later
    )

    candidates = []
    if args.cache_tag:
        candidates.append(_tagged(eval_path, args.cache_tag))
    # Auto-tag candidate (always tried after explicit tag)
    candidates.append(_tagged(eval_path, auto_tag))
    candidates.append(eval_path)
    if args.cache_tag:
        candidates.append(_tagged(legacy_path, args.cache_tag))
    candidates.append(legacy_path)

    cache_path = None
    for c in candidates:
        if c is None:
            continue
        if c.exists() and c.stat().st_size >= 100:
            cache_path = c
            break

    if cache_path is None:
        rgb_w = resolve_path(cfg["rgb_weights"])
        ir_w = resolve_path(cfg["ir_weights"])
        print(f"  [CACHE-MISS] No cache found for {ds_name}.")
        print(f"  [CACHE-MISS] Auto-tag tried: {auto_tag}")
        print(f"  [CACHE-MISS] Run:")
        print(f"    python eval/cache_inference.py --dataset {ds_name} "
              f"--imgsz {args.imgsz} "
              f'--rgb-weights "{rgb_w}" --ir-weights "{ir_w}"')
        return

    root = Path(ds_cfg["root"])
    rgb_img_dir = root / ds_cfg.get("rgb_images", "RGB/images")
    ir_img_dir = root / ds_cfg.get("ir_images", "IR/images")

    out_dir = Path(args.output_dir) / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provenance manifest (written before heavy work so crashes still leave a record)
    write_manifest(
        out_dir=out_dir,
        args=args,
        cfg=cfg,
        weights_paths={
            "rgb_weights": resolve_path(cfg.get("rgb_weights", "")),
            "ir_weights": resolve_path(cfg.get("ir_weights", "")),
            "classifier_path": resolve_path(cfg.get("classifier_path", "")),
            "patch_rgb_weights": resolve_path(cfg.get("patch_rgb_weights", "")),
            "patch_ir_weights": resolve_path(cfg.get("patch_ir_weights", "")),
        },
        cache_paths=[cache_path],
        extra={"dataset": ds_name, "stage": "eval_pipeline.evaluate_paired",
               "scoring": getattr(args, "scoring", "dual")},
    )

    # Load components as needed
    need_clf = any(c in active_configs for c in
                   ["classifier", "classifier_then_filter", "filter_then_classifier"])
    need_flt = any(c in active_configs for c in
                   ["ir_filter", "rgb_filter", "classifier_then_filter",
                    "filter_then_classifier"])

    clf_model = clf_feats = None
    if need_clf:
        bundle = _load_classifier(cfg)
        clf_model = bundle["model"]
        clf_feats = bundle["features"]

    p_rgb = p_ir = None
    if need_flt:
        p_rgb, p_ir = _load_patch_verifiers(cfg)

    # Load cache
    print(f"[{ds_name}] Loading cache: {cache_path.name}")
    raw_data = json.loads(cache_path.read_text())
    keys = sorted(raw_data.keys())
    if args.stride > 1:
        keys = keys[::args.stride]
    if args.limit:
        keys = keys[:args.limit]
    print(f"[{ds_name}] {len(keys):,} frame pairs")

    # Counters
    counters = {rule: {c: {"tp": 0, "fp": 0, "fn": 0} for c in active_configs}
                for rule in RULES}
    fp_by_cat = {rule: {c: {cat: 0 for cat in CATEGORIES} for c in active_configs}
                 for rule in RULES}
    size_dist = {c: {"small": 0, "medium": 0, "large": 0} for c in active_configs}
    perdet_buf = []
    patch_cache = {}

    # Try loading existing patch cache
    patch_path = out_dir / "patch_probs.json"
    if patch_path.exists():
        try:
            patch_cache = json.loads(patch_path.read_text())
        except Exception:
            pass

    t0 = time.time()
    ckpt_every = 500

    for idx, key in enumerate(keys):
        entry = raw_data[key]
        rgb_dets_all = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["rgb_dets"]]
        ir_dets_all = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["ir_dets"]]
        rgb_raw = [d for d in rgb_dets_all if d[1] >= args.rgb_conf]
        ir_raw = [d for d in ir_dets_all if d[1] >= args.ir_conf]
        rw, rh = entry["rgb_w"], entry["rgb_h"]
        iw, ih = entry["ir_w"], entry["ir_h"]
        rgb_gt = read_yolo_labels(Path(entry["rgb_lbl"]), rw, rh)
        ir_gt = read_yolo_labels(Path(entry["ir_lbl"]), iw, ih)

        # Filter
        rgb_flt, ir_flt = rgb_raw, ir_raw
        rgb_probs_all, ir_probs_all = [], []
        if need_flt:
            rgb_path = img_from_label(Path(entry["rgb_lbl"]), rgb_img_dir)
            ir_path = img_from_label(Path(entry["ir_lbl"]), ir_img_dir)
            if rgb_path is None or ir_path is None:
                continue
            rgb_img = cv2.imread(str(rgb_path))
            ir_img = cv2.imread(str(ir_path))
            if rgb_img is None or ir_img is None:
                continue

            cached_p = patch_cache.get(key)
            if cached_p is None:
                rgb_probs_all = (p_rgb.predict_boxes(rgb_img, [d[0] for d in rgb_dets_all]).tolist()
                                 if rgb_dets_all else [])
                ir_probs_all = (p_ir.predict_boxes(ir_img, [d[0] for d in ir_dets_all]).tolist()
                                if ir_dets_all else [])
                patch_cache[key] = {"rgb": rgb_probs_all, "ir": ir_probs_all}
            else:
                rgb_probs_all = cached_p["rgb"]
                ir_probs_all = cached_p["ir"]
            rgb_probs = [p for d, p in zip(rgb_dets_all, rgb_probs_all) if d[1] >= args.rgb_conf]
            ir_probs = [p for d, p in zip(ir_dets_all, ir_probs_all) if d[1] >= args.ir_conf]
            rgb_flt = [d for d, p in zip(rgb_raw, rgb_probs) if p < args.patch_thr]
            ir_flt = [d for d, p in zip(ir_raw, ir_probs) if p < args.patch_thr]

        # Classifier
        clf_raw_label = clf_flt_label = 3
        if need_clf:
            rgb_path = img_from_label(Path(entry["rgb_lbl"]), rgb_img_dir)
            ir_path = img_from_label(Path(entry["ir_lbl"]), ir_img_dir)
            if rgb_path is None or ir_path is None:
                continue
            rgb_img = cv2.imread(str(rgb_path))
            ir_img = cv2.imread(str(ir_path))
            if rgb_img is None or ir_img is None:
                continue
            rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
            feats_raw = build_features(rgb_raw, ir_raw, rgb_gray, ir_gray)
            feats_flt = build_features(rgb_flt, ir_flt, rgb_gray, ir_gray)
            x_raw = np.array([[feats_raw.get(c, 0) for c in clf_feats]], dtype=np.float32)
            x_flt = np.array([[feats_flt.get(c, 0) for c in clf_feats]], dtype=np.float32)
            clf_raw_label = int(clf_model.predict(x_raw)[0])
            clf_flt_label = int(clf_model.predict(x_flt)[0])

        configs = apply_configs(rgb_raw, ir_raw, rgb_flt, ir_flt,
                                clf_raw_label, clf_flt_label)
        cat = detect_category(key)

        for c_name in active_configs:
            if c_name not in configs:
                continue
            kr, ki = configs[c_name]
            if args.scoring == "trust_aware":
                scope = trust_aware_scope(c_name, clf_raw_label, clf_flt_label)
            else:
                scope = GT_SCOPE[c_name]
            # Size distribution
            for d in kr:
                size_dist[c_name][classify_size(d[0], rw, rh)] += 1
            for d in ki:
                size_dist[c_name][classify_size(d[0], iw, ih)] += 1

            for rule in RULES:
                tp = fp = fn = 0
                iou_t, iop_t = args.iou_thr, args.iop_thr
                if "rgb" in scope:
                    t, f, n = score_detections(kr, rgb_gt, rule=rule,
                                               iou_thr=iou_t, iop_thr=iop_t)
                    tp += t; fp += f; fn += n
                else:
                    t, f, _ = score_detections(kr, [], rule=rule)
                    fp += f
                if "ir" in scope:
                    t, f, n = score_detections(ki, ir_gt, rule=rule,
                                               iou_thr=iou_t, iop_thr=iop_t)
                    tp += t; fp += f; fn += n
                else:
                    t, f, _ = score_detections(ki, [], rule=rule)
                    fp += f
                counters[rule][c_name]["tp"] += tp
                counters[rule][c_name]["fp"] += fp
                counters[rule][c_name]["fn"] += fn
                fp_by_cat[rule][c_name][cat] += fp

        # Per-det records for PR curves
        if rgb_dets_all or ir_dets_all:
            rgb_det_detailed = score_detections_detailed(rgb_dets_all, rgb_gt,
                                                         args.iou_thr, args.iop_thr)
            ir_det_detailed = score_detections_detailed(ir_dets_all, ir_gt,
                                                        args.iou_thr, args.iop_thr)
            rec = {
                "key": key,
                "rgb": [[d["conf"], rgb_probs_all[i] if i < len(rgb_probs_all) else 0,
                         d["matched_iou"], d["matched_iop"]]
                        for i, d in enumerate(rgb_det_detailed)],
                "ir": [[d["conf"], ir_probs_all[i] if i < len(ir_probs_all) else 0,
                        d["matched_iou"], d["matched_iop"]]
                       for i, d in enumerate(ir_det_detailed)],
                "rgb_n_gt": len(rgb_gt), "ir_n_gt": len(ir_gt),
                "clf_raw": clf_raw_label, "clf_flt": clf_flt_label,
            }
            perdet_buf.append(json.dumps(rec))

        if (idx + 1) % ckpt_every == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            remaining = (len(keys) - idx - 1) / max(fps, 1e-6)
            print(f"  [{ds_name}] {idx + 1:>6,}/{len(keys):,}  "
                  f"{fps:.1f} fps  ETA {remaining / 60:.1f} min")

    # Save patch cache
    if patch_cache:
        patch_path.write_text(json.dumps(patch_cache))

    # Save per-det JSONL
    if perdet_buf:
        save_jsonl(perdet_buf, out_dir / "per_det.jsonl")

    # Compute and output metrics
    for rule in RULES:
        rows = []
        for c_name in active_configs:
            c = counters[rule][c_name]
            row = compute_prf(c["tp"], c["fp"], c["fn"])
            row["config"] = c_name
            rows.append(row)
        print_metrics_table(rows, f"[{ds_name}] RESULTS ({rule.upper()} match)")
        save_metrics_csv(rows, out_dir / f"metrics_{rule}.csv")
        print_fp_by_category(fp_by_cat[rule], active_configs, CATEGORIES,
                             f"[{ds_name}] FP by category ({rule.upper()})")
        save_fp_category_csv(fp_by_cat[rule], active_configs, CATEGORIES,
                             out_dir / f"fp_by_category_{rule}.csv")
        if args.plot:
            plot_metrics_bars(rows, out_dir, f"{ds_name} [{rule.upper()}]",
                              suffix=f"_{rule}")
            plot_confusion_matrices(rows, out_dir, f"{ds_name} [{rule.upper()}]",
                                     suffix=f"_{rule}")

    print_size_distribution(size_dist, f"[{ds_name}] Detection size distribution")
    if args.plot:
        plot_size_distribution(size_dist, out_dir, ds_name)
        # PR curves from per-det
        perdet_path = out_dir / "per_det.jsonl"
        if perdet_path.exists():
            rgb_recs, ir_recs = [], []
            n_rgb_gt = n_ir_gt = 0
            for ln in perdet_path.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                rgb_recs.extend(r["rgb"])
                ir_recs.extend(r["ir"])
                n_rgb_gt += r["rgb_n_gt"]
                n_ir_gt += r["ir_n_gt"]
            plot_pr_curves(rgb_recs, ir_recs, n_rgb_gt, n_ir_gt,
                           out_dir, ds_name, args.patch_thr)

    print(f"[{ds_name}] Done. Output: {out_dir}")


# ── YouTube video evaluation ─────────────────────────────────────

def evaluate_youtube(ds_name, cfg, args):
    """Evaluate filter on YouTube OOD videos."""
    ds_cfg = cfg["datasets"][ds_name]
    vds = VideoDataset(ds_cfg)
    available = vds.available_videos()
    if not available:
        print(f"  [SKIP] No videos found for {ds_name}")
        return

    out_dir = Path(args.output_dir) / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    modality = ds_cfg.get("modality", "ir")
    print(f"\n[{ds_name}] {len(available)} videos, modality={modality}")

    # Load model + verifier
    from ultralytics import YOLO
    weights_key = "ir_weights" if modality == "ir" else "rgb_weights"
    model = YOLO(str(resolve_path(cfg[weights_key])))
    conf_thr = args.ir_conf if modality == "ir" else args.rgb_conf

    # Provenance
    write_manifest(
        out_dir=out_dir,
        args=args,
        cfg=cfg,
        weights_paths={
            weights_key: resolve_path(cfg[weights_key]),
            "patch_rgb_weights": resolve_path(cfg.get("patch_rgb_weights", "")),
            "patch_ir_weights": resolve_path(cfg.get("patch_ir_weights", "")),
        },
        extra={"dataset": ds_name, "stage": "eval_pipeline.evaluate_youtube",
               "modality": modality},
    )

    sys.path.insert(0, str(REPO / "classifier"))
    from patch_verifier import PatchVerifier
    pv_key = "patch_ir_weights" if modality == "ir" else "patch_rgb_weights"
    verifier = PatchVerifier(str(resolve_path(cfg[pv_key])))

    results = []
    t0 = time.time()

    for vi, vinfo in enumerate(available):
        fname = vinfo["filename"]
        cat = vinfo["category"]
        fpath = vinfo["path"]

        raw_frames = 0
        raw_dets = 0
        filt_frames = 0
        filt_dets = 0
        processed = 0

        for frame_idx, frame in vds.iter_frames(fpath, args.stride, args.limit or 0):
            processed += 1
            res = model.predict(frame, conf=conf_thr, verbose=False, imgsz=args.imgsz)
            boxes = res[0].boxes
            n_raw = len(boxes)
            raw_dets += n_raw
            if n_raw > 0:
                raw_frames += 1
            n_surv = 0
            if n_raw > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                probs = verifier.predict_boxes(frame, xyxy)
                n_surv = int((probs < args.patch_thr).sum())
                filt_dets += n_surv
                if n_surv > 0:
                    filt_frames += 1

        r = {
            "video": fname, "category": cat,
            "quality": vinfo.get("quality", ""),
            "frames": processed,
            "raw_det_frames": raw_frames, "raw_dets": raw_dets,
            "raw_det_rate": raw_frames / max(processed, 1),
            "filter_det_frames": filt_frames, "filter_dets": filt_dets,
            "filter_det_rate": filt_frames / max(processed, 1),
            "suppression": 1.0 - (filt_frames / max(raw_frames, 1)) if raw_frames > 0 else 0.0,
        }
        results.append(r)
        print(f"  [{vi + 1}/{len(available)}] {cat:12s} {fname:35s} "
              f"frames={processed:5d}  raw={r['raw_det_rate']:.1%}  "
              f"filt={r['filter_det_rate']:.1%}  supp={r['suppression']:.1%}")

    # Summary
    import csv
    csv_path = out_dir / "per_video.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_path}")
    if args.plot:
        plot_youtube_summary(results, out_dir)
    print(f"[{ds_name}] Done. Output: {out_dir}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Unified pipeline evaluation")
    ap.add_argument("--dataset", default="both",
                    choices=["antiuav", "svanstrom", "youtube_ir", "youtube_rgb",
                             "both", "all"],
                    help="Dataset to evaluate (both=antiuav+svanstrom, all=everything)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="Max frames (0=all)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Configs to evaluate (default: all)")
    ap.add_argument("--yolo-only", action="store_true",
                    help="Only ir_only + rgb_only")
    ap.add_argument("--classifier-only", action="store_true",
                    help="Only classifier configs")
    ap.add_argument("--filter-only", action="store_true",
                    help="Only filter configs")
    ap.add_argument("--plot", action="store_true", help="Generate all plots")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--iop-thr", type=float, default=0.5)
    ap.add_argument("--rgb-conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--patch-rgb-weights", type=str, default="",
                    help="Override cfg.patch_rgb_weights for this run")
    ap.add_argument("--patch-ir-weights", type=str, default="",
                    help="Override cfg.patch_ir_weights for this run")
    ap.add_argument("--classifier-path", type=str, default="",
                    help="Override cfg.classifier_path for this run")
    ap.add_argument("--scoring", choices=["dual", "trust_aware"], default="dual",
                    help="Scoring rule for classifier configs. "
                         "'dual' counts both modalities' GT every frame (current/strict). "
                         "'trust_aware' counts only the modality the classifier trusted (email rule).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-tag", type=str, default="",
                    help="Cache suffix e.g. 'v3more'")
    ap.add_argument("--cached-only", action="store_true",
                    help="Skip any dataset that requires live YOLO inference (no cache)")
    ap.add_argument("--output-dir", type=str,
                    default=str(EVAL_DIR / "results"))
    ap.add_argument("--preset", choices=["ablation", "full"],
                    help="Run a preset evaluation suite")
    args = ap.parse_args()

    cfg = load_config()

    # CLI overrides for patch verifier weights — useful for A/B/C comparisons.
    # Accept paths relative to repo root (natural call site) OR eval/ dir OR absolute.
    def _resolve_override(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        # Try repo-root-relative first (matches user shell cwd convention),
        # then eval-dir-relative (matches config.yaml convention).
        for base in (REPO, EVAL_DIR):
            candidate = (base / pp).resolve()
            if candidate.exists():
                return str(candidate)
        # Fall back to as-given so the existing error path reports the bad value.
        return p

    if args.patch_rgb_weights:
        cfg["patch_rgb_weights"] = _resolve_override(args.patch_rgb_weights)
    if args.patch_ir_weights:
        cfg["patch_ir_weights"] = _resolve_override(args.patch_ir_weights)
    if args.classifier_path:
        cfg["classifier_path"] = _resolve_override(args.classifier_path)
        global _clf_bundle
        _clf_bundle = None  # force re-load
    reset_patch_verifiers()  # ensure overrides take effect even within one process

    # Determine active configs
    if args.configs:
        active = args.configs
    elif args.yolo_only:
        active = ["ir_only", "rgb_only"]
    elif args.classifier_only:
        active = ["classifier", "classifier_then_filter", "filter_then_classifier"]
    elif args.filter_only:
        active = ["ir_filter", "rgb_filter"]
    else:
        active = CONFIGS_ALL

    # Presets
    if args.preset == "full":
        args.dataset = "all"
        args.plot = True
    elif args.preset == "ablation":
        args.dataset = "all"
        args.plot = True

    print(f"[eval_pipeline] configs={active}  stride={args.stride}  "
          f"iou={args.iou_thr}  iop={args.iop_thr}  "
          f"rgb_conf={args.rgb_conf}  ir_conf={args.ir_conf}")

    # Dispatch
    paired_ds = []
    if args.dataset in ("antiuav", "both", "all"):
        paired_ds.append("antiuav")
    if args.dataset in ("svanstrom", "both", "all"):
        paired_ds.append("svanstrom")

    for ds in paired_ds:
        evaluate_paired(ds, cfg, args, active)

    if args.dataset in ("youtube_ir", "all"):
        if args.cached_only:
            print("\n[youtube_ir] SKIP — requires live inference (no cache). Run cache_inference.py first.")
        else:
            evaluate_youtube("youtube_ir", cfg, args)
    if args.dataset in ("youtube_rgb", "all"):
        if args.cached_only:
            print("\n[youtube_rgb] SKIP — requires live inference (no cache). Run cache_inference.py first.")
        else:
            evaluate_youtube("youtube_rgb", cfg, args)

    print("\n[eval_pipeline] All done.")


if __name__ == "__main__":
    main()
