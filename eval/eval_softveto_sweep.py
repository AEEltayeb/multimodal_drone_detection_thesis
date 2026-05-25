"""
eval_softveto_sweep.py — Quick threshold sweep for the soft-veto classifier
rule, ablating baseline RGB + IR (native + grayscale) + classifier + both
filters across antiuav, svanstrom, and drone-detection video tests.

Soft-veto rule (per check.txt diagnostic):
  - If RGB has at least one detection:
      keep RGB unless classifier prob(reject_both) >= threshold
  - Else (RGB missed):
      if classifier trusts IR modality (argmax in {2, 3}) -> use IR dets
      else -> []

Datasets and scoring:
  - antiuav (paired, IoU @ 0.5)
  - svanstrom (paired, IoP @ 0.5)
  - drone_video_tests (RGB-only drone clips, IoP @ 0.5)

Each dataset is subsampled to ~1000 frames via uniform stride.

Output:
  docs/analysis/full_pipeline_ablations/softveto_ablation.md
  docs/analysis/full_pipeline_ablations/csv/softveto_ablation.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import score_per_size, score_trust_aware  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402
from det_cache import DetCache  # noqa: E402
from eval_full_pipeline_persize import (  # noqa: E402
    DATASETS, RGB_MODELS, IR_WEIGHTS,
    CLASSIFIERS, build_clf_features, load_classifier, get_patch,
    enumerate_video_clips, precision, recall, f1, list_frames,
)


DOC_ROOT = REPO / "docs" / "analysis" / "full_pipeline_ablations"
CSV_DIR = DOC_ROOT / "csv"

THRESHOLDS = [0.5, 0.7, 0.85, 0.95, 0.99]
N_TARGET_FRAMES = 1000
DRONE_CLIP_KEYS = [c["key"] for c in enumerate_video_clips()
                   if c["has_drone_gt"]]


def find_ds(key: str):
    for d in DATASETS:
        if d["key"] == key:
            return d
    return None


def soft_veto_effective_label(rgb_dets, ir_dets, probs, threshold) -> int:
    """probs = [p_reject, p_rgb, p_ir, p_both].

    Returns the effective trust label (0/1/2/3) that the soft-veto rule emits:
      - rgb_dets non-empty + P(reject) < τ -> label 1 (trust RGB)
      - rgb_dets non-empty + P(reject) ≥ τ -> label 0 (reject)
      - rgb_dets empty + argmax ∈ {2, 3} -> the corresponding label (IR-trust)
      - otherwise -> label 0 (reject)

    Downstream scoring uses trust-aware semantics (see metrics.score_trust_aware).
    """
    p_reject = float(probs[0])
    argmax = int(np.argmax(probs))
    if rgb_dets:
        return 0 if p_reject >= threshold else 1
    if argmax in (2, 3) and ir_dets:
        return argmax
    return 0


def score(dets, gts, w, h, rule):
    """Returns (TP, FP, FN) by summing per-size buckets for the requested rule."""
    s = score_per_size(dets, gts, w, h, iop_thr=0.5)[rule]
    tp = sum(s[b]["tp"] for b in s)
    fp = sum(s[b]["fp"] for b in s)
    fn = sum(s[b]["fn"] for b in s)
    return tp, fp, fn


def score_ta(label, rgb_dets, ir_dets, gts, ir_gts, w, h, iw, ih, is_paired, rule):
    """Trust-aware sum across size buckets."""
    s = score_trust_aware(label, rgb_dets, ir_dets, gts, ir_gts,
                          w, h, iw, ih, is_paired, rule=rule)
    tp = sum(s[b]["tp"] for b in s)
    fp = sum(s[b]["fp"] for b in s)
    fn = sum(s[b]["fn"] for b in s)
    return tp, fp, fn


def run_one_dataset(ds, n_target: int, args, classifier, feat_cols,
                    yolo_rgb, yolo_ir, det_cache,
                    patch_rgb, patch_ir, baseline_imgsz,
                    rgb_detector_key: str = "baseline"):
    """Process one dataset (paired or single-clip). Returns dict
    {stage_label: {"TP":..., "FP":..., "FN":...}}."""
    frames = list_frames(ds)
    if not frames:
        return None
    n_orig = len(frames)
    stride = max(1, n_orig // n_target)
    frames = frames[::stride]
    n = len(frames)
    is_paired = ds["type"] == "paired"
    score_rule = ds.get("scoring", "iop")
    has_gt = ds["has_drone_gt"]
    drone_cls = {ds.get("drone_class", 0)}

    print(f"[{ds['key']}] {n_orig} frames -> stride {stride} -> {n} frames "
          f"(paired={is_paired}, rule={score_rule.upper()})")

    # Stage counters
    stages = ["rgb_only", "rgb_filter", "ir_native", "ir_native_filter",
              "ir_grayscale", "ir_grayscale_filter",
              "classifier_argmax", "classifier_argmax_filter"]
    for t in THRESHOLDS:
        stages.append(f"softveto_{t}")
        stages.append(f"softveto_{t}_filter")
    counters: dict[str, dict] = {s: {"TP": 0, "FP": 0, "FN": 0} for s in stages}
    counters["__meta__"] = {"n_frames": n, "scoring": score_rule,
                            "paired": is_paired, "dataset": ds["key"]}

    t0 = time.time()
    n_done = 0
    for stem, rgb_path, ir_path, rgb_lbl, ir_lbl in frames:
        img = cv2.imread(str(rgb_path))
        if img is None: continue
        h, w = img.shape[:2]
        gray_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)
        gts = read_yolo_labels(rgb_lbl, w, h, drone_classes=drone_cls) if has_gt else []

        ir_img = None
        ir_gray = None
        ir_gts: list = []
        iw, ih = 0, 0
        if is_paired and ir_path is not None:
            ir_img = cv2.imread(str(ir_path))
            if ir_img is not None:
                ih, iw = ir_img.shape[:2]
                ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
                if ir_lbl is not None and ir_lbl.exists():
                    ir_gts = read_yolo_labels(ir_lbl, iw, ih,
                                              drone_classes=drone_cls)

        # ── Baseline RGB ──
        rgb_w_path, rgb_imgsz = RGB_MODELS[rgb_detector_key]
        cached = det_cache.get_dets(ds["key"], rgb_detector_key, rgb_w_path,
                                    rgb_imgsz, stem,
                                    ir_weights_path=IR_WEIGHTS if is_paired else None)
        if cached is not None:
            rgb_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
        else:
            res = yolo_rgb.predict(img, imgsz=rgb_imgsz, conf=args.conf,
                                   device=args.device, verbose=False)
            r0 = res[0]
            rgb_dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                rgb_dets = [(tuple(map(float, b)), float(c))
                            for b, c in zip(xyxy, confs)]
            det_cache.put_dets(ds["key"], rgb_detector_key, rgb_w_path,
                                rgb_imgsz, stem,
                                [(b[0], b[1], b[2], b[3], c) for b, c in rgb_dets])

        # ── IR native (paired only) ──
        ir_dets_native: list = []
        if is_paired and ir_img is not None:
            cached = det_cache.get_dets(ds["key"], "ir_model", IR_WEIGHTS, 640, stem)
            if cached is not None:
                ir_dets_native = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
            else:
                res = yolo_ir.predict(ir_img, imgsz=640, conf=args.ir_conf,
                                      device=args.device, verbose=False)
                r0 = res[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    ir_dets_native = [(tuple(map(float, b)), float(c))
                                      for b, c in zip(xyxy, confs)]
                det_cache.put_dets(ds["key"], "ir_model", IR_WEIGHTS, 640, stem,
                                   [(b[0], b[1], b[2], b[3], c)
                                    for b, c in ir_dets_native])

        # ── IR grayscale (always; for RGB-only it's the IR side) ──
        cached = det_cache.get_dets(ds["key"], "ir_grayscale", IR_WEIGHTS, 640, stem)
        if cached is not None:
            ir_dets_gray = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
        else:
            res = yolo_ir.predict(gray_3ch, imgsz=640, conf=args.ir_conf,
                                  device=args.device, verbose=False)
            r0 = res[0]
            ir_dets_gray = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                ir_dets_gray = [(tuple(map(float, b)), float(c))
                                for b, c in zip(xyxy, confs)]
            det_cache.put_dets(ds["key"], "ir_grayscale", IR_WEIGHTS, 640, stem,
                               [(b[0], b[1], b[2], b[3], c) for b, c in ir_dets_gray])

        # ── Patch verifier probs per detector set (one call each) ──
        def filter_dets(dets, img_in, patch_obj):
            if not dets:
                return []
            probs = patch_obj.predict_boxes(img_in, [b for b, _ in dets])
            return [d for d, p in zip(dets, probs) if p < args.patch_thr]
        rgb_kept = filter_dets(rgb_dets, img, patch_rgb)
        ir_native_kept = filter_dets(ir_dets_native, ir_img if ir_img is not None else img, patch_ir)
        ir_gray_kept = filter_dets(ir_dets_gray, gray_3ch, patch_rgb)

        # ── Classifier features + predict + predict_proba ──
        ir_for_clf = ir_dets_native if is_paired else ir_dets_gray
        ir_gray_features = ir_gray if ir_gray is not None else gray_rgb
        x = build_clf_features(rgb_dets, ir_for_clf, gray_rgb, ir_gray_features, feat_cols)
        try:
            probs = classifier.predict_proba(x)[0]
            argmax = int(np.argmax(probs))
        except Exception:
            argmax = 3
            probs = np.array([0.0, 0.0, 0.0, 1.0])

        def add(stage, dets, use_gts, uw, uh):
            tp, fp, fn = score(dets, use_gts, uw, uh, score_rule)
            counters[stage]["TP"] += tp
            counters[stage]["FP"] += fp
            counters[stage]["FN"] += fn

        def add_ta(stage, label, rgb_d, ir_d, rgb_d_filt, ir_d_filt):
            """Two writes: <stage> from raw dets, <stage>_filter from filtered."""
            tp, fp, fn = score_ta(label, rgb_d, ir_d, gts, ir_gts,
                                  w, h, iw or w, ih or h, is_paired, score_rule)
            counters[stage]["TP"] += tp
            counters[stage]["FP"] += fp
            counters[stage]["FN"] += fn
            tp, fp, fn = score_ta(label, rgb_d_filt, ir_d_filt, gts, ir_gts,
                                  w, h, iw or w, ih or h, is_paired, score_rule)
            counters[f"{stage}_filter"]["TP"] += tp
            counters[f"{stage}_filter"]["FP"] += fp
            counters[f"{stage}_filter"]["FN"] += fn

        # Non-classifier single-modality stages (raw + filtered).
        add("rgb_only", rgb_dets, gts, w, h)
        add("rgb_filter", rgb_kept, gts, w, h)
        add("ir_grayscale", ir_dets_gray, gts, w, h)
        add("ir_grayscale_filter", ir_gray_kept, gts, w, h)
        if is_paired and ir_img is not None:
            add("ir_native", ir_dets_native, ir_gts, iw, ih)
            add("ir_native_filter", ir_native_kept, ir_gts, iw, ih)

        # ── Classifier (trust-aware scoring) ──
        # Filtered versions of each side, picked by the side's appropriate patch.
        # For paired: ir uses ir_filter; for RGB-only: ir_dets came from ir_grayscale, filtered via rgb_filter.
        if is_paired and ir_img is not None:
            ir_filt_for_ta = ir_native_kept
            ir_for_ta = ir_dets_native
        else:
            ir_filt_for_ta = ir_gray_kept
            ir_for_ta = ir_dets_gray
        add_ta("classifier_argmax", argmax, rgb_dets, ir_for_ta, rgb_kept, ir_filt_for_ta)

        # Soft-veto sweep (each threshold emits an effective trust label, scored TA).
        for t in THRESHOLDS:
            eff_label = soft_veto_effective_label(rgb_dets, ir_for_ta, probs, t)
            add_ta(f"softveto_{t}", eff_label, rgb_dets, ir_for_ta, rgb_kept, ir_filt_for_ta)

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            fps = n_done / elapsed if elapsed > 0 else 0
            eta = (n - n_done) / fps if fps > 0 else 0
            print(f"  {n_done}/{n} {fps:.1f} fps ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"[{ds['key']}] done in {dt:.0f}s")
    return counters


def merge_video_clips_into_dataset(target_n_frames: int):
    """Treat all drone clips as one virtual 'drone_video_drone' dataset by
    iterating through their frame lists and tagging each frame with its clip
    key for caching."""
    out_frames = []
    for ckey in DRONE_CLIP_KEYS:
        ds = find_ds(ckey)
        if ds is None: continue
        frames = list_frames(ds)
        out_frames.append((ds, frames))
    # Flatten and pick stride
    total = sum(len(f) for _, f in out_frames)
    if total == 0:
        return None, 0
    stride = max(1, total // target_n_frames)
    return out_frames, stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--n-target", type=int, default=N_TARGET_FRAMES)
    ap.add_argument("--rgb-detector", type=str, default="baseline",
                    choices=list(RGB_MODELS.keys()),
                    help="Which RGB detector to ablate. Default baseline.")
    ap.add_argument("--datasets", nargs="+",
                    default=["antiuav", "svanstrom", "drone_video_drone"],
                    help="Subset of {antiuav, svanstrom, drone_video_drone}.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix appended to output filenames "
                         "(e.g. '_selcom_960' -> softveto_ablation_selcom_960.md).")
    args = ap.parse_args()

    cpath = CLASSIFIERS["sa32"]
    classifier, feat_cols = load_classifier(cpath)
    print(f"Loaded classifier sa32 ({len(feat_cols)} features)")

    rgb_w, rgb_sz = RGB_MODELS[args.rgb_detector]
    print(f"Using RGB detector: {args.rgb_detector} @ imgsz={rgb_sz}")
    yolo_rgb = YOLO(str(rgb_w))
    yolo_ir = YOLO(str(IR_WEIGHTS))
    patch_rgb = get_patch("rgb_filter")
    patch_ir = get_patch("ir_filter")
    det_cache = DetCache(REPO)

    # Per-dataset run
    all_results: dict[str, dict] = {}
    for ds_key in ("antiuav", "svanstrom"):
        if ds_key not in args.datasets:
            continue
        ds = find_ds(ds_key)
        res = run_one_dataset(ds, args.n_target, args, classifier, feat_cols,
                              yolo_rgb, yolo_ir, det_cache,
                              patch_rgb, patch_ir, rgb_sz,
                              rgb_detector_key=args.rgb_detector)
        if res:
            all_results[ds_key] = res
        det_cache.flush()

    # Drone-video aggregate: iterate clips and sum
    if "drone_video_drone" not in args.datasets:
        print("\nSkipping drone_video_drone (not in --datasets)")
    else:
        print("\n[drone_video_drone] aggregating across drone clips")
        drone_counters: dict[str, dict] = defaultdict(
            lambda: {"TP": 0, "FP": 0, "FN": 0})
        drone_counters["__meta__"] = {"n_frames": 0, "scoring": "iop",
                                      "paired": False, "dataset": "drone_video_drone"}
        for ckey in DRONE_CLIP_KEYS:
            ds = find_ds(ckey)
            if ds is None: continue
            res = run_one_dataset(ds, n_target=10_000_000, args=args,
                                  classifier=classifier, feat_cols=feat_cols,
                                  yolo_rgb=yolo_rgb, yolo_ir=yolo_ir,
                                  det_cache=det_cache, patch_rgb=patch_rgb,
                                  patch_ir=patch_ir, baseline_imgsz=rgb_sz,
                                  rgb_detector_key=args.rgb_detector)
            if not res: continue
            for stage, ctr in res.items():
                if stage == "__meta__":
                    drone_counters["__meta__"]["n_frames"] += ctr["n_frames"]
                    continue
                drone_counters[stage]["TP"] += ctr["TP"]
                drone_counters[stage]["FP"] += ctr["FP"]
                drone_counters[stage]["FN"] += ctr["FN"]
        all_results["drone_video_drone"] = dict(drone_counters)
        det_cache.flush()
        print(f"[drone_video_drone] n_frames={drone_counters['__meta__']['n_frames']}")

    # ── Write CSV ──
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"softveto_ablation{args.out_suffix}.csv"
    fieldnames = ["dataset", "n_frames", "scoring", "stage",
                  "TP", "FP", "FN", "P", "R", "F1"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ds_key, ctrs in all_results.items():
            meta = ctrs["__meta__"]
            for stage, c in ctrs.items():
                if stage == "__meta__": continue
                P = precision(c["TP"], c["FP"])
                R = recall(c["TP"], c["FN"])
                F = f1(P, R)
                w.writerow({
                    "dataset": ds_key, "n_frames": meta["n_frames"],
                    "scoring": meta["scoring"], "stage": stage,
                    "TP": c["TP"], "FP": c["FP"], "FN": c["FN"],
                    "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                })
    print(f"  wrote {csv_path}")

    # ── Pick best soft-veto threshold by averaged F1 across datasets ──
    f1_by_threshold: dict[float, list[float]] = defaultdict(list)
    for ds_key, ctrs in all_results.items():
        for t in THRESHOLDS:
            c = ctrs.get(f"softveto_{t}")
            if not c: continue
            P = precision(c["TP"], c["FP"])
            R = recall(c["TP"], c["FN"])
            F = f1(P, R)
            f1_by_threshold[t].append(F)
    mean_f1 = {t: float(np.mean(vs)) for t, vs in f1_by_threshold.items() if vs}
    best_t = max(mean_f1, key=mean_f1.get) if mean_f1 else 0.85

    # ── Markdown ──
    L: list[str] = []
    L.append("# Soft-Veto Threshold Sweep — baseline RGB + IR + classifier + filters")
    L.append("")
    L.append("**Scope:** ablate baseline RGB, IR (native + grayscale), sa32 classifier "
             "(argmax + soft-veto sweep), and both patch filters across three datasets. "
             f"~{args.n_target} frames per dataset via uniform stride. "
             "Single-frame stages only (no temporal).")
    L.append("")
    L.append("**Soft-veto rule:** if RGB has ≥1 detection, keep RGB unless "
             "`P(reject_both) ≥ τ`; if RGB missed and classifier argmax ∈ {IR_only, both}, "
             "fall back to IR dets.")
    L.append("")
    L.append(f"**Sweep grid:** τ ∈ {{{', '.join(str(t) for t in THRESHOLDS)}}}")
    L.append("")
    L.append("## Headline — recommended τ")
    L.append("")
    L.append(f"**τ = {best_t}** maximises mean F1 across the three datasets "
             f"(mean F1 = {mean_f1.get(best_t, 0):.4f}).")
    L.append("")
    L.append("| τ | mean F1 across datasets |")
    L.append("|---:|---:|")
    for t in THRESHOLDS:
        L.append(f"| {t} | {mean_f1.get(t, 0):.4f}{' ←' if t == best_t else ''} |")
    L.append("")

    # Per-dataset section
    for ds_key, ctrs in all_results.items():
        meta = ctrs["__meta__"]
        L.append(f"## {ds_key}")
        L.append(f"- n_frames = {meta['n_frames']}, scoring = {meta['scoring'].upper()} @ 0.5, paired = {meta['paired']}")
        L.append("")
        L.append("| Stage | TP | FP | FN | P | R | F1 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        stage_order = ["rgb_only", "rgb_filter"]
        if meta["paired"]:
            stage_order += ["ir_native", "ir_native_filter"]
        stage_order += ["ir_grayscale", "ir_grayscale_filter",
                        "classifier_argmax", "classifier_argmax_filter"]
        for t in THRESHOLDS:
            stage_order.append(f"softveto_{t}")
            stage_order.append(f"softveto_{t}_filter")
        for st in stage_order:
            c = ctrs.get(st)
            if not c: continue
            tp, fp, fn = c["TP"], c["FP"], c["FN"]
            if tp + fp + fn == 0: continue
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            mark = " ←" if st == f"softveto_{best_t}" else ""
            L.append(f"| {st}{mark} | {tp:,} | {fp:,} | {fn:,} | "
                     f"{P:.4f} | {R:.4f} | {F:.4f} |")
        L.append("")

    L.append("## Notes")
    L.append("")
    L.append("- Soft-veto only changes behaviour when the classifier's `argmax` "
             "is `reject_both` (class 0). At τ → 1.0 the rule reduces to *never veto*; "
             "at τ → 0 it reduces to argmax (and may even *force-keep* RGB).")
    L.append("- The IR side fed to the classifier is the native IR detector on paired "
             "datasets, and `ir_grayscale` on RGB-only datasets (matches the production "
             "PySide pipeline's `_process_grayscale` mode).")
    L.append("- `*_filter` rows apply the rgb_filter patch verifier on top of the kept "
             "detections (matches the deployed cascade).")
    L.append("")

    md_path = DOC_ROOT / f"softveto_ablation{args.out_suffix}.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {md_path}")


if __name__ == "__main__":
    main()
