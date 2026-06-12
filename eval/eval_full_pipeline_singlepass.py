"""
eval_full_pipeline_singlepass.py — Single-pass multi-detector runner.

For one dataset, processes each frame ONCE and scores all (detector x
classifier) combos in a single loop. Writes per-combo summary.csv files in
the same format as eval_full_pipeline_persize.py so the aggregator
(analytics/spec_analysis/10_full_pipeline_doc.py) works unchanged.

Why: the per-combo runner re-imreads each frame, re-computes global
classifier features, and re-runs the patch verifier once per combo. With 8
combos that's ~8x redundant per-frame work. This single-pass version pays
each per-frame cost ONCE and forks only at the (detector-dependent) bbox
matching and classifier-decision steps.

Usage:
  python eval/eval_full_pipeline_singlepass.py --dataset antiuav \\
      --rgb-detectors baseline retrained_v2 selcom_1280 \\
      --ir-detectors ir_model ir_grayscale \\
      --classifiers sa32

Outputs:
  eval/results/full_pipeline_persize/<dataset>/<detector>/<classifier>/summary.csv
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

from metrics import SIZE_BUCKETS, classify_size, score_per_size, score_trust_aware  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402
from det_cache import DetCache  # noqa: E402

# Reuse the per-detector / classifier / dataset catalogues from the
# per-combo runner. We want the SAME weights paths, sizes, and per-dataset
# scoring rule.
from eval_full_pipeline_persize import (  # noqa: E402
    DATASETS, RGB_MODELS, IR_WEIGHTS, CLASSIFIERS,
    auto_stride, get_patch, build_clf_features, load_classifier,
    precision, recall, f1, list_frames,
)


# ── Combo packing ───────────────────────────────────────────────────

def find_dataset(key: str) -> dict:
    for d in DATASETS:
        if d["key"] == key:
            return d
    raise SystemExit(f"unknown dataset {key}")


def build_combos(rgb_detectors: list[str], ir_detectors: list[str],
                 classifiers: list[str]) -> list[tuple]:
    """Return list of (detector_key, weights, imgsz, modality, patch_filter, classifier_key).

    Rules:
      - RGB detectors run with every requested classifier (none + sa32 etc.)
      - IR detectors only run with classifier=none (their classifier rows
        are dropped by the aggregator since the canonical classifier row is
        attributed to the RGB detector)
    """
    combos = []
    for k in rgb_detectors:
        w, sz = RGB_MODELS[k]
        for ck in classifiers:
            combos.append((k, w, sz, "rgb", "rgb_filter", ck))
    for k in ir_detectors:
        if k == "ir_model":
            combos.append((k, IR_WEIGHTS, 640, "ir", "ir_filter", None))
        elif k == "ir_grayscale":
            combos.append((k, IR_WEIGHTS, 640, "ir_grayscale", "rgb_filter", None))
        else:
            raise SystemExit(f"unknown IR detector {k}")
    return combos


# ── Counters ────────────────────────────────────────────────────────

def _empty_counts() -> dict:
    return {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}


def _add(into: dict, by_size: dict):
    for b, s in by_size.items():
        into[b]["tp"] += s["tp"]
        into[b]["fp"] += s["fp"]
        into[b]["fn"] += s["fn"]


# ── Per-detector inference (with cache) ─────────────────────────────

def detect_or_cache(yolo_pool: dict, det_cache: DetCache, dataset_key: str,
                    detector_key: str, weights_path: Path, imgsz: int,
                    image_bgr: np.ndarray, stem: str, conf: float, device: str,
                    ir_weights_for_paired: Path | None = None) -> list[tuple]:
    """Return [((x1,y1,x2,y2), conf), ...] for one (detector, frame).

    Looks in det_cache first; if miss, runs YOLO and writes back to cache.
    """
    cached = det_cache.get_dets(dataset_key, detector_key, weights_path, imgsz,
                                stem, ir_weights_path=ir_weights_for_paired)
    if cached is not None:
        return [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
    if detector_key not in yolo_pool:
        yolo_pool[detector_key] = YOLO(str(weights_path))
    res = yolo_pool[detector_key].predict(image_bgr, imgsz=imgsz, conf=conf,
                                          device=device, verbose=False)
    r0 = res[0]
    out: list[tuple] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        out = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
    flat = [(b[0], b[1], b[2], b[3], c) for (b, c) in out]
    det_cache.put_dets(dataset_key, detector_key, weights_path, imgsz, stem, flat)
    return out


# ── Per-combo writer ────────────────────────────────────────────────

def write_combo_csv(out_root: Path, ds_key: str, det_key: str, clf_key: str,
                    counts: dict, seg_rows: list[dict], n_frames: int,
                    score_rule: str) -> Path:
    out_dir = out_root / ds_key / det_key / (clf_key or "no_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.csv"
    stages_in_order = [
        "S0_detector", "S1_+classifier", "S2_+classifier+patch",
        "S3_+patch_only",
    ]
    rows: list[dict] = []
    for st in stages_in_order:
        if st not in counts:
            continue
        for b in SIZE_BUCKETS:
            c = counts[st][b]
            tp, fp, fn, n_gt = c["tp"], c["fp"], c["fn"], c["n_gt"]
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            rows.append({
                "dataset": ds_key, "detector": det_key,
                "classifier": clf_key or "none",
                "stage": st, "size_bucket": b, "scoring": score_rule,
                "TP": tp, "FP": fp, "FN": fn, "TN": "",
                "n_gt": n_gt, "n_frames": n_frames,
                "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
                "fppi": round(fp / n_frames, 4) if n_frames else 0.0,
            })
    rows.extend(seg_rows)
    fieldnames = ["dataset","detector","classifier","stage","size_bucket","scoring",
                  "TP","FP","FN","TN","n_gt","n_frames",
                  "precision","recall","f1","fppi"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return out_path


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rgb-detectors", nargs="*",
                    default=["baseline", "retrained_v2", "selcom_1280"])
    ap.add_argument("--ir-detectors", nargs="*",
                    default=["ir_model", "ir_grayscale"])
    ap.add_argument("--classifiers", nargs="*", default=["sa32"],
                    help="Classifier keys to evaluate. 'none' / 'no_classifier' "
                         "is implicit (every detector always emits a no-clf "
                         "combo).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--stride-cap", type=int, default=5000,
                    help="Auto-stride cap (frames per combo). Lower = faster.")
    ap.add_argument("--output-dir", type=str,
                    default=str(REPO / "docs" / "analysis" /
                                "full_pipeline_ablations" / "raw_results"))
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    ds = find_dataset(args.dataset)
    score_rule = ds.get("scoring", "iop")
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # ir_grayscale is the cross-modal fallback for RGB-only datasets. On
    # paired datasets we have real IR data, so it's noise and must be
    # excluded unless the user explicitly insists by passing it as the ONLY
    # IR detector.
    if ds["type"] == "paired" and "ir_grayscale" in args.ir_detectors \
            and "ir_model" in args.ir_detectors:
        print("  Dropping ir_grayscale on paired dataset (use --ir-detectors "
              "ir_grayscale alone to force-include)")
        args.ir_detectors = [d for d in args.ir_detectors if d != "ir_grayscale"]

    # Validate
    for k in args.rgb_detectors:
        if k not in RGB_MODELS:
            raise SystemExit(f"unknown RGB detector {k}")
        if not RGB_MODELS[k][0].exists():
            raise SystemExit(f"missing weights for {k}: {RGB_MODELS[k][0]}")
    if not IR_WEIGHTS.exists():
        raise SystemExit(f"missing IR weights: {IR_WEIGHTS}")

    # Load classifiers
    classifier_objs: dict[str | None, tuple] = {None: (None, None)}
    for ck in args.classifiers:
        if ck in (None, "none", "no_classifier"):
            continue
        if ck not in CLASSIFIERS:
            raise SystemExit(f"unknown classifier {ck}")
        cpath = CLASSIFIERS[ck]
        if not cpath.exists():
            raise SystemExit(f"missing classifier weights {cpath}")
        model, feats = load_classifier(cpath)
        classifier_objs[ck] = (model, feats)
        print(f"  Loaded classifier {ck}: {len(feats)} features")

    # Build combo list
    combos = build_combos(args.rgb_detectors, args.ir_detectors,
                           list(classifier_objs.keys()))
    # Per-combo state
    combo_keys = [(c[0], c[5]) for c in combos]
    counters = {k: {} for k in combo_keys}
    # Per-combo per-frame booleans for segment voting
    fired_raw = {k: [] for k in combo_keys}
    fired_post = {k: [] for k in combo_keys}
    gt_present_seq: list[bool] = []     # RGB-side GT (used by rgb / ir_grayscale combos)
    ir_gt_present_seq: list[bool] = []  # IR-side GT (used by ir_model combo)

    # Skip existing unless --redo
    keep_combos = []
    for c in combos:
        det_key, _, _, _, _, clf_key = c
        out_dir = out_root / ds["key"] / det_key / (clf_key or "no_classifier")
        if (out_dir / "summary.csv").exists() and not args.redo:
            print(f"  SKIP (have summary) {det_key}/{clf_key or 'no_classifier'}")
            continue
        keep_combos.append(c)
    combos = keep_combos
    if not combos:
        print("Nothing to do.")
        return

    # Initialize counters per combo
    for det_key, _, _, _, _, clf_key in combos:
        ck = clf_key
        c = counters[(det_key, ck)]
        c["S0_detector"] = _empty_counts()
        c["S3_+patch_only"] = _empty_counts()
        if ck is not None:
            c["S1_+classifier"] = _empty_counts()
            c["S2_+classifier+patch"] = _empty_counts()

    # Group combos by detector_key so per-frame we iterate detectors once.
    by_detector: dict[str, list[tuple]] = defaultdict(list)
    for c in combos:
        by_detector[c[0]].append(c)

    # Cache pool
    det_cache = DetCache(REPO)
    yolo_pool: dict[str, YOLO] = {}

    # Load patch verifiers up front
    patch_rgb = get_patch("rgb_filter")
    patch_ir = get_patch("ir_filter")

    # Frame iteration
    frames = list_frames(ds)
    if not frames:
        raise SystemExit(f"no frames for dataset {ds['key']}")
    n_total_orig = len(frames)
    stride = auto_stride(n_total_orig, cap=args.stride_cap)
    if stride > 1:
        frames = frames[::stride]
    n_total = len(frames)
    has_gt = ds["has_drone_gt"]
    drone_cls = {ds.get("drone_class", 0)}
    is_paired = ds["type"] == "paired"
    is_sequence = ds.get("is_sequence", False) or ds["key"].startswith("video_")

    print(f"Dataset {ds['key']}: {n_total_orig} frames -> stride {stride} -> {n_total} frames")
    print(f"Combos: {len(combos)} (detectors: {sorted(by_detector.keys())}, "
          f"classifiers: {sorted({c[5] for c in combos}, key=lambda x: x or '')})")
    print(f"Scoring: {score_rule.upper()} @ 0.5")

    t0 = time.time()
    n_done = 0
    for stem, rgb_path, ir_path, rgb_lbl, ir_lbl in frames:
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_yolo_labels(rgb_lbl, w, h, drone_classes=drone_cls) if has_gt else []
        gray_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Paired datasets: also load IR GT in IR coord space for native-IR
        # detector scoring. RGB/IR cameras have different FOVs, so IR boxes
        # do NOT map onto RGB GT. (ir_grayscale runs on a grayscale copy of
        # the RGB frame, so it stays in RGB coord space and uses gts above.)
        ir_gts: list = []
        iw, ih = 0, 0

        # IR side (only if paired and we either need it for ir_model rows or
        # for any classifier combo)
        need_ir_native = ("ir_model" in by_detector) or any(
            ck is not None for _, _, _, _, _, ck in combos
            if _ == _ or True  # we need ir_dets for classifier on paired data
        )
        ir_img = None
        ir_gray = None
        if is_paired and ir_path is not None:
            ir_img = cv2.imread(str(ir_path))
            if ir_img is not None:
                ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
                ih, iw = ir_img.shape[:2]
                if ir_lbl is not None and ir_lbl.exists():
                    ir_gts = read_yolo_labels(ir_lbl, iw, ih, drone_classes=drone_cls)

        # Run each detector once. dets_per[detector_key] = list of (box, conf).
        dets_per: dict[str, list[tuple]] = {}
        patch_probs_per: dict[str, np.ndarray] = {}
        for det_key, group in by_detector.items():
            _, w_path, det_imgsz, mod, patch_kind, _ = group[0]
            if mod == "rgb":
                this_conf = args.conf
                d = detect_or_cache(yolo_pool, det_cache, ds["key"], det_key,
                                    w_path, det_imgsz, img, stem, this_conf,
                                    args.device,
                                    ir_weights_for_paired=IR_WEIGHTS if is_paired else None)
                inp_for_patch = img
                patch_obj = patch_rgb
            elif mod == "ir":
                this_conf = args.ir_conf
                inp = ir_img if ir_img is not None else img
                d = detect_or_cache(yolo_pool, det_cache, ds["key"], det_key,
                                    w_path, det_imgsz, inp, stem, this_conf,
                                    args.device)
                inp_for_patch = inp
                patch_obj = patch_ir
            elif mod == "ir_grayscale":
                this_conf = args.ir_conf
                gray3 = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)
                d = detect_or_cache(yolo_pool, det_cache, ds["key"], det_key,
                                    w_path, det_imgsz, gray3, stem, this_conf,
                                    args.device)
                inp_for_patch = gray3
                patch_obj = patch_rgb
            else:
                raise SystemExit(f"bad modality {mod}")
            dets_per[det_key] = d
            # Patch verifier once per detector
            if d:
                probs = patch_obj.predict_boxes(inp_for_patch, [b for b, _ in d])
            else:
                probs = np.zeros(0, dtype=np.float32)
            patch_probs_per[det_key] = probs

        # IR-side dets for classifier features. Priority:
        #   1. Paired data with native IR detector ("ir_model" in scope) -> use
        #      its dets.
        #   2. RGB-only data with cross-modal fallback ("ir_grayscale" in scope)
        #      -> use its dets. This mirrors gui/fusion/pipeline.py
        #      _process_grayscale (production grayscale mode), where the IR
        #      detector runs on a grayscale copy of the RGB frame and provides
        #      the IR-side detections to the classifier.
        #   3. Paired but ir_model not in scope -> run IR YOLO on the IR frame
        #      and cache as "ir_native".
        #   4. Otherwise empty (no classifier IR signal available).
        ir_dets: list[tuple] = []
        if "ir_model" in dets_per:
            ir_dets = dets_per["ir_model"]
        elif "ir_grayscale" in dets_per:
            ir_dets = dets_per["ir_grayscale"]
        elif is_paired and any(clf is not None for _, clf in combo_keys):
            inp = ir_img if ir_img is not None else img
            ir_dets = detect_or_cache(yolo_pool, det_cache, ds["key"],
                                      "ir_native", IR_WEIGHTS, 640, inp, stem,
                                      args.ir_conf, args.device)

        # Increment per-stage n_gt for every combo (per size of GT).
        # IR-native combos count IR-side GT (different coord space / camera);
        # all other combos count RGB GT.
        def _gt_for(det_key2: str, mod2: str):
            if mod2 == "ir":
                return ir_gts, iw or w, ih or h
            return gts, w, h
        for (combo_det, _), stages_d in counters.items():
            cmod = "ir" if combo_det == "ir_model" else (
                "ir_grayscale" if combo_det == "ir_grayscale" else "rgb")
            cg, cw, ch = _gt_for(combo_det, cmod)
            for stage_d in stages_d.values():
                for g in cg:
                    stage_d[classify_size(g, cw, ch)]["n_gt"] += 1

        # ── Score per combo ──
        for det_key, group in by_detector.items():
            dets = dets_per[det_key]
            probs = patch_probs_per[det_key]
            # Detector-specific GT routing (IR-native -> IR GT)
            mod_for_gt = group[0][3]  # modality
            use_gts, use_w, use_h = _gt_for(det_key, mod_for_gt)
            # S0: detector only
            s0 = score_per_size(dets, use_gts, use_w, use_h, iop_thr=0.5)[score_rule]
            # S3: + patch on this detector's dets
            kept_patch = [d for d, p in zip(dets, probs) if p < args.patch_thr] if len(dets) else []
            s3 = score_per_size(kept_patch, use_gts, use_w, use_h, iop_thr=0.5)[score_rule]

            for det_key2, _, _, mod2, patch_kind2, clf_key in group:
                combo = (det_key2, clf_key)
                _add(counters[combo]["S0_detector"], s0)
                _add(counters[combo]["S3_+patch_only"], s3)
                if is_sequence:
                    fired_raw[combo].append(len(dets) > 0)
                    fired_post[combo].append(len(kept_patch) > 0)

                if clf_key is None:
                    continue

                # Classifier features need both RGB + IR; we use this detector's
                # dets as the RGB-side input (when modality is RGB) or as
                # "trusted-only" path for IR-side detectors.
                rgb_dets_for_clf = dets if mod2 in ("rgb", "ir_grayscale") else []
                model, feat_cols = classifier_objs[clf_key]
                ir_gray_for_clf = ir_gray if ir_gray is not None else gray_rgb
                x = build_clf_features(rgb_dets_for_clf, ir_dets,
                                       gray_rgb, ir_gray_for_clf, feat_cols)
                try:
                    label = int(model.predict(x)[0])
                except Exception:
                    label = 3

                # ── Trust-aware scoring (the only rule we publish) ──
                # label=0 vetoes both modalities (both GTs become FN).
                # label=1 / label=2 / label=3 score each modality's dets
                # against its own GT and sum.
                s1 = score_trust_aware(
                    label, dets, ir_dets, gts, ir_gts,
                    w, h, iw or w, ih or h,
                    is_paired, rule=score_rule)
                _add(counters[combo]["S1_+classifier"], s1)

                # S2: + patch verifier on each side's kept dets.
                rgb_filt = dets
                ir_filt = ir_dets
                if label in (1, 3) and dets and mod2 in ("rgb", "ir_grayscale"):
                    inp_rgb = img if mod2 == "rgb" else cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)
                    p_rgb = patch_rgb.predict_boxes(inp_rgb, [b for b, _ in dets])
                    rgb_filt = [d for d, p in zip(dets, p_rgb) if p < args.patch_thr]
                if label in (2, 3) and ir_dets:
                    if is_paired and ir_img is not None:
                        p_ir = patch_ir.predict_boxes(ir_img, [b for b, _ in ir_dets])
                    else:
                        # RGB-only: ir_dets came from ir_grayscale, filter via rgb_filter
                        p_ir = patch_rgb.predict_boxes(
                            cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR),
                            [b for b, _ in ir_dets])
                    ir_filt = [d for d, p in zip(ir_dets, p_ir) if p < args.patch_thr]
                s2 = score_trust_aware(
                    label, rgb_filt, ir_filt, gts, ir_gts,
                    w, h, iw or w, ih or h,
                    is_paired, rule=score_rule)
                _add(counters[combo]["S2_+classifier+patch"], s2)

        if is_sequence:
            gt_present_seq.append(len(gts) > 0)
            ir_gt_present_seq.append(len(ir_gts) > 0)

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            fps = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / fps if fps > 0 else 0
            print(f"  {n_done}/{n_total}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"\nMain loop: {n_done} frames in {dt:.0f}s "
          f"({n_done/dt if dt else 0:.1f} fps)")

    # Segment voting (per combo)
    def seg_vote(per_frame_bool: list[bool], seg: int = 3) -> list[bool]:
        return [sum(per_frame_bool[i:i+seg]) >= 2
                for i in range(0, len(per_frame_bool), seg)]

    def seg_gt(per_frame_gt: list[bool], seg: int = 3) -> list[bool]:
        return [any(per_frame_gt[i:i+seg])
                for i in range(0, len(per_frame_gt), seg)]

    # Write outputs
    for det_key, group in by_detector.items():
        for _, _, _, _, _, clf_key in group:
            combo = (det_key, clf_key)
            seg_rows: list[dict] = []
            if is_sequence and gt_present_seq:
                # Pick the GT presence series that matches this combo's modality.
                if det_key == "ir_model" and any(ir_gt_present_seq):
                    gtl = seg_gt(ir_gt_present_seq)
                else:
                    gtl = seg_gt(gt_present_seq)
                for st_name, fire_list in (
                    ("S4_temporal_no_filter", fired_raw[combo]),
                    ("S5_alert_gate_filter", fired_post[combo]),
                ):
                    sfire = seg_vote(fire_list)
                    tp = fp = fn = tn = 0
                    for fg, gt in zip(sfire, gtl):
                        if gt and fg: tp += 1
                        elif gt and not fg: fn += 1
                        elif (not gt) and fg: fp += 1
                        else: tn += 1
                    n_seg = len(sfire)
                    P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
                    seg_rows.append({
                        "dataset": ds["key"], "detector": det_key,
                        "classifier": clf_key or "none",
                        "stage": st_name, "size_bucket": "all",
                        "scoring": score_rule,
                        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "n_gt": sum(gtl), "n_frames": n_seg,
                        "precision": round(P, 4), "recall": round(R, 4),
                        "f1": round(F, 4),
                        "fppi": round(fp / n_seg, 4) if n_seg else 0.0,
                    })
            out_path = write_combo_csv(out_root, ds["key"], det_key, clf_key,
                                       counters[combo], seg_rows, n_done,
                                       score_rule)
            print(f"  -> {out_path.relative_to(REPO).as_posix()}")

    n_flushed = det_cache.flush()
    if n_flushed:
        print(f"Flushed {n_flushed} cache file(s)")
    print(f"Cache stats: {det_cache.stats()}")


if __name__ == "__main__":
    main()
