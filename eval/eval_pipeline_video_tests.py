"""
eval_pipeline_video_tests.py — Full architecture evaluation on video test datasets.

Pipeline stages:
  1. RGB YOLO on raw frame
  2. IR YOLO on grayscale-converted frame
  3. Classifier (scene_aware_v3more_32feat) -> trust decision
  4. Temporal 2-of-3 alert gate (with spatial continuity)
  5. Patch verifier veto (only when alert fires)

Metrics:
  - Tier 1 (Detection): per-frame P/R/F1 for stages 1-3
  - Tier 2 (Alert): 3-frame segment P/R/F1 for stages 4-5

Usage:
    python eval/eval_pipeline_video_tests.py
    python eval/eval_pipeline_video_tests.py --categories drone
    python eval/eval_pipeline_video_tests.py --rgb-models baseline_trained
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import joblib
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import compute_prf, iou_iop, score_detections
from datasets import ImageDataset, read_yolo_labels
from patch_verifier import PatchVerifier
from fusion.temporal import PerModalityTemporalState
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

# ── Paths ────────────────────────────────────────────────────────
DATASET_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
PATCH_RGB = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
IR_WEIGHTS = REPO / "models" / "IR_final_cleaned" / "weights" / "best.pt"

RGB_MODELS = {
    "baseline_trained": {
        "weights": str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
        "imgsz": 640,
    },
    "retrained_v2": {
        "weights": str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
        "imgsz": 640,
    },
    "selcom_1280": {
        "weights": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"),
        "imgsz": 1280,
    },
    "selcom_640": {
        "weights": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"),
        "imgsz": 640,
    },
}

NEGATIVE_CATEGORIES = {"airplanes", "birds", "helicopters"}
ALL_CATEGORIES = ["airplanes", "birds", "drone", "helicopters"]
SEGMENT_SIZE = 3  # matches temporal 2-of-3 window


def score_dets_pipeline(dets, gt_boxes, is_negative):
    """Score detections using proper bipartite matching (IoP rule).
    Returns (tp, fp, fn) counts."""
    if is_negative:
        # All detections are FP, no GT
        return 0, len(dets), 0
    return score_detections(dets, gt_boxes, rule="iop", iop_thr=0.5)


def run_yolo(model, img, conf, imgsz, device):
    res = model.predict(img, conf=conf, verbose=False, imgsz=imgsz, device=device)
    boxes = res[0].boxes
    dets = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        c = float(boxes.conf[i])
        dets.append(((float(xyxy[0]), float(xyxy[1]),
                       float(xyxy[2]), float(xyxy[3])), c))
    return dets


def build_classifier_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}
    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6)})
    rgb_global = compute_global_features(rgb_gray, modality="rgb")
    ir_global = compute_global_features(ir_gray, modality="ir")
    feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
    feats.update({f"ir_{k}": v for k, v in ir_global.items()})
    for prefix, dets, gray, gw, gh in [
        ("rgb", rgb_dets, rgb_gray, rgb_w, rgb_h),
        ("ir", ir_dets, ir_gray, ir_w, ir_h),
    ]:
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    return np.array([[feats.get(c, 0) for c in feat_cols]], dtype=np.float32)


def trust_decision_to_dets(label, rgb_dets, ir_dets):
    if label == 0: return []
    elif label == 1: return rgb_dets
    elif label == 2: return ir_dets
    else: return rgb_dets + ir_dets


def segments_prf(per_frame_gt_present, per_frame_fired, seg_size):
    """Compute segment-level P/R/F1. Each segment: positive if any GT, detected if any fired."""
    n = len(per_frame_gt_present)
    tp = fp = fn = tn = 0
    for i in range(0, n, seg_size):
        chunk_gt = per_frame_gt_present[i:i+seg_size]
        chunk_det = per_frame_fired[i:i+seg_size]
        has_gt = any(chunk_gt)
        has_det = any(chunk_det)
        if has_gt and has_det: tp += 1
        elif not has_gt and has_det: fp += 1
        elif has_gt and not has_det: fn += 1
        else: tn += 1
    n_seg = tp + fp + fn + tn
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2*p*r / max(p+r, 1e-9)
    return {"segments": n_seg, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def eval_pipeline(rgb_yolo, ir_yolo, classifier, feat_cols, verifier,
                  rgb_name, rgb_imgsz, ds_path, category,
                  rgb_conf, ir_conf, patch_thr, device):
    is_negative = category in NEGATIVE_CATEGORIES
    img_dir = ds_path / "images" / "test"
    lbl_dir = ds_path / "labels" / "test"
    if not img_dir.exists():
        return {}
    ds = ImageDataset(img_dir, lbl_dir)
    images = ds.list_images()
    if not images:
        return {}
    total = len(images)

    # Per-stage detection-level accumulators (proper bipartite matching)
    stages = ["rgb_yolo", "ir_yolo", "after_classifier"]
    det_acc = {s: {"TP": 0, "FP": 0, "FN": 0} for s in stages}
    trust_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # Per-frame logs for segment-based temporal metrics
    pf_gt_present = []      # bool: does this frame have GT drone?
    pf_temporal_fired = []  # bool: was alert active after temporal?
    pf_final_fired = []     # bool: was alert active after patch veto?

    alert_events = 0
    vetoed_alerts = 0
    prev_alert_temporal = False
    prev_alert_final = False

    temporal = PerModalityTemporalState(
        stride=1, warning_window=3, warning_require=2,
        alert_window=3, alert_require=2,
        warning_cooldown_frames=0, alert_cooldown_frames=0,
    )

    t0 = time.time()
    for idx, img_path in enumerate(images):
        frame_data = ds.load_frame(img_path)
        if frame_data is None:
            continue
        img = frame_data["img"]
        w, h = frame_data["w"], frame_data["h"]
        stem = frame_data["stem"]

        if is_negative:
            gt = []
        else:
            lbl_path = lbl_dir / f"{stem}.txt"
            gt = read_yolo_labels(lbl_path, w, h, drone_classes={0})

        has_gt = len(gt) > 0
        pf_gt_present.append(has_gt)

        # Stage 1: RGB YOLO
        rgb_dets = run_yolo(rgb_yolo, img, rgb_conf, rgb_imgsz, device)
        rgb_xyxy = [d[0] for d in rgb_dets]
        tp, fp, fn = score_dets_pipeline(rgb_dets, gt, is_negative)
        det_acc["rgb_yolo"]["TP"] += tp; det_acc["rgb_yolo"]["FP"] += fp; det_acc["rgb_yolo"]["FN"] += fn

        # Stage 2: IR YOLO on grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ir_dets = run_yolo(ir_yolo, gray_3ch, ir_conf, 640, device)
        ir_xyxy = [d[0] for d in ir_dets]
        tp, fp, fn = score_dets_pipeline(ir_dets, gt, is_negative)
        det_acc["ir_yolo"]["TP"] += tp; det_acc["ir_yolo"]["FP"] += fp; det_acc["ir_yolo"]["FN"] += fn

        # Stage 3: Classifier
        x = build_classifier_features(rgb_dets, ir_dets, gray, gray, feat_cols)
        label = int(classifier.predict(x)[0])
        trust_counts[label] += 1
        trusted_dets = trust_decision_to_dets(label, rgb_dets, ir_dets)
        trusted_xyxy = [d[0] for d in trusted_dets]
        tp, fp, fn = score_dets_pipeline(trusted_dets, gt, is_negative)
        det_acc["after_classifier"]["TP"] += tp; det_acc["after_classifier"]["FP"] += fp; det_acc["after_classifier"]["FN"] += fn

        # Stage 4: Temporal 2-of-3
        temporal_input = [[d[0][0], d[0][1], d[0][2], d[0][3], d[1]]
                          for d in trusted_dets]
        warn, alert = temporal.update(temporal_input, w, h)
        pf_temporal_fired.append(bool(alert))

        # Stage 5: Patch verifier (only on new alert transition)
        final_alert = alert
        vetoed = False
        if alert and not prev_alert_temporal:
            if trusted_xyxy:
                probs = verifier.predict_boxes(img, trusted_xyxy)
                if any(p >= patch_thr for p in probs):
                    final_alert = False
                    vetoed = True
                    vetoed_alerts += 1
            if final_alert:
                alert_events += 1

        # Sustain final_alert state (if not vetoed, stays active while temporal says so)
        if not alert:
            final_alert = False
        pf_final_fired.append(bool(final_alert))

        prev_alert_temporal = bool(alert)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"      {idx+1:>5d}/{total}  {(idx+1)/elapsed:.1f} fps")

    elapsed = time.time() - t0

    # Compute per-frame P/R for detection stages
    result = {
        "rgb_model": rgb_name, "dataset": ds_path.name, "category": category,
        "is_negative": is_negative, "total_frames": total,
        "alert_events": alert_events, "vetoed_alerts": vetoed_alerts,
        "trust_counts": trust_counts, "elapsed_s": round(elapsed, 1),
    }

    for stage in stages:
        da = det_acc[stage]
        tp, fp, fn = da["TP"], da["FP"], da["FN"]
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p+r, 1e-9)
        result[stage] = {**da, "precision": round(p, 4), "recall": round(r, 4),
                         "f1": round(f1, 4)}

    # Segment-based metrics for temporal stages
    result["seg_temporal"] = segments_prf(pf_gt_present, pf_temporal_fired, SEGMENT_SIZE)
    result["seg_final"] = segments_prf(pf_gt_present, pf_final_fired, SEGMENT_SIZE)

    return result


def print_summary(all_rows, rgb_models):
    pos = [r for r in all_rows if not r["is_negative"]]
    neg = [r for r in all_rows if r["is_negative"]]
    models = list(rgb_models.keys())

    def agg_frame(rows, stage):
        tp = sum(r[stage]["TP"] for r in rows)
        fp = sum(r[stage]["FP"] for r in rows)
        fn = sum(r[stage]["FN"] for r in rows)
        p = tp / max(tp+fp, 1); r_ = tp / max(tp+fn, 1)
        f1 = 2*p*r_ / max(p+r_, 1e-9)
        return tp, fp, fn, p, r_, f1

    def agg_seg(rows, stage):
        tp = sum(r[stage]["TP"] for r in rows)
        fp = sum(r[stage]["FP"] for r in rows)
        fn = sum(r[stage]["FN"] for r in rows)
        ns = sum(r[stage]["segments"] for r in rows)
        p = tp / max(tp+fp, 1); r_ = tp / max(tp+fn, 1)
        f1 = 2*p*r_ / max(p+r_, 1e-9)
        return ns, tp, fp, fn, p, r_, f1

    # ── TIER 1 ──
    print(f"\n{'='*90}")
    print(f"  TIER 1: PER-FRAME Detection Metrics (P / R / F1)")
    print(f"{'='*90}")

    if pos:
        print(f"\n  DRONE (positive, {sum(r['total_frames'] for r in pos if r['rgb_model']==models[0])} frames):")
        hdr = f"  {'Model':<18s} {'Stage':<20s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>7s} {'R':>7s} {'F1':>7s}"
        print(hdr)
        print(f"  {'-'*len(hdr)}")
        for mn in models:
            mr = [r for r in pos if r["rgb_model"] == mn]
            if not mr: continue
            for stage, label in [("rgb_yolo","RGB YOLO"), ("ir_yolo","IR YOLO (gray)"),
                                 ("after_classifier","Classifier")]:
                tp,fp,fn,p,r_,f1 = agg_frame(mr, stage)
                print(f"  {mn:<18s} {label:<20s} {tp:>6d} {fp:>6d} {fn:>6d} {p:>7.3f} {r_:>7.3f} {f1:>7.3f}")
            print()

    if neg:
        print(f"\n  CONFUSERS (negative, {sum(r['total_frames'] for r in neg if r['rgb_model']==models[0])} frames):")
        hdr = f"  {'Model':<18s} {'Stage':<20s} {'FP':>6s} {'Total':>7s} {'FPR':>7s}"
        print(hdr)
        print(f"  {'-'*len(hdr)}")
        for mn in models:
            mr = [r for r in neg if r["rgb_model"] == mn]
            if not mr: continue
            tot = sum(r["total_frames"] for r in mr)
            for stage, label in [("rgb_yolo","RGB YOLO"), ("ir_yolo","IR YOLO (gray)"),
                                 ("after_classifier","Classifier")]:
                fp = sum(r[stage]["FP"] for r in mr)
                print(f"  {mn:<18s} {label:<20s} {fp:>6d} {tot:>7d} {fp/tot:>7.3f}")
            print()

    # ── TIER 2 ──
    print(f"\n{'='*90}")
    print(f"  TIER 2: SEGMENT-BASED Alert Metrics ({SEGMENT_SIZE}-frame segments, P / R / F1)")
    print(f"{'='*90}")

    if pos:
        print(f"\n  DRONE (positive):")
        hdr = f"  {'Model':<18s} {'Stage':<20s} {'Segs':>6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>7s} {'R':>7s} {'F1':>7s}"
        print(hdr)
        print(f"  {'-'*len(hdr)}")
        for mn in models:
            mr = [r for r in pos if r["rgb_model"] == mn]
            if not mr: continue
            for stage, label in [("seg_temporal","Temporal 2/3"), ("seg_final","+ Patch veto")]:
                ns,tp,fp,fn,p,r_,f1 = agg_seg(mr, stage)
                print(f"  {mn:<18s} {label:<20s} {ns:>6d} {tp:>6d} {fp:>6d} {fn:>6d} {p:>7.3f} {r_:>7.3f} {f1:>7.3f}")
            al = sum(r["alert_events"] for r in mr)
            ve = sum(r["vetoed_alerts"] for r in mr)
            print(f"  {mn:<18s} {'Alert events':<20s} passed={al}, vetoed={ve}, total={al+ve}")
            print()

    if neg:
        print(f"\n  CONFUSERS (negative):")
        hdr = f"  {'Model':<18s} {'Stage':<20s} {'Segs':>6s} {'FP':>6s} {'Total':>7s} {'FPR':>7s}"
        print(hdr)
        print(f"  {'-'*len(hdr)}")
        for mn in models:
            mr = [r for r in neg if r["rgb_model"] == mn]
            if not mr: continue
            for stage, label in [("seg_temporal","Temporal 2/3"), ("seg_final","+ Patch veto")]:
                ns,tp,fp,fn,p,r_,f1 = agg_seg(mr, stage)
                tot_seg = ns
                print(f"  {mn:<18s} {label:<20s} {ns:>6d} {fp:>6d} {tot_seg:>7d} {fp/max(tot_seg,1):>7.3f}")
            al = sum(r["alert_events"] for r in mr)
            ve = sum(r["vetoed_alerts"] for r in mr)
            print(f"  {mn:<18s} {'Alert events':<20s} passed={al}, vetoed={ve}, total={al+ve}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--rgb-models", nargs="*", default=None)
    ap.add_argument("--classifier", type=str, default=None,
                    help="Classifier name under models/routers/<name>/model.joblib "
                         "(e.g. control_v3more_40feat), OR an explicit relative path to a "
                         ".joblib file (e.g. classifier/runs/reliability/fusion/fusion_no_fn_model_v1.1.joblib). "
                         "Defaults to scene_aware_v3more_32feat.")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="Optional suffix for output dir: results/pipeline_video_tests_<tag>/. "
                         "Use to keep separate runs from clobbering each other's cached JSONs.")
    args = ap.parse_args()

    # Resolve classifier override
    global CLASSIFIER_PATH
    if args.classifier:
        # Two accepted forms:
        #   (a) bare name -> models/routers/<name>/model.joblib
        #   (b) relative path ending in .joblib -> use as-is
        if args.classifier.endswith(".joblib"):
            CLASSIFIER_PATH = REPO / args.classifier
        else:
            CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / args.classifier / "model.joblib"

    categories = args.categories or ALL_CATEGORIES
    rgb_models = RGB_MODELS
    if args.rgb_models:
        rgb_models = {k: v for k, v in RGB_MODELS.items() if k in args.rgb_models}

    datasets = []
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists(): continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if ds_dir.is_dir() and (ds_dir / "images" / "test").exists():
                datasets.append((cat, ds_dir))

    print(f"Full Pipeline Evaluation")
    print(f"  Stages: RGB YOLO -> IR YOLO(gray) -> Classifier -> Temporal(2/3) -> Patch veto")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  RGB models: {list(rgb_models.keys())}")
    print(f"  Segment:    {SEGMENT_SIZE} frames (matches temporal window)")
    print(f"  RGB conf:   {args.rgb_conf}  IR conf: {args.ir_conf}  Patch thr: {args.patch_thr}")

    out_dir_name = "pipeline_video_tests"
    if args.out_tag:
        out_dir_name = f"pipeline_video_tests_{args.out_tag}"
    out_dir = EVAL_DIR / "results" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Classifier: {CLASSIFIER_PATH.parent.name}")
    print(f"  Output dir: {out_dir}")

    from ultralytics import YOLO
    print(f"\n  Loading IR model...")
    ir_yolo = YOLO(str(IR_WEIGHTS))
    print(f"  Loading classifier...")
    clf_loaded = joblib.load(CLASSIFIER_PATH)
    if isinstance(clf_loaded, dict) and "model" in clf_loaded:
        classifier = clf_loaded["model"]
        feat_cols = clf_loaded["features"]
    else:
        # Raw model (e.g. fusion_no_fn_v1.1). Look for sibling <stem>_metrics.json
        # with a "features" list — the post-hoc extracted metadata.
        classifier = clf_loaded
        stem = CLASSIFIER_PATH.stem  # e.g. "fusion_no_fn_model_v1.1"
        # Try the bare "<tag>_metrics.json" form first, then "<stem>_metrics.json".
        candidates = [
            CLASSIFIER_PATH.parent / "fusion_no_fn_v1.1_metrics.json",
            CLASSIFIER_PATH.parent / f"{stem}_metrics.json",
        ]
        meta_path = next((p for p in candidates if p.exists()), None)
        if meta_path is None:
            raise RuntimeError(
                f"Raw classifier at {CLASSIFIER_PATH} has no sibling features metadata; "
                f"expected one of: {[str(p) for p in candidates]}"
            )
        with open(meta_path) as f:
            feat_cols = json.load(f)["features"]
    print(f"    Features: {len(feat_cols)}, classes: {list(classifier.classes_)}")
    print(f"  Loading patch verifier...")
    verifier = PatchVerifier(str(PATCH_RGB), device="cuda")

    loaded_rgb = {}
    for name, info in rgb_models.items():
        print(f"  Loading RGB: {name}...")
        loaded_rgb[name] = YOLO(info["weights"])

    all_rows = []
    for cat, ds_dir in datasets:
        n_imgs = len(list((ds_dir / "images" / "test").glob("*.jpg")))
        is_neg = cat in NEGATIVE_CATEGORIES
        print(f"\n{'='*70}")
        print(f"  {cat}/{ds_dir.name}  ({n_imgs} frames, {'NEG' if is_neg else 'POS'})")
        print(f"{'='*70}")

        for rgb_name, rgb_info in rgb_models.items():
            # Skip if already computed with current format (has seg_final)
            vid_out = out_dir / cat / ds_dir.name
            cached_json = vid_out / f"{rgb_name}.json"
            if cached_json.exists():
                try:
                    cached = json.loads(cached_json.read_text())
                    if "seg_final" in cached:
                        print(f"\n    RGB: {rgb_name} -- cached, skipping")
                        all_rows.append(cached)
                        continue
                except Exception:
                    pass

            print(f"\n    RGB: {rgb_name} (imgsz={rgb_info['imgsz']})")
            result = eval_pipeline(
                loaded_rgb[rgb_name], ir_yolo, classifier, feat_cols,
                verifier, rgb_name, rgb_info["imgsz"],
                ds_dir, cat, args.rgb_conf, args.ir_conf, args.patch_thr, args.device)
            if not result: continue

            # Inline per-video
            total_trig = result['alert_events'] + result['vetoed_alerts']
            if is_neg:
                raw = result["rgb_yolo"]["FP"]
                clf = result["after_classifier"]["FP"]
                seg_t = result["seg_temporal"]["FP"]
                seg_f = result["seg_final"]["FP"]
                print(f"    FP: raw={raw} -> clf={clf} -> seg_temp={seg_t} -> seg_final={seg_f}  "
                      f"triggers={total_trig} (passed={result['alert_events']}, vetoed={result['vetoed_alerts']})")
            else:
                sf = result["seg_final"]
                print(f"    Seg P/R/F1: {sf['precision']:.3f}/{sf['recall']:.3f}/{sf['f1']:.3f}  "
                      f"triggers={total_trig} (passed={result['alert_events']}, vetoed={result['vetoed_alerts']})")

            vid_out.mkdir(parents=True, exist_ok=True)
            with open(cached_json, "w") as f:
                json.dump(result, f, indent=2)
            all_rows.append(result)

    # ── Print summary tables ──
    print_summary(all_rows, rgb_models)

    # ── Save CSV ──
    csv_path = out_dir / "pipeline_comparison.csv"
    fields = [
        "dataset", "category", "rgb_model", "total_frames",
        # Tier 1: per-frame
        "rgb_tp", "rgb_fp", "rgb_fn", "rgb_p", "rgb_r", "rgb_f1",
        "ir_tp", "ir_fp", "ir_fn", "ir_p", "ir_r", "ir_f1",
        "clf_tp", "clf_fp", "clf_fn", "clf_p", "clf_r", "clf_f1",
        # Tier 2: segment-based
        "seg_temp_segs", "seg_temp_tp", "seg_temp_fp", "seg_temp_fn",
        "seg_temp_p", "seg_temp_r", "seg_temp_f1",
        "seg_final_segs", "seg_final_tp", "seg_final_fp", "seg_final_fn",
        "seg_final_p", "seg_final_r", "seg_final_f1",
        # Alert events
        "alert_events", "vetoed_alerts",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            row = {
                "dataset": r["dataset"], "category": r["category"],
                "rgb_model": r["rgb_model"], "total_frames": r["total_frames"],
            }
            for prefix, stage in [("rgb","rgb_yolo"),("ir","ir_yolo"),("clf","after_classifier")]:
                s = r[stage]
                row.update({f"{prefix}_tp": s["TP"], f"{prefix}_fp": s["FP"],
                           f"{prefix}_fn": s["FN"], f"{prefix}_p": s["precision"],
                           f"{prefix}_r": s["recall"], f"{prefix}_f1": s["f1"]})
            for prefix, stage in [("seg_temp","seg_temporal"),("seg_final","seg_final")]:
                s = r[stage]
                row.update({f"{prefix}_segs": s["segments"], f"{prefix}_tp": s["TP"],
                           f"{prefix}_fp": s["FP"], f"{prefix}_fn": s["FN"],
                           f"{prefix}_p": s["precision"], f"{prefix}_r": s["recall"],
                           f"{prefix}_f1": s["f1"]})
            row["alert_events"] = r["alert_events"]
            row["vetoed_alerts"] = r["vetoed_alerts"]
            w.writerow(row)
    print(f"\n  Saved: {csv_path}")

    json_path = out_dir / "pipeline_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"\n[pipeline eval] Done.")


if __name__ == "__main__":
    main()
