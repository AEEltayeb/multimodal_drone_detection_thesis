"""
eval_thesis_ablation.py — Focused thesis ablation for 3 dataset groups.

Stages per frame (on RGB-input datasets):
  1. RGB YOLO alone
  2. IR YOLO (grayscale) alone
  3. Classifier (sa32) trust fusion  →  R(clf) >= max(R(rgb), R(ir))
  4. + Filter (RGB patch verifier on IR-gray dets)

For IR-only dataset (thermal):
  1. IR YOLO native (IoU scoring)
  2. + IR filter

Datasets:
  A. Video tests  (drone + confuser clips, IoP)
  B. IR_dset_final test  (thermal, IoU)
  C. RGB dataset test  (G:/drone/dataset, IoP)

Models:
  - baseline (Yolo26n_trained, imgsz=640)
  - selcom_960 (Yolo26n_selcom_mixed_ft2_1280, imgsz=960)
  - ir_v3b grayscale (on RGB datasets, IoP, RGB filter)
  - ir_v3b native (on IR dataset, IoU, IR filter)

Usage:
    python eval/eval_thesis_ablation.py
    python eval/eval_thesis_ablation.py --max-per-dataset 1500
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import score_detections, compute_prf, iou_iop
from datasets import ImageDataset, read_yolo_labels


def frame_level_score(dets, gt, rule="iop", thr=0.5):
    """Frame-level binary scoring: did ANY detection overlap ANY GT?
    Returns (tp, fp, fn) where each is 0 or 1 per frame.
    - tp=1 if gt present AND at least one det overlaps a gt
    - fp=1 if no gt AND at least one det exists
    - fn=1 if gt present AND no det overlaps any gt
    """
    has_gt = len(gt) > 0
    has_det = len(dets) > 0
    if not has_gt and not has_det:
        return 0, 0, 0  # TN
    if not has_gt and has_det:
        return 0, 1, 0  # FP frame
    if has_gt and not has_det:
        return 0, 0, 1  # FN frame
    # Both present — check if any det overlaps any gt
    for d_box, _ in dets:
        for g in gt:
            iu, ip = iou_iop(d_box, g)
            s = iu if rule == "iou" else ip
            if s >= thr:
                return 1, 0, 0  # TP frame
    return 0, 0, 1  # dets exist but none overlap GT → FN
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

# ── Paths ────────────────────────────────────────────────────────
RGB_BASELINE = str(REPO / "models/rgb/Yolo26n_trained/weights/best.pt")
RGB_SELCOM   = str(REPO / "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt")
IR_V3B       = str(REPO / "models/ir/IR_final_cleaned/weights/best.pt")
CLF_PATH     = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
RGB_FILTER   = str(REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt")
IR_FILTER    = str(REPO / "models/patches/confuser_filter4_ir_v2_backup.pt")

VIDEO_ROOT   = REPO / "datasets/drone detection video tests/rgb"
IR_TEST_IMG  = Path("G:/drone/IR_dset_final/test/images")
IR_TEST_LBL  = Path("G:/drone/IR_dset_final/test/labels")
# Try common YOLO dataset structures for RGB
_RGB_CANDIDATES = [
    (Path("G:/drone/dataset/dataset/images/test"), Path("G:/drone/dataset/dataset/labels/test")),
    (Path("G:/drone/dataset/images/test"), Path("G:/drone/dataset/labels/test")),
    (Path("G:/drone/dataset/test/images"), Path("G:/drone/dataset/test/labels")),
]
RGB_TEST_IMG, RGB_TEST_LBL = next(
    ((i, l) for i, l in _RGB_CANDIDATES if i.exists()),
    _RGB_CANDIDATES[0]  # fallback
)

SELCOM_VAL_IMG = Path("G:/drone/_finetune_selcom_mixed_ft2/images/val")
SELCOM_VAL_LBL = Path("G:/drone/_finetune_selcom_mixed_ft2/labels/val")


# ── Helpers ──────────────────────────────────────────────────────
def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def infer(model, img, imgsz, conf, device, grayscale=False):
    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img, conf=conf, verbose=False, imgsz=imgsz, device=device)
    boxes = res[0].boxes
    dets = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        c = float(boxes.conf[i])
        dets.append(((float(xyxy[0]), float(xyxy[1]),
                       float(xyxy[2]), float(xyxy[3])), c))
    return dets


def filter_dets(verifier, img, dets, thr=0.70):
    if not dets:
        return []
    xyxy_list = [d[0] for d in dets]
    probs = verifier.predict_boxes(img, xyxy_list)
    return [d for d, p in zip(dets, probs) if p < thr]


def stride_for_max(total, max_n):
    return max(1, math.ceil(total / max_n))


def build_classifier_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    """Build the 32-feature vector for the sa32 classifier."""
    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}

    # Confidence features
    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6)})

    # Global scene features — modality parameter is critical for correct fills
    rgb_global = compute_global_features(rgb_gray, modality="rgb")
    ir_global = compute_global_features(ir_gray, modality="ir")
    feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
    feats.update({f"ir_{k}": v for k, v in ir_global.items()})

    # Target features (best-confidence box)
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
    """Classifier trust label -> which detections to keep.
    0=trust_neither, 1=trust_rgb, 2=trust_ir, 3=trust_both (union)."""
    if label == 0: return []
    elif label == 1: return list(rgb_dets)
    elif label == 2: return list(ir_dets)
    else: return list(rgb_dets) + list(ir_dets)  # union → R >= max(R_rgb, R_ir)


def collect_video_images():
    """Discover video test images grouped by category."""
    cat_data = defaultdict(list)  # cat -> [(img_path, lbl_dir)]
    for cat_dir in sorted(VIDEO_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        for vid_dir in sorted(cat_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            img_d = vid_dir / "images/test"
            lbl_d = vid_dir / "labels/test"
            if not img_d.exists():
                continue
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for p in sorted(img_d.iterdir()):
                if p.suffix.lower() in exts:
                    cat_data[cat].append((p, lbl_d))
    return cat_data


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-dataset", type=int, default=2000)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--device", default="0")
    ap.add_argument("--sections", nargs="*", default=["A", "B", "C", "D"],
                     help="Which sections to run: A=video, B=IR, C=RGB, D=SelCom (default: all)")
    args = ap.parse_args()
    args.sections = [s.upper() for s in args.sections]
    MAX = args.max_per_dataset

    print("Loading models...")
    m_base = load_yolo(RGB_BASELINE)
    m_selcom = load_yolo(RGB_SELCOM)
    m_ir = load_yolo(IR_V3B)

    from patch_verifier import PatchVerifier
    pv_rgb = PatchVerifier(RGB_FILTER)
    pv_ir = PatchVerifier(IR_FILTER)
    print("  Filters loaded")

    clf_data = joblib.load(CLF_PATH)
    classifier = clf_data["model"]
    feat_cols = clf_data["features"]
    print(f"  Classifier: {len(feat_cols)} features, classes={list(classifier.classes_)}")

    out_dir = EVAL_DIR / "results/thesis_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ─── Helper: eval one dataset group with full pipeline ───────
    def eval_rgb_dataset_group(group_name, pairs, is_negative, max_n):
        """Run baseline + selcom_960 + IR gray + classifier + filter on RGB images.

        pairs: list of (img_path, lbl_dir)
        """
        s = stride_for_max(len(pairs), max_n)
        pairs = pairs[::s]
        n = len(pairs)
        neg_tag = "NEG" if is_negative else "POS"
        print(f"\n  [{group_name}] {n} frames (stride={s}, {neg_tag})")

        rgb_models = [
            ("baseline",   m_base,   640),
            ("selcom_960", m_selcom, 960),
        ]

        for rgb_name, rgb_model, rgb_imgsz in rgb_models:
            # Accumulators per stage
            acc = {stage: {"tp": 0, "fp": 0, "fn": 0}
                   for stage in ["rgb", "ir_gray", "classifier", "clf+filter"]}
            fp_frames = {stage: 0 for stage in acc}

            t0 = time.time()
            for idx, (img_path, lbl_dir) in enumerate(pairs):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gt = [] if is_negative else read_yolo_labels(
                    lbl_dir / f"{img_path.stem}.txt", w, h)

                # Stage 1: RGB YOLO — frame-level scoring
                rgb_dets = infer(rgb_model, img, rgb_imgsz, args.conf, args.device)
                tp, fp, fn = frame_level_score(rgb_dets, gt, rule="iop")
                acc["rgb"]["tp"] += tp; acc["rgb"]["fp"] += fp; acc["rgb"]["fn"] += fn
                if (rgb_dets and is_negative): fp_frames["rgb"] += 1

                # Stage 2: IR YOLO (grayscale) — frame-level, IoP scoring
                ir_dets = infer(m_ir, img, 640, args.ir_conf, args.device, grayscale=True)
                tp, fp, fn = frame_level_score(ir_dets, gt, rule="iop")
                acc["ir_gray"]["tp"] += tp; acc["ir_gray"]["fp"] += fp; acc["ir_gray"]["fn"] += fn
                if (ir_dets and is_negative): fp_frames["ir_gray"] += 1

                # Stage 3: Classifier (trust fusion) — frame-level scoring
                # Any detection from a trusted modality covering the drone = TP
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                x = build_classifier_features(rgb_dets, ir_dets, gray, gray, feat_cols)
                label = int(classifier.predict(x)[0])
                clf_dets = trust_decision_to_dets(label, rgb_dets, ir_dets)
                tp, fp, fn = frame_level_score(clf_dets, gt, rule="iop")
                acc["classifier"]["tp"] += tp; acc["classifier"]["fp"] += fp; acc["classifier"]["fn"] += fn
                if (clf_dets and is_negative): fp_frames["classifier"] += 1

                # Stage 4: Classifier + RGB filter — frame-level scoring
                clf_filtered = filter_dets(pv_rgb, img, clf_dets, args.patch_thr)
                tp, fp, fn = frame_level_score(clf_filtered, gt, rule="iop")
                acc["clf+filter"]["tp"] += tp; acc["clf+filter"]["fp"] += fp; acc["clf+filter"]["fn"] += fn
                if (clf_filtered and is_negative): fp_frames["clf+filter"] += 1

                if (idx + 1) % 300 == 0:
                    elapsed = time.time() - t0
                    print(f"      {idx+1:>5d}/{n}  {(idx+1)/elapsed:.1f} fps")

            elapsed = time.time() - t0

            # Build rows
            for stage in ["rgb", "ir_gray", "classifier", "clf+filter"]:
                m = compute_prf(acc[stage]["tp"], acc[stage]["fp"], acc[stage]["fn"])
                row = {
                    "dataset": group_name, "model": rgb_name, "stage": stage,
                    "n": n, "rule": "iop",
                    "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                    "P": m["precision"], "R": m["recall"], "F1": m["f1"],
                }
                if is_negative:
                    row["FP%"] = round(fp_frames[stage] / max(n, 1), 4)
                rows.append(row)

            # Quick inline summary
            r_rgb = compute_prf(acc["rgb"]["tp"], acc["rgb"]["fp"], acc["rgb"]["fn"])
            r_ir  = compute_prf(acc["ir_gray"]["tp"], acc["ir_gray"]["fp"], acc["ir_gray"]["fn"])
            r_clf = compute_prf(acc["classifier"]["tp"], acc["classifier"]["fp"], acc["classifier"]["fn"])
            r_flt = compute_prf(acc["clf+filter"]["tp"], acc["clf+filter"]["fp"], acc["clf+filter"]["fn"])

            if is_negative:
                print(f"    {rgb_name:15s}  FP%: rgb={fp_frames['rgb']/max(n,1):.3f}  "
                      f"ir={fp_frames['ir_gray']/max(n,1):.3f}  "
                      f"clf={fp_frames['classifier']/max(n,1):.3f}  "
                      f"clf+flt={fp_frames['clf+filter']/max(n,1):.3f}")
            else:
                # Verify classifier recall >= max(rgb, ir)
                clf_ok = "✓" if r_clf["recall"] >= max(r_rgb["recall"], r_ir["recall"]) - 0.001 else "✗"
                print(f"    {rgb_name:15s}  R: rgb={r_rgb['recall']:.3f}  ir={r_ir['recall']:.3f}  "
                      f"clf={r_clf['recall']:.3f} {clf_ok}  clf+flt={r_flt['recall']:.3f}  "
                      f"| P: clf={r_clf['precision']:.3f}  F1: clf={r_clf['f1']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION A: Video Tests
    # ═══════════════════════════════════════════════════════════════
    if "A" in args.sections:
        print("\n" + "="*70)
        print("  SECTION A: Video Tests")
        print("="*70)

        cat_data = collect_video_images()
        for cat in ["drone", "airplanes", "birds", "helicopters"]:
            if cat not in cat_data:
                continue
            is_neg = cat in {"airplanes", "birds", "helicopters"}
            eval_rgb_dataset_group(f"vid_{cat}", cat_data[cat], is_neg, MAX)

    # ═══════════════════════════════════════════════════════════════
    # SECTION B: IR_dset_final test (thermal, IR-only, IoU, IR filter)
    # ═══════════════════════════════════════════════════════════════
    if "B" in args.sections:
        print("\n" + "="*70)
        print("  SECTION B: IR_dset_final test (thermal IR)")
        print("="*70)

        if IR_TEST_IMG.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_imgs = sorted(p for p in IR_TEST_IMG.iterdir() if p.suffix.lower() in exts)
            s = stride_for_max(len(all_imgs), MAX)
            imgs = all_imgs[::s]
            n = len(imgs)
            print(f"  {n} frames (stride={s} from {len(all_imgs)})")

            tp_raw = fp_raw = fn_raw = 0
            tp_flt = fp_flt = fn_flt = 0
            t0 = time.time()

            for idx, img_path in enumerate(imgs):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gt = read_yolo_labels(IR_TEST_LBL / f"{img_path.stem}.txt", w, h)

                # IR native — IoU scoring
                dets = infer(m_ir, img, 640, args.ir_conf, args.device, grayscale=False)
                t, f_, m_ = score_detections(dets, gt, rule="iou")
                tp_raw += t; fp_raw += f_; fn_raw += m_

                # + IR filter
                dets_f = filter_dets(pv_ir, img, dets, args.patch_thr)
                t2, f2, m2 = score_detections(dets_f, gt, rule="iou")
                tp_flt += t2; fp_flt += f2; fn_flt += m2

                if (idx + 1) % 300 == 0:
                    elapsed = time.time() - t0
                    print(f"      {idx+1:>5d}/{n}  {(idx+1)/elapsed:.1f} fps")

            raw_m = compute_prf(tp_raw, fp_raw, fn_raw)
            flt_m = compute_prf(tp_flt, fp_flt, fn_flt)

            rows.append({"dataset": "ir_dset_test", "model": "ir_v3b", "stage": "raw",
                          "n": n, "rule": "iou", **raw_m,
                          "P": raw_m["precision"], "R": raw_m["recall"], "F1": raw_m["f1"]})
            rows.append({"dataset": "ir_dset_test", "model": "ir_v3b", "stage": "ir_filter",
                          "n": n, "rule": "iou", **flt_m,
                          "P": flt_m["precision"], "R": flt_m["recall"], "F1": flt_m["f1"]})

            print(f"    ir_v3b raw:    P={raw_m['precision']:.3f} R={raw_m['recall']:.3f} F1={raw_m['f1']:.3f}")
            print(f"    ir_v3b +filt:  P={flt_m['precision']:.3f} R={flt_m['recall']:.3f} F1={flt_m['f1']:.3f}")
        else:
            print(f"  SKIP: {IR_TEST_IMG} not found")

    # ═══════════════════════════════════════════════════════════════
    # SECTION C: RGB dataset test (G:/drone/dataset)
    # ═══════════════════════════════════════════════════════════════
    if "C" in args.sections:
        print("\n" + "="*70)
        print("  SECTION C: RGB dataset test")
        print("="*70)

        if RGB_TEST_IMG.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_imgs = sorted(p for p in RGB_TEST_IMG.iterdir() if p.suffix.lower() in exts)
            pairs = [(p, RGB_TEST_LBL) for p in all_imgs]
            eval_rgb_dataset_group("rgb_dataset_test", pairs, False, MAX)
        else:
            print(f"  SKIP: {RGB_TEST_IMG} not found")

    # ═══════════════════════════════════════════════════════════════
    # SECTION D: SelCom Footage val split
    # ═══════════════════════════════════════════════════════════════
    if "D" in args.sections:
        print("\n" + "="*70)
        print("  SECTION D: SelCom Footage val split")
        print("="*70)

        if SELCOM_VAL_IMG.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_imgs = sorted(p for p in SELCOM_VAL_IMG.iterdir() if p.suffix.lower() in exts)
            pairs = [(p, SELCOM_VAL_LBL) for p in all_imgs]
            eval_rgb_dataset_group("selcom_val_split", pairs, False, MAX)
        else:
            print(f"  SKIP: {SELCOM_VAL_IMG} not found")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    csv_path = out_dir / "thesis_ablation.csv"
    if rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved: {csv_path}")

    json_path = out_dir / "thesis_ablation.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Print summary table ──────────────────────────────────────
    print("\n" + "="*80)
    print("  THESIS ABLATION SUMMARY")
    print("="*80)
    print(f"  {'Dataset':<22s} {'Model':<15s} {'Stage':<13s} {'Rule':<5s} "
          f"{'N':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'FP%':>6s}")
    print(f"  {'-'*100}")
    for r in rows:
        fp_pct = f"{r['FP%']:.3f}" if 'FP%' in r else ""
        print(f"  {r['dataset']:<22s} {r['model']:<15s} {r['stage']:<13s} "
              f"{r['rule']:<5s} {r['n']:>5d} {r['P']:>6.3f} {r['R']:>6.3f} "
              f"{r['F1']:>6.3f} {fp_pct:>6s}")

    # ── Recall verification ──────────────────────────────────────
    print(f"\n  RECALL VERIFICATION (classifier R >= max(R_rgb, R_ir)):")
    pos_rows = [r for r in rows if "FP%" not in r and r["stage"] in ("rgb", "ir_gray", "classifier")]
    datasets_seen = set()
    for r in pos_rows:
        key = (r["dataset"], r["model"])
        if key in datasets_seen:
            continue
        datasets_seen.add(key)
        rgb_r = next((x["R"] for x in rows if x["dataset"]==r["dataset"]
                       and x["model"]==r["model"] and x["stage"]=="rgb"), 0)
        ir_r  = next((x["R"] for x in rows if x["dataset"]==r["dataset"]
                       and x["model"]==r["model"] and x["stage"]=="ir_gray"), 0)
        clf_r = next((x["R"] for x in rows if x["dataset"]==r["dataset"]
                       and x["model"]==r["model"] and x["stage"]=="classifier"), 0)
        ok = "✓" if clf_r >= max(rgb_r, ir_r) - 0.001 else "✗ VIOLATION"
        print(f"    {r['dataset']:<22s} {r['model']:<15s}  "
              f"R_rgb={rgb_r:.3f}  R_ir={ir_r:.3f}  R_clf={clf_r:.3f}  {ok}")

    print(f"\n[eval_thesis_ablation] Done.")


if __name__ == "__main__":
    main()
