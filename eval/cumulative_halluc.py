"""Cumulative halluc-rate chart: RGB-alone → +classifier → +patch verifier.

Two modes:
  --mode confuser   : RGB confuser-zoo dataset (no GT). IR fed grayscale-replicate.
                      Halluc = alert fires on a frame with no drone GT.
  --mode svanstrom  : paired Svanstrom (real IR + GT). Halluc still counts alerts
                      on non-DRONE-category frames; on DRONE frames computes
                      recall vs GT (IoP @ 0.5, per project convention).

Stages reported (cumulative):
  S1  RGB YOLO alone fires (any det with conf >= rgb_conf).
  S2  Trust-classifier-gated: alert iff classifier_label != 0 (not reject_both).
  S3  + patch verifier on alert gate: revoke alert if any det in the trusted
      modality has patch_prob >= patch_thr (matches production `alert_gate_only`).

Temporal stage NOT included — needs ordered video; left as a separate study.

Output dir: eval/results/_cumulative_halluc/<mode>_<tag>/
  - per_frame.csv : every frame's stage decisions + class category
  - summary.json  : halluc / firing rates per stage, broken down by category
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "classifier"))

from datasets import ImageDataset, PairedDataset, detect_category
from metrics import iou_iop
from run_manifest import write_manifest

DEFAULT_RGB = REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"
DEFAULT_IR = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
DEFAULT_CLF = REPO / "classifier" / "runs" / "reliability" / "fusion" / "fusion_no_fn_model_v1.1.joblib"
DEFAULT_PATCH_RGB = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
DEFAULT_PATCH_IR = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"

CONFUSER_ROOT = Path("G:/drone/rgb_confusers_merged")
SVANSTROM_ROOT = Path("G:/drone/svanstrom_paired")

CATEGORIES = ["DRONE", "BIRD", "AIRPLANE", "HELICOPTER", "OTHER"]
# Stage 2 trust labels: 0=reject_both, 1=trust_rgb, 2=trust_ir, 3=trust_both


def _frame_decisions(rgb_dets, ir_dets, rgb_probs, ir_probs, clf_label, patch_thr):
    """Compute stage firings for one frame.

    Returns dict with bool s1, s2, s3 and the trusted-modality vetoed flag.
    """
    s1 = len(rgb_dets) > 0
    if clf_label == 1:
        trusted = "rgb"; trusted_n = len(rgb_dets); trusted_probs = rgb_probs
    elif clf_label == 2:
        trusted = "ir"; trusted_n = len(ir_dets); trusted_probs = ir_probs
    elif clf_label == 3:
        trusted = "both"
        trusted_n = len(rgb_dets) + len(ir_dets)
        trusted_probs = list(rgb_probs) + list(ir_probs)
    else:
        trusted = "none"; trusted_n = 0; trusted_probs = []
    s2 = (clf_label != 0) and (trusted_n > 0)
    if not s2:
        s3 = False
        vetoed = False
    else:
        vetoed = any(p >= patch_thr for p in trusted_probs)
        s3 = not vetoed
    return {
        "s1": s1, "s2": s2, "s3": s3,
        "trusted": trusted, "vetoed": vetoed,
        "clf_label": clf_label,
        "trusted_n": trusted_n,
    }


def _yolo_dets(model, img, conf, imgsz):
    r = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)
    b = r[0].boxes
    return [
        ((float(b.xyxy[i][0]), float(b.xyxy[i][1]),
          float(b.xyxy[i][2]), float(b.xyxy[i][3])), float(b.conf[i]))
        for i in range(len(b))
    ]


def _classify_confuser_source(stem: str) -> str:
    """Map confuser-zoo filename to a category bucket."""
    if stem.startswith("airplane_") or "_AIRPLANE_" in stem: return "AIRPLANE"
    if stem.startswith("helicopter_") or "_HELICOPTER_" in stem: return "HELICOPTER"
    if stem.startswith("bird_") or stem.startswith("raihanrsd_") or "_BIRD_" in stem: return "BIRD"
    return "OTHER"


def run_confuser(args, model_rgb, model_ir, clf, patch_rgb, patch_ir, clf_feats, build_features):
    img_dir = CONFUSER_ROOT / "images" / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.stride > 1:
        imgs = imgs[::args.stride]
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[confuser] {len(imgs)} images @ imgsz={args.imgsz}")

    per_frame = []
    by_cat = defaultdict(lambda: {"n": 0, "s1": 0, "s2": 0, "s3": 0})
    t0 = time.time()
    for idx, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None: continue
        cat = _classify_confuser_source(p.stem)

        rgb_dets = _yolo_dets(model_rgb, img, args.rgb_conf, args.imgsz)
        gray_rep = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        ir_dets = _yolo_dets(model_ir, gray_rep, args.ir_conf, args.imgsz)

        rgb_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(gray_rep, cv2.COLOR_BGR2GRAY)
        feats = build_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
        x = np.array([[feats.get(c, 0) for c in clf_feats]], dtype=np.float32)
        clf_label = int(clf.predict(x)[0])

        rgb_probs = patch_rgb.predict_boxes(img, [d[0] for d in rgb_dets]).tolist() if rgb_dets else []
        ir_probs = patch_ir.predict_boxes(gray_rep, [d[0] for d in ir_dets]).tolist() if ir_dets else []

        d = _frame_decisions(rgb_dets, ir_dets, rgb_probs, ir_probs, clf_label, args.patch_thr)
        per_frame.append({
            "frame": p.stem, "category": cat,
            "n_rgb": len(rgb_dets), "n_ir": len(ir_dets),
            "clf_label": clf_label, "trusted": d["trusted"],
            "s1": int(d["s1"]), "s2": int(d["s2"]), "s3": int(d["s3"]),
            "patch_vetoed": int(d["vetoed"]),
            "max_rgb_patch": round(max(rgb_probs) if rgb_probs else 0.0, 4),
            "max_ir_patch": round(max(ir_probs) if ir_probs else 0.0, 4),
        })
        s = by_cat[cat]
        s["n"] += 1
        s["s1"] += int(d["s1"]); s["s2"] += int(d["s2"]); s["s3"] += int(d["s3"])
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(imgs)}  {(idx + 1) / (time.time() - t0):.1f} fps")
    return per_frame, by_cat


def run_svanstrom(args, model_rgb, model_ir, clf, patch_rgb, patch_ir, clf_feats, build_features):
    rgb_dir = SVANSTROM_ROOT / "RGB" / "images"
    ir_dir = SVANSTROM_ROOT / "IR" / "images"
    rgb_lbl = SVANSTROM_ROOT / "RGB" / "labels"
    rgb_imgs = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.stride > 1:
        rgb_imgs = rgb_imgs[::args.stride]
    if args.limit:
        rgb_imgs = rgb_imgs[: args.limit]
    print(f"[svanstrom] {len(rgb_imgs)} paired frames @ imgsz={args.imgsz} (IoP scoring)")

    from datasets import read_yolo_labels
    per_frame = []
    by_cat = defaultdict(lambda: {"n": 0, "s1": 0, "s2": 0, "s3": 0,
                                  "tp_s1": 0, "tp_s2": 0, "tp_s3": 0,
                                  "fp_s1": 0, "fp_s2": 0, "fp_s3": 0,
                                  "fn_s1": 0, "fn_s2": 0, "fn_s3": 0})
    t0 = time.time()
    for idx, rgb_p in enumerate(rgb_imgs):
        stem = rgb_p.stem
        cat = detect_category(stem)
        # Resolve IR partner — Svanstrom convention: RGB stems end in _visible, IR in _infrared
        ir_stem = stem.replace("_visible", "_infrared")
        ir_candidates = list(ir_dir.glob(f"{ir_stem}.*"))
        if not ir_candidates: continue
        ir_p = ir_candidates[0]
        rgb_img = cv2.imread(str(rgb_p))
        ir_img = cv2.imread(str(ir_p))
        if rgb_img is None or ir_img is None: continue
        rh, rw = rgb_img.shape[:2]
        gt_rgb = read_yolo_labels(rgb_lbl / f"{stem}.txt", rw, rh)

        rgb_dets = _yolo_dets(model_rgb, rgb_img, args.rgb_conf, args.imgsz)
        ir_dets = _yolo_dets(model_ir, ir_img, args.ir_conf, args.imgsz)

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        feats = build_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
        x = np.array([[feats.get(c, 0) for c in clf_feats]], dtype=np.float32)
        clf_label = int(clf.predict(x)[0])

        rgb_probs = patch_rgb.predict_boxes(rgb_img, [d[0] for d in rgb_dets]).tolist() if rgb_dets else []
        ir_probs = patch_ir.predict_boxes(ir_img, [d[0] for d in ir_dets]).tolist() if ir_dets else []

        d = _frame_decisions(rgb_dets, ir_dets, rgb_probs, ir_probs, clf_label, args.patch_thr)
        # Stage-resolved per-frame "what would the system have detected"
        # Use RGB-side scoring with IoP @ 0.5 since user mandate is IoP for Svanstrom RGB.
        def _score(stage_fires, kept_dets):
            if cat != "DRONE":
                if stage_fires and kept_dets:
                    return 0, len(kept_dets), 0  # all alerts on non-DRONE are FP
                return 0, 0, 0
            if not stage_fires:
                return 0, 0, len(gt_rgb)
            used = set(); tp = fp = 0
            for db, _c in kept_dets:
                best_ip, bi = 0.0, -1
                for gi, g in enumerate(gt_rgb):
                    _iu, ip = iou_iop(db, g)
                    if ip > best_ip: best_ip, bi = ip, gi
                if best_ip >= 0.5 and bi not in used:
                    tp += 1; used.add(bi)
                else:
                    fp += 1
            fn = len(gt_rgb) - len(used)
            return tp, fp, fn
        kept_s1 = rgb_dets
        kept_s2 = rgb_dets if d["clf_label"] in (1, 3) else (ir_dets if d["clf_label"] == 2 else [])
        kept_s3 = kept_s2 if not d["vetoed"] else []
        tp1, fp1, fn1 = _score(d["s1"], kept_s1)
        tp2, fp2, fn2 = _score(d["s2"], kept_s2 if cat == "DRONE" else kept_s1)
        tp3, fp3, fn3 = _score(d["s3"], kept_s3 if cat == "DRONE" else kept_s1)

        per_frame.append({
            "frame": stem, "category": cat,
            "n_rgb": len(rgb_dets), "n_ir": len(ir_dets),
            "clf_label": clf_label, "trusted": d["trusted"],
            "s1": int(d["s1"]), "s2": int(d["s2"]), "s3": int(d["s3"]),
            "tp_s1": tp1, "fp_s1": fp1, "fn_s1": fn1,
            "tp_s3": tp3, "fp_s3": fp3, "fn_s3": fn3,
            "patch_vetoed": int(d["vetoed"]),
        })
        s = by_cat[cat]
        s["n"] += 1
        for k, val in [("s1", d["s1"]), ("s2", d["s2"]), ("s3", d["s3"])]:
            s[k] += int(val)
        s["tp_s1"] += tp1; s["fp_s1"] += fp1; s["fn_s1"] += fn1
        s["tp_s2"] += tp2; s["fp_s2"] += fp2; s["fn_s2"] += fn2
        s["tp_s3"] += tp3; s["fp_s3"] += fp3; s["fn_s3"] += fn3
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(rgb_imgs)}  {(idx + 1) / (time.time() - t0):.1f} fps")
    return per_frame, by_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["confuser", "svanstrom"])
    ap.add_argument("--rgb-weights", default=str(DEFAULT_RGB))
    ap.add_argument("--ir-weights", default=str(DEFAULT_IR))
    ap.add_argument("--classifier-path", default=str(DEFAULT_CLF))
    ap.add_argument("--patch-rgb", default=str(DEFAULT_PATCH_RGB))
    ap.add_argument("--patch-ir", default=str(DEFAULT_PATCH_IR))
    ap.add_argument("--rgb-conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="test", help="confuser mode only")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    tag = args.tag or Path(args.classifier_path).stem
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO / "eval" / "results" / "_cumulative_halluc" / f"{args.mode}_{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cumulative] out: {out_dir}")

    write_manifest(
        out_dir=out_dir, args=vars(args),
        weights_paths={
            "rgb": args.rgb_weights, "ir": args.ir_weights,
            "classifier": args.classifier_path,
            "patch_rgb": args.patch_rgb, "patch_ir": args.patch_ir,
        },
        cache_paths=[],
        extra={"mode": args.mode, "split": args.split},
    )

    from ultralytics import YOLO
    import joblib
    from patch_verifier import PatchVerifier
    sys.path.insert(0, str(REPO / "ir_gui"))
    from eval_pipeline import build_features  # type: ignore

    print("[cumulative] loading models...")
    model_rgb = YOLO(args.rgb_weights)
    model_ir = YOLO(args.ir_weights)
    bundle = joblib.load(args.classifier_path)
    clf = bundle["model"]
    clf_feats = bundle["features"]
    patch_rgb = PatchVerifier(args.patch_rgb)
    patch_ir = PatchVerifier(args.patch_ir)

    if args.mode == "confuser":
        per_frame, by_cat = run_confuser(args, model_rgb, model_ir, clf, patch_rgb, patch_ir, clf_feats, build_features)
    else:
        per_frame, by_cat = run_svanstrom(args, model_rgb, model_ir, clf, patch_rgb, patch_ir, clf_feats, build_features)

    # Write per-frame CSV
    csv_path = out_dir / "per_frame.csv"
    if per_frame:
        cols = list(per_frame[0].keys())
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for r in per_frame:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")

    # Summary
    summary = {"mode": args.mode, "n_frames": len(per_frame), "by_category": {}}
    overall = {"n": 0, "s1": 0, "s2": 0, "s3": 0}
    for cat, s in by_cat.items():
        n = max(s["n"], 1)
        cat_row = {
            "n": s["n"],
            "s1_fire_rate": round(s["s1"] / n, 4),
            "s2_fire_rate": round(s["s2"] / n, 4),
            "s3_fire_rate": round(s["s3"] / n, 4),
        }
        if args.mode == "svanstrom":
            for tag in ("s1", "s2", "s3"):
                tp = s[f"tp_{tag}"]; fp = s[f"fp_{tag}"]; fn = s[f"fn_{tag}"]
                p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
                f1 = 2 * p_ * r_ / max(p_ + r_, 1e-9)
                cat_row[f"{tag}_TP"] = tp; cat_row[f"{tag}_FP"] = fp; cat_row[f"{tag}_FN"] = fn
                cat_row[f"{tag}_P"] = round(p_, 4); cat_row[f"{tag}_R"] = round(r_, 4); cat_row[f"{tag}_F1"] = round(f1, 4)
        summary["by_category"][cat] = cat_row
        overall["n"] += s["n"]; overall["s1"] += s["s1"]
        overall["s2"] += s["s2"]; overall["s3"] += s["s3"]
    n = max(overall["n"], 1)
    summary["overall"] = {
        "n": overall["n"],
        "s1_fire_rate": round(overall["s1"] / n, 4),
        "s2_fire_rate": round(overall["s2"] / n, 4),
        "s3_fire_rate": round(overall["s3"] / n, 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[cumulative] Done. Out: {out_dir}\n")
    print(f"{'Category':12s} {'N':>5s} {'S1_fire':>8s} {'S2_fire':>8s} {'S3_fire':>8s}")
    print("-" * 50)
    for cat in CATEGORIES:
        if cat not in summary["by_category"]: continue
        r = summary["by_category"][cat]
        print(f"{cat:12s} {r['n']:>5d} {r['s1_fire_rate']:>8.3f} {r['s2_fire_rate']:>8.3f} {r['s3_fire_rate']:>8.3f}")
    o = summary["overall"]
    print("-" * 50)
    print(f"{'OVERALL':12s} {o['n']:>5d} {o['s1_fire_rate']:>8.3f} {o['s2_fire_rate']:>8.3f} {o['s3_fire_rate']:>8.3f}")


if __name__ == "__main__":
    main()
