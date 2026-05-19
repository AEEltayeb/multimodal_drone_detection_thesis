"""
03_per_model_failures.py — Per-frame TP/FN/FP log with size + clutter features.

For each (model, dataset) pair at imgsz=1280, dumps one CSV with rows per
GT box (status=TP|FN, conf if matched, drone size in px, local clutter) and
rows per unmatched prediction (status=FP, conf).

The result is what the document's "drone-size × success" plots and
"per-model fault profile" sections cite.

Conf threshold = 0.25 (production). IoP@0.5.

GPU only. Run AFTER 02_imgsz_sweep finishes.
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RGB model"))
from finetune_selcom import load_gt, iop  # noqa: E402

OUT_DIR = ROOT / "analytics" / "spec_analysis" / "results"
OUT_CSV = OUT_DIR / "failures.csv"

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
IOP_THRESH = 0.5
CONF_PROD  = 0.25

MODELS = {
    "old_baseline":   ROOT / "RGB model" / "Yolo26n_trained"               / "weights" / "best_pre_finetune.pt",
    "ft2_1280":       ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
    "hardneg_v3more": ROOT / "RGB model" / "Yolo26n_hardneg_v3_more"       / "weights" / "best.pt",
    "retrained_v2":   ROOT / "RGB model" / "Yolo26n_retrained_v2"          / "weights" / "best.pt",
}

DATASETS = {
    "selcom_val":  dict(images=Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val"),
                         labels=Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
                         stride=1, max_n=None),
    "dataset_rgb": dict(images=Path(r"G:/drone/dataset/dataset/images/test"),
                         labels=Path(r"G:/drone/dataset/dataset/labels/test"),
                         stride=5, max_n=None),
    "antiuav":     dict(images=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
                         labels=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
                         stride=200, max_n=500),
}

PAIRS = [
    ("old_baseline",   "selcom_val"),
    ("old_baseline",   "dataset_rgb"),
    ("old_baseline",   "antiuav"),
    ("ft2_1280",       "selcom_val"),
    ("ft2_1280",       "dataset_rgb"),
    ("ft2_1280",       "antiuav"),
    ("hardneg_v3more", "selcom_val"),
    ("hardneg_v3more", "antiuav"),
    ("retrained_v2",   "selcom_val"),
]
IMGSZ = 1280   # all pairs run at 1280 (the cross-cutting decision)


def source_bucket(stem: str) -> str:
    if stem.startswith("anti_uav_"):          return "anti_uav"
    if stem.startswith("anti-muav-roboflow"): return "anti-muav-roboflow"
    if stem.startswith("wosdetc_"):           return "wosdetc"
    if stem.startswith("AirBird"):            return "AirBird"
    if stem.startswith("FBD-SV"):             return "FBD-SV"
    if stem.startswith("selcom"):             return "selcom"
    return stem.split("_", 1)[0] if "_" in stem else "other"


def clutter_at_box(img_bgr, x1, y1, x2, y2, expand=2.0) -> float:
    """Laplacian variance on the box neighbourhood (clamped to frame)."""
    H, W = img_bgr.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * expand, (y2 - y1) * expand
    nx1, ny1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    nx2, ny2 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
    if nx2 <= nx1 or ny2 <= ny1: return 0.0
    crop = cv2.cvtColor(img_bgr[ny1:ny2, nx1:nx2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def run_pair(model, model_tag, ds_tag, ds_cfg):
    imgs = sorted(p for p in ds_cfg["images"].iterdir() if p.suffix.lower() in IMG_EXTS)
    if ds_cfg.get("stride", 1) > 1: imgs = imgs[::ds_cfg["stride"]]
    if ds_cfg.get("max_n"): imgs = imgs[:ds_cfg["max_n"]]

    rows = []
    t0 = time.perf_counter()
    for i, ip in enumerate(imgs):
        frame = cv2.imread(str(ip))
        if frame is None: continue
        H, W = frame.shape[:2]
        src = source_bucket(ip.stem)

        r = model.predict(frame, conf=CONF_PROD, iou=0.30,
                          imgsz=IMGSZ, verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds.append((x1, y1, x2, y2, float(r.boxes.conf[j])))

        gt = load_gt(ds_cfg["labels"] / (ip.stem + ".txt"))   # normalized
        matched_gt, matched_pred = set(), set()

        # Match each pred to best GT by IoP
        for pi, (px1, py1, px2, py2, pc) in enumerate(preds):
            best, bi = 0.0, -1
            for gi, gb in enumerate(gt):
                if gi in matched_gt: continue
                # gt is in normalized [0..1]; convert pred to normalized
                pn = (px1/W, py1/H, px2/W, py2/H)
                s = iop(pn, gb)
                if s > best: best, bi = s, gi
            if best >= IOP_THRESH:
                matched_gt.add(bi); matched_pred.add(pi)
                gx1n, gy1n, gx2n, gy2n = gt[bi]
                bw_px = (gx2n - gx1n) * W
                bh_px = (gy2n - gy1n) * H
                sqrt_px = float(np.sqrt(max(bw_px * bh_px, 1e-6)))
                rows.append(dict(
                    model=model_tag, dataset=ds_tag, source=src, frame=ip.name,
                    status="TP",
                    gt_cx=round(0.5 * (gx1n + gx2n), 4),
                    gt_cy=round(0.5 * (gy1n + gy2n), 4),
                    gt_w_px=round(bw_px, 1), gt_h_px=round(bh_px, 1),
                    sqrt_area_px=round(sqrt_px, 2),
                    clutter=round(clutter_at_box(frame, gx1n*W, gy1n*H, gx2n*W, gy2n*H), 1),
                    conf=round(pc, 4), iop=round(best, 3),
                ))

        # FN: unmatched GTs
        for gi, gb in enumerate(gt):
            if gi in matched_gt: continue
            gx1n, gy1n, gx2n, gy2n = gb
            bw_px = (gx2n - gx1n) * W
            bh_px = (gy2n - gy1n) * H
            sqrt_px = float(np.sqrt(max(bw_px * bh_px, 1e-6)))
            rows.append(dict(
                model=model_tag, dataset=ds_tag, source=src, frame=ip.name,
                status="FN",
                gt_cx=round(0.5 * (gx1n + gx2n), 4),
                gt_cy=round(0.5 * (gy1n + gy2n), 4),
                gt_w_px=round(bw_px, 1), gt_h_px=round(bh_px, 1),
                sqrt_area_px=round(sqrt_px, 2),
                clutter=round(clutter_at_box(frame, gx1n*W, gy1n*H, gx2n*W, gy2n*H), 1),
                conf=0.0, iop=0.0,
            ))

        # FP: unmatched preds
        for pi, (px1, py1, px2, py2, pc) in enumerate(preds):
            if pi in matched_pred: continue
            bw_px = px2 - px1
            bh_px = py2 - py1
            sqrt_px = float(np.sqrt(max(bw_px * bh_px, 1e-6)))
            rows.append(dict(
                model=model_tag, dataset=ds_tag, source=src, frame=ip.name,
                status="FP",
                gt_cx=round((px1+px2) / (2*W), 4),
                gt_cy=round((py1+py2) / (2*H), 4),
                gt_w_px=round(bw_px, 1), gt_h_px=round(bh_px, 1),
                sqrt_area_px=round(sqrt_px, 2),
                clutter=round(clutter_at_box(frame, px1, py1, px2, py2), 1),
                conf=round(pc, 4), iop=0.0,
            ))

        if (i+1) % 200 == 0:
            print(f"    {i+1}/{len(imgs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_pairs = set()
    if args.resume and OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for row in csv.DictReader(f):
                done_pairs.add((row["model"], row["dataset"]))
        print(f"Resume: {len(done_pairs)} pairs already in CSV")

    from ultralytics import YOLO
    loaded = {}

    fieldnames = ["model", "dataset", "source", "frame", "status",
                  "gt_cx", "gt_cy", "gt_w_px", "gt_h_px",
                  "sqrt_area_px", "clutter", "conf", "iop"]
    is_new = not OUT_CSV.exists()
    f_out = OUT_CSV.open("a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if is_new: writer.writeheader(); f_out.flush()

    for i, (m, d) in enumerate(PAIRS, 1):
        if (m, d) in done_pairs:
            print(f"[{i}/{len(PAIRS)}] {m} x {d}  [skip]"); continue
        if not MODELS[m].exists():
            print(f"[{i}/{len(PAIRS)}] {m} weights missing"); continue
        if m not in loaded:
            print(f"  loading {m}")
            loaded[m] = YOLO(str(MODELS[m]))
        print(f"[{i}/{len(PAIRS)}] {m} x {d} @ imgsz={IMGSZ}", flush=True)
        try:
            rows = run_pair(loaded[m], m, d, DATASETS[d])
            writer.writerows(rows); f_out.flush()
            print(f"  wrote {len(rows)} rows")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    f_out.close()
    print(f"\nDONE. {OUT_CSV}")


if __name__ == "__main__":
    main()
