"""diagnose_rgbtest_imgsz_640_vs_1280.py — does eval imgsz explain the mlp_v5
recall regression on rgb_dataset_test?

The shipped mlp_v5 RGB filter was distilled with Svanstrom/Selcom drone crops
extracted at imgsz=1280; the production rgb_dataset_test cache (and the thesis
per-size table) is built at imgsz=640. The per-detection diagnosis showed the
falsely-vetoed real drones form a fully separable APPEARANCE sub-population
(LDA=1.0, conf delta=0). Hypothesis: at imgsz=640 a small drone's P3/P5
features land outside the 1280-trained drone manifold, so the filter vetoes it.

This runs the SAME small sample of rgb_dataset_test images through ft4 at
imgsz=640 and imgsz=1280, extracts the exact 517-D mlp_v5 features, scores the
filter, and compares detector recall + filter-kept recall + filter veto rate,
overall and per GT-size bucket (GT sqrt-area in original-image px).

  py eval/diagnose_rgbtest_imgsz_640_vs_1280.py --n 500

GPU (ultralytics predict). Read-only on data; writes a JSON + prints a table.
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import (DetectInputHook, _extract_detection_features,
                                 _iou)  # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier  # noqa: E402

FT4 = REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
MLP_V5 = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
IMG_DIR = Path("G:/drone/dataset/dataset/images/test")
LBL_DIR = Path("G:/drone/dataset/dataset/labels/test")
CONF = 0.25          # detector conf floor (matches production cache)
THR = 0.25           # mlp_v5 P(drone) accept threshold (RGB shipped)
DRONE_CLASS = 0
SIZE_EDGES = [(0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, 1e9, ">=64px")]


def size_bucket(box):
    s = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
    for lo, hi, name in SIZE_EDGES:
        if lo <= s < hi:
            return name
    return ">=64px"


def load_gt(stem, iw, ih):
    p = LBL_DIR / (stem + ".txt")
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            t = line.split()
            if len(t) >= 5 and int(t[0]) == DRONE_CLASS:
                xc, yc, bw, bh = map(float, t[1:5])
                out.append(((xc - bw / 2) * iw, (yc - bh / 2) * ih,
                            (xc + bw / 2) * iw, (yc + bh / 2) * ih))
    return out


def run_imgsz(model, hook, mlp, images, imgsz):
    # per-bucket counts: matched(=detector recall numerator), kept(after filter)
    buckets = {b[2]: {"n_gt": 0, "matched": 0, "kept": 0} for b in SIZE_EDGES}
    tot = {"n_gt": 0, "matched": 0, "kept": 0}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        gt = load_gt(img_path.stem, iw, ih)
        if not gt:
            continue
        hook.clear()
        res = model.predict(img, imgsz=imgsz, conf=CONF, verbose=False, device="cuda")
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                xyxy = tuple(boxes.xyxy[i].cpu().numpy())
                dets.append((xyxy, float(boxes.conf[i])))
        # score filter on each det
        probs = []
        if dets:
            feats = np.array([_extract_detection_features(hook, b, (ih, iw), c)
                              for b, c in dets], dtype=np.float32)
            probs = mlp.predict_drone_probs(feats)
        # match each GT to best det (recall view)
        for g in gt:
            bk = size_bucket(g)
            buckets[bk]["n_gt"] += 1; tot["n_gt"] += 1
            best_i, best_iou = -1, 0.0
            for i, (b, _c) in enumerate(dets):
                v = _iou(b, g)
                if v >= 0.5 and v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0:
                buckets[bk]["matched"] += 1; tot["matched"] += 1
                if probs[best_i] >= THR:
                    buckets[bk]["kept"] += 1; tot["kept"] += 1
    return buckets, tot


def rec(d):
    return round(d["matched"] / max(d["n_gt"], 1), 4)


def filt_rec(d):
    return round(d["kept"] / max(d["n_gt"], 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    all_imgs = sorted(p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    # only images that actually have a drone GT (so the sample is informative)
    with_gt = [p for p in all_imgs if (LBL_DIR / (p.stem + ".txt")).exists()
               and any(int(l.split()[0]) == DRONE_CLASS
                       for l in (LBL_DIR / (p.stem + ".txt")).read_text().splitlines()
                       if l.split())]
    random.shuffle(with_gt)
    sample = with_gt[:args.n]
    print(f"rgb_dataset_test: {len(all_imgs)} imgs, {len(with_gt)} with drone GT; "
          f"sampling {len(sample)}")

    model = YOLO(str(FT4))
    hook = DetectInputHook(); h = hook.register(model)
    mlp = MLPv4Verifier(MLP_V5, device="cuda")

    out = {}
    for imgsz in (640, 1280):
        b, t = run_imgsz(model, hook, mlp, sample, imgsz)
        out[imgsz] = {"buckets": b, "total": t}
        print(f"\n=== imgsz={imgsz} ===  (n_gt={t['n_gt']})")
        print(f"  detector recall      {rec(t):.4f}   ({t['matched']}/{t['n_gt']})")
        print(f"  filter-kept recall   {filt_rec(t):.4f}   ({t['kept']}/{t['n_gt']})")
        loss = (t["matched"] - t["kept"]) / max(t["matched"], 1)
        print(f"  filter veto of TPs   {loss:.4f}   ({t['matched']-t['kept']}/{t['matched']})")
        print(f"  {'bucket':<10} {'n_gt':>6} {'det_R':>8} {'filt_R':>8} {'veto%':>7}")
        for k in (e[2] for e in SIZE_EDGES):
            d = b[k]
            vl = (d["matched"] - d["kept"]) / max(d["matched"], 1)
            print(f"  {k:<10} {d['n_gt']:>6} {rec(d):>8.4f} {filt_rec(d):>8.4f} {vl:>7.2%}")
    h.remove()

    outp = REPO / "docs/analysis/2026-06-17_rgbtest_imgsz_640_vs_1280.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
