"""
eval/svan_resolution_sweep.py — one RGB model on Svanström RGB at a chosen imgsz (round-5 N5).

Purpose: fill the resolution 2x2 (baseline/retrained_v2 x 640/1280) under ONE harness, so the
thesis's resolution figure compares like with like. The old figure mixed models (baseline@1280 vs
a May-10 "rgb_only"@640 whose model identity the ledger never names); this script settles both
the missing cell and the attribution.

Protocol: Svanström paired RGB images, stride 7 (~4,102 frames, same density as the Tier-1
sample), detector conf 0.25, IoP@0.5 scoring (loose Svanström GT boxes), drones only.

  py -u eval/svan_resolution_sweep.py --model baseline --imgsz 640
  py -u eval/svan_resolution_sweep.py --model baseline --imgsz 1280
  py -u eval/svan_resolution_sweep.py --model retrained_v2 --imgsz 640
  py -u eval/svan_resolution_sweep.py --model retrained_v2 --imgsz 1280

Each run APPENDS its cell to eval/results/svan_resolution_sweep.json (key "<model>@<imgsz>").
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

import cv2
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
import sys
sys.path.insert(0, str(EVAL_DIR))
from metrics import score_detections, compute_prf  # noqa: E402

SVAN_IMG = Path("G:/drone/svanstrom_paired/RGB/images")
SVAN_LBL = Path("G:/drone/svanstrom_paired/RGB/labels")
OUT = REPO / "eval" / "results" / "svan_resolution_sweep.json"

MODELS = {
    "baseline":     REPO / "models/rgb/Yolo26n_trained/weights/best.pt",
    "retrained_v2": REPO / "models/rgb/Yolo26n_retrained_v2/weights/best.pt",
    "ft4":          REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt",
}


def gt_boxes(stem, w, h):
    p = SVAN_LBL / (stem + ".txt")
    out = []
    if p.exists():
        for ln in open(p, encoding="utf-8"):
            t = ln.split()
            if len(t) >= 5 and t[0] == "0":
                cx, cy, bw, bh = (float(x) for x in t[1:5])
                out.append((max((cx - bw / 2) * w, 0), max((cy - bh / 2) * h, 0),
                            min((cx + bw / 2) * w, w), min((cy + bh / 2) * h, h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="baseline | retrained_v2 | ft4 | path to .pt")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    wpath = Path(MODELS.get(args.model, args.model))
    assert wpath.exists(), f"weights not found: {wpath}"
    whash = hashlib.sha256(wpath.read_bytes()).hexdigest()[:12]
    model = YOLO(str(wpath))

    imgs = sorted(p for p in SVAN_IMG.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = imgs[::args.stride]
    print(f"{args.model}@{args.imgsz}  weights={wpath.name} sha256={whash}  "
          f"n={len(imgs)} (stride {args.stride})  conf={args.conf}  rule=IoP@0.5")

    tp = fp = fn = n_gt = 0
    t0 = time.time()
    for k, p in enumerate(imgs):
        if k and k % 500 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(imgs)}  {k/el:.1f} fps  ETA {(len(imgs)-k)/(k/el):.0f}s", flush=True)
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)
        dets = [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c))
                for b, c in zip(res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.conf.cpu().numpy())]
        gt = gt_boxes(p.stem.replace("_visible", ""), w, h) or gt_boxes(p.stem, w, h)
        t, f, n = score_detections(dets, gt, rule="iop")
        tp += t; fp += f; fn += n; n_gt += len(gt)

    prf = compute_prf(tp, fp, fn)
    cell = {**prf, "TP": tp, "FP": fp, "FN": fn, "n_gt": n_gt, "n_frames": len(imgs),
            "weights": str(wpath.relative_to(REPO)), "weights_sha256_12": whash,
            "imgsz": args.imgsz, "stride": args.stride, "conf": args.conf,
            "rule": "iop@0.5 drones-only", "date": time.strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allcells = json.load(open(OUT)) if OUT.exists() else {}
    allcells[f"{args.model}@{args.imgsz}"] = cell
    json.dump(allcells, open(OUT, "w"), indent=2)
    print(f"\n{args.model}@{args.imgsz}:  P {prf['precision']}  R {prf['recall']}  F1 {prf['f1']}  "
          f"(TP {tp} / FP {fp} / FN {fn}, n_gt {n_gt})\n-> {OUT}")


if __name__ == "__main__":
    main()
