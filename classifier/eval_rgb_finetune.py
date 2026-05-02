"""
eval_rgb_finetune.py — Side-by-side RGB-only evaluation:
old (best_pre_finetune.pt) vs new (Yolo26n_hardneg/weights/best.pt)
on a fixed set of test corpora.

Test corpora:
  1. Anti-UAV-RGBT test/RGB         stride-sample 1/10 → ~8.5k frames
                                    drones-only — regression hard floor
  2. Svanström paired RGB           full 28.7k frames
                                    drones + airplane/bird/heli confusers
  3. Airplane test split            100 frames, all confuser (any det = FP)
  4. New_Dataset test split         drone-class frames excluded → confuser only
  5. Helicopter test (carved)       200 frames, all confuser

Per detection scoring (IoU + IoP @ 0.5). Confuser-only test sets get
"any det = FP" treatment (no GT positives).

Resume checkpoints + progress every 200 frames so you can ctrl-C and
restart without losing work. Reuses no caches — each run does its own
YOLO inference for the model it's testing.

Usage:
    python classifier/eval_rgb_finetune.py
    python classifier/eval_rgb_finetune.py --models old new      # default
    python classifier/eval_rgb_finetune.py --models new          # only new
    python classifier/eval_rgb_finetune.py --datasets antiuav svanstrom
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent

# Named model registry. Add new candidates here; --models flag picks subsets.
MODELS = {
    "old":      REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt",
    "frozen":   REPO / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt",
    "freeze8":  REPO / "RGB model" / "Yolo26n_freeze8" / "weights" / "best.pt",
    "unfrozen": REPO / "RGB model" / "Yolo26n_unfrozen" / "weights" / "best.pt",
    "hardneg_v2": REPO / "RGB model" / "Yolo26n_hardneg_v2" / "weights" / "best.pt",
}
# Back-compat for callers/scripts that still pass --models old new
MODELS["new"] = MODELS["frozen"]

OUT_ROOT = SCRIPT_DIR / "runs" / "rgb_finetune_eval"

ANTIUAV_RGB_IMG = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_RGB_LBL = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels")

SVAN_RGB_IMG    = Path(r"G:/drone/svanstrom_paired/RGB/images")
SVAN_RGB_LBL    = Path(r"G:/drone/svanstrom_paired/RGB/labels")

AIRPLANE_TEST   = Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb/test/images")
NEW_DS_TEST_IMG = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb/test/images")
NEW_DS_TEST_LBL = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb/test/labels")
NEW_DS_DRONE_CLASS = 2

HELI_TEST       = Path(r"G:/drone/finetune_dataset/images/test/helicopter")

DRONE_DSET_TEST_IMG = Path(r"G:/drone/dataset/dataset/images/test")
DRONE_DSET_TEST_LBL = Path(r"G:/drone/dataset/dataset/labels/test")

RGB_CONF  = 0.25
IOU_MATCH = 0.5
IOP_MATCH = 0.5
CKPT_EVERY = 200

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")


# ─── helpers ──────────────────────────────────────────────────────

def svan_category(stem: str) -> str:
    for c in SVAN_CATS:
        if f"_{c}_" in stem:
            return c
    return "OTHER"


def read_yolo_labels(path: Path, w: int, h: int, drop_class: int | None = None):
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        try:
            cls = int(p[0])
        except ValueError:
            continue
        if drop_class is not None and cls == drop_class:
            return None  # signal "skip this frame"
        # only treat class 0 as drone-positive GT
        if cls != 0 and drop_class is None:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ))
    return boxes


def iou_iop(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    iu = inter / u if u > 0 else 0.0
    ip = inter / aa if aa > 0 else 0.0
    return iu, ip


def score_dets(dets, gts, rule="iou", thr=0.5):
    """Return TP, FP, FN. dets: list of ((x1,y1,x2,y2), conf). gts: list of boxes."""
    tp = fp = 0
    used = set()
    for (db, _c) in dets:
        best_i, best_s = -1, 0.0
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(db, g)
            s = iu if rule == "iou" else ip
            if s > best_s:
                best_s, best_i = s, gi
        if best_s >= thr and best_i not in used:
            tp += 1
            used.add(best_i)
        else:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn


# ─── frame iterators ──────────────────────────────────────────────

def stride(items, n):
    if n <= 0 or n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def iter_antiuav(sample=8537):
    imgs = sorted(p for p in ANTIUAV_RGB_IMG.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = stride(imgs, sample)
    for p in imgs:
        yield {"img": p, "lbl": ANTIUAV_RGB_LBL / (p.stem + ".txt"),
               "key": p.stem, "category": "DRONE", "kind": "drone"}


def iter_dataset_rgb():
    """Original RGB training-data test split — full 17,209 frames, drones,
    in-distribution regression check."""
    imgs = sorted(p for p in DRONE_DSET_TEST_IMG.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in imgs:
        yield {"img": p, "lbl": DRONE_DSET_TEST_LBL / (p.stem + ".txt"),
               "key": f"dsrgb__{p.stem}", "category": "DRONE", "kind": "drone"}


def iter_svanstrom():
    imgs = sorted(p for p in SVAN_RGB_IMG.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in imgs:
        cat = svan_category(p.stem)
        yield {"img": p, "lbl": SVAN_RGB_LBL / (p.stem + ".txt"),
               "key": p.stem, "category": cat,
               "kind": ("drone" if cat == "DRONE" else "confuser")}


def iter_dir_confuser(d: Path, prefix: str, drop_class: int | None = None,
                      lbl_dir: Path | None = None):
    imgs = sorted(p for p in d.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in imgs:
        # Only used to filter: if drop_class set, skip frames with that class
        skip = False
        if drop_class is not None and lbl_dir is not None:
            lbl = lbl_dir / (p.stem + ".txt")
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    parts = ln.strip().split()
                    if parts and parts[0].isdigit() and int(parts[0]) == drop_class:
                        skip = True
                        break
        if skip:
            continue
        yield {"img": p, "lbl": None, "key": f"{prefix}__{p.stem}",
               "category": prefix.upper(), "kind": "confuser"}


# ─── main per-dataset eval ────────────────────────────────────────

def run_eval(model_name: str, model_path: Path, dataset_name: str, frames):
    out_dir = OUT_ROOT / dataset_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "progress.jsonl"

    # Resume
    counters_iou = {"tp": 0, "fp": 0, "fn": 0}
    counters_iop = {"tp": 0, "fp": 0, "fn": 0}
    fp_by_cat_iou = Counter()
    fp_by_cat_iop = Counter()
    n_drone_frames = 0
    n_confuser_frames = 0
    confuser_det_frames = 0  # any-det frames among confuser-only (FP rate)
    done = set()
    if prog_path.exists():
        for ln in prog_path.read_text().splitlines():
            if not ln.strip(): continue
            r = json.loads(ln)
            done.add(r["key"])
            counters_iou["tp"] += r["iou"][0]; counters_iou["fp"] += r["iou"][1]; counters_iou["fn"] += r["iou"][2]
            counters_iop["tp"] += r["iop"][0]; counters_iop["fp"] += r["iop"][1]; counters_iop["fn"] += r["iop"][2]
            fp_by_cat_iou[r["cat"]] += r["iou"][1]
            fp_by_cat_iop[r["cat"]] += r["iop"][1]
            if r["kind"] == "drone": n_drone_frames += 1
            else:
                n_confuser_frames += 1
                if r["any_det"]: confuser_det_frames += 1
        print(f"  [resume] {len(done):,} frames already processed")

    print(f"\n[{model_name}] dataset={dataset_name}  loading {model_path.name} ...")
    model = YOLO(str(model_path))

    frames = list(frames)
    total = len(frames)
    print(f"[{model_name}] {dataset_name}: {total:,} frames")

    t0 = time.time()
    buf = []
    n_session = 0

    for idx, f in enumerate(frames):
        if f["key"] in done:
            continue
        img = cv2.imread(str(f["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT boxes (drone-class only)
        if f["kind"] == "drone":
            gts = read_yolo_labels(f["lbl"], w, h) if f["lbl"] else []
        else:
            gts = []

        # YOLO
        res = model.predict(img, conf=RGB_CONF, iou=0.45, imgsz=640,
                            verbose=False, device=0, max_det=300)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(confs)):
                dets.append(((float(xyxy[i, 0]), float(xyxy[i, 1]),
                              float(xyxy[i, 2]), float(xyxy[i, 3])),
                             float(confs[i])))

        any_det = bool(dets)

        # Score
        tp_u, fp_u, fn_u = score_dets(dets, gts, rule="iou", thr=IOU_MATCH)
        tp_p, fp_p, fn_p = score_dets(dets, gts, rule="iop", thr=IOP_MATCH)

        counters_iou["tp"] += tp_u; counters_iou["fp"] += fp_u; counters_iou["fn"] += fn_u
        counters_iop["tp"] += tp_p; counters_iop["fp"] += fp_p; counters_iop["fn"] += fn_p
        fp_by_cat_iou[f["category"]] += fp_u
        fp_by_cat_iop[f["category"]] += fp_p
        if f["kind"] == "drone":
            n_drone_frames += 1
        else:
            n_confuser_frames += 1
            if any_det:
                confuser_det_frames += 1

        buf.append(json.dumps({
            "key": f["key"], "cat": f["category"], "kind": f["kind"],
            "iou": [tp_u, fp_u, fn_u], "iop": [tp_p, fp_p, fn_p],
            "any_det": any_det,
        }))
        n_session += 1

        if n_session % CKPT_EVERY == 0:
            with prog_path.open("a") as fh:
                fh.write("\n".join(buf) + "\n")
            buf.clear()
            elapsed = time.time() - t0
            fps = n_session / elapsed
            remaining = (total - len(done) - n_session) / max(fps, 1e-6)
            print(f"  [{model_name}/{dataset_name}] "
                  f"{len(done) + n_session:>6,}/{total:,}  "
                  f"{fps:.1f} fps  ETA {remaining/60:.1f} min")

    if buf:
        with prog_path.open("a") as fh:
            fh.write("\n".join(buf) + "\n")

    # Summary
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "n_frames": n_drone_frames + n_confuser_frames,
        "n_drone_frames": n_drone_frames,
        "n_confuser_frames": n_confuser_frames,
        "confuser_any_det_rate": (confuser_det_frames / n_confuser_frames
                                  if n_confuser_frames > 0 else None),
        "iou": _metrics_block(counters_iou),
        "iop": _metrics_block(counters_iop),
        "fp_by_category_iou": dict(fp_by_cat_iou),
        "fp_by_category_iop": dict(fp_by_cat_iop),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _metrics_block(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ─── orchestration ────────────────────────────────────────────────

DATASETS = {
    "antiuav":      lambda: list(iter_antiuav()),
    "dataset_rgb":  lambda: list(iter_dataset_rgb()),
    "svanstrom":    lambda: list(iter_svanstrom()),
    "airplane":     lambda: list(iter_dir_confuser(AIRPLANE_TEST, "airplane")),
    "new_dataset":  lambda: list(iter_dir_confuser(NEW_DS_TEST_IMG, "new_dataset",
                                                    drop_class=NEW_DS_DRONE_CLASS,
                                                    lbl_dir=NEW_DS_TEST_LBL)),
    "helicopter":   lambda: list(iter_dir_confuser(HELI_TEST, "helicopter")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["old", "frozen", "freeze8"],
                    choices=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--add-model", action="append", default=[],
                    help="add custom model: NAME=PATH (repeatable)")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    weights = dict(MODELS)
    for spec in args.add_model:
        if "=" not in spec:
            print(f"[fatal] --add-model expects NAME=PATH, got {spec}")
            sys.exit(1)
        name, path = spec.split("=", 1)
        weights[name] = Path(path)
        if name not in args.models:
            args.models.append(name)

    for k in args.models:
        if k not in weights:
            print(f"[fatal] unknown model {k}; available: {sorted(weights)}")
            sys.exit(1)
        if not weights[k].exists():
            print(f"[fatal] weights missing for {k}: {weights[k]}")
            sys.exit(1)

    all_results = {}
    for ds_name in args.datasets:
        all_results[ds_name] = {}
        frames = DATASETS[ds_name]()
        for m in args.models:
            print("\n" + "=" * 72)
            print(f"  {m.upper()} model on {ds_name.upper()}  ({len(frames):,} frames)")
            print("=" * 72)
            all_results[ds_name][m] = run_eval(m, weights[m], ds_name, frames)

    # Final side-by-side
    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    for ds_name, by_model in all_results.items():
        print(f"\n[{ds_name}]")
        for rule in ("iou", "iop"):
            print(f"  {rule.upper()} match:")
            print(f"    {'model':<10s} {'TP':>8s} {'FP':>8s} {'FN':>8s} "
                  f"{'P':>7s} {'R':>7s} {'F1':>7s}")
            for m, summ in by_model.items():
                b = summ[rule]
                print(f"    {m:<10s} {b['TP']:>8d} {b['FP']:>8d} {b['FN']:>8d} "
                      f"{b['precision']:>7.4f} {b['recall']:>7.4f} {b['f1']:>7.4f}")
        # category breakdown
        if any(s["fp_by_category_iop"] for s in by_model.values()):
            print(f"  FP by category (IoP):")
            cats = sorted({c for s in by_model.values() for c in s["fp_by_category_iop"]})
            print(f"    {'model':<10s}" + "".join(f"{c:>12s}" for c in cats))
            for m, summ in by_model.items():
                row = f"    {m:<10s}" + "".join(
                    f"{summ['fp_by_category_iop'].get(c, 0):>12d}" for c in cats)
                print(row)
        if any(s["confuser_any_det_rate"] is not None for s in by_model.values()):
            print(f"  Confuser any-det rate (lower = better):")
            for m, summ in by_model.items():
                rate = summ["confuser_any_det_rate"]
                if rate is None: continue
                print(f"    {m:<10s} {rate*100:>6.2f}%   "
                      f"({summ['n_confuser_frames']:,} frames)")

    out_json = OUT_ROOT / "comparison.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
