"""
eval_rgb_finetune_ablation.py — Evaluate RGB finetune ablations:

  1. old       — best_pre_finetune.pt  @ conf 0.25  (baseline)
  2. epoch2    — epoch2.pt             @ conf 0.25
  3. best@0.30 — best.pt (epoch3)      @ conf 0.30
  4. ep2@0.30  — epoch2.pt             @ conf 0.30
  5. conf_sweep — epoch2.pt            @ 0.10,0.15,...,0.50

Uses the same test corpora and scoring as eval_rgb_finetune.py.

Usage:
    python classifier/eval_rgb_finetune_ablation.py
    python classifier/eval_rgb_finetune_ablation.py --configs old epoch2
    python classifier/eval_rgb_finetune_ablation.py --datasets antiuav svanstrom
    python classifier/eval_rgb_finetune_ablation.py --sweep-only
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
OUT_ROOT   = SCRIPT_DIR / "runs" / "rgb_finetune_ablation"

# ── Weights ───────────────────────────────────────────────────────
W_OLD    = REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt"
W_BEST   = REPO / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt"
W_EP2    = REPO / "RGB model" / "Yolo26n_hardneg" / "weights" / "epoch2.pt"

CONFIGS = {
    "old":       (W_OLD,  0.25),
    "epoch2":    (W_EP2,  0.25),
    "best@0.30": (W_BEST, 0.30),
    "ep2@0.30":  (W_EP2,  0.30),
}

# ── Dataset paths ─────────────────────────────────────────────────
ANTIUAV_RGB_IMG = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_RGB_LBL = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels")
SVAN_RGB_IMG    = Path(r"G:/drone/svanstrom_paired/RGB/images")
SVAN_RGB_LBL    = Path(r"G:/drone/svanstrom_paired/RGB/labels")
AIRPLANE_TEST   = Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb/test/images")
NEW_DS_TEST_IMG = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb/test/images")
NEW_DS_TEST_LBL = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb/test/labels")
HELI_TEST       = Path(r"G:/drone/finetune_dataset/images/test/helicopter")
DRONE_DSET_IMG  = Path(r"G:/drone/dataset/dataset/images/test")
DRONE_DSET_LBL  = Path(r"G:/drone/dataset/dataset/labels/test")
NEW_DS_DRONE_CLASS = 2
SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
IMG_EXTS  = {".jpg", ".jpeg", ".png"}

# ── Helpers ───────────────────────────────────────────────────────

def svan_category(stem):
    for c in SVAN_CATS:
        if f"_{c}_" in stem:
            return c
    return "OTHER"

def read_gt(path, w, h, drop_class=None):
    boxes = []
    if not path or not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5: continue
        try: cls = int(p[0])
        except ValueError: continue
        if drop_class is not None and cls == drop_class:
            return None
        if cls != 0 and drop_class is None:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return boxes

def iou_iop(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0., 0.
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if (aa+bb-inter)>0 else 0., inter/aa if aa>0 else 0.

def score_dets(dets, gts, rule="iou", thr=0.5):
    tp = fp = 0; used = set()
    for db, _c in dets:
        best_i, best_s = -1, 0.
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(db, g)
            s = iu if rule == "iou" else ip
            if s > best_s: best_s, best_i = s, gi
        if best_s >= thr and best_i not in used:
            tp += 1; used.add(best_i)
        else: fp += 1
    return tp, fp, len(gts) - len(used)

def stride(items, n):
    if n <= 0 or n >= len(items): return list(items)
    step = len(items) / float(n)
    return [items[int(i*step)] for i in range(n)]

# ── Frame iterators ───────────────────────────────────────────────

def iter_antiuav(sample=8537):
    imgs = sorted(p for p in ANTIUAV_RGB_IMG.iterdir() if p.suffix.lower() in IMG_EXTS)
    for p in stride(imgs, sample):
        yield {"img": p, "lbl": ANTIUAV_RGB_LBL/(p.stem+".txt"),
               "category": "DRONE", "kind": "drone"}

def iter_dataset_rgb():
    for p in sorted(p for p in DRONE_DSET_IMG.iterdir() if p.suffix.lower() in IMG_EXTS):
        yield {"img": p, "lbl": DRONE_DSET_LBL/(p.stem+".txt"),
               "category": "DRONE", "kind": "drone"}

def iter_svanstrom():
    for p in sorted(p for p in SVAN_RGB_IMG.iterdir() if p.suffix.lower() in IMG_EXTS):
        cat = svan_category(p.stem)
        yield {"img": p, "lbl": SVAN_RGB_LBL/(p.stem+".txt"),
               "category": cat, "kind": "drone" if cat == "DRONE" else "confuser"}

def iter_confuser(d, prefix, drop_class=None, lbl_dir=None):
    for p in sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS):
        if drop_class is not None and lbl_dir is not None:
            lbl = lbl_dir/(p.stem+".txt")
            if lbl.exists():
                skip = any(ln.strip().split()[0] == str(drop_class)
                           for ln in lbl.read_text().splitlines()
                           if ln.strip().split())
                if skip: continue
        yield {"img": p, "lbl": None, "category": prefix.upper(), "kind": "confuser"}

DATASETS = {
    "antiuav":     lambda: list(iter_antiuav()),
    "dataset_rgb": lambda: list(iter_dataset_rgb()),
    "svanstrom":   lambda: list(iter_svanstrom()),
    "airplane":    lambda: list(iter_confuser(AIRPLANE_TEST, "airplane")),
    "new_dataset": lambda: list(iter_confuser(NEW_DS_TEST_IMG, "new_dataset",
                                              NEW_DS_DRONE_CLASS, NEW_DS_TEST_LBL)),
    "helicopter":  lambda: list(iter_confuser(HELI_TEST, "helicopter")),
}

# ── Run raw YOLO once, cache boxes ────────────────────────────────

def run_yolo_raw(model, frames):
    """Run YOLO at conf=0.01 to get ALL detections. Filter by conf later."""
    all_dets = []
    t0 = time.time()
    for idx, f in enumerate(frames):
        img = cv2.imread(str(f["img"]))
        if img is None:
            all_dets.append(([], f)); continue
        h, w = img.shape[:2]
        gts = read_gt(f["lbl"], w, h) if f["kind"] == "drone" and f["lbl"] else []
        res = model.predict(img, conf=0.01, iou=0.45, imgsz=640,
                            verbose=False, device=0, max_det=300)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(confs)):
                dets.append(((float(xyxy[i,0]), float(xyxy[i,1]),
                              float(xyxy[i,2]), float(xyxy[i,3])),
                             float(confs[i])))
        f_copy = dict(f); f_copy["gts"] = gts
        all_dets.append((dets, f_copy))
        if (idx+1) % 500 == 0:
            fps = (idx+1)/(time.time()-t0)
            print(f"    {idx+1:>6,}/{len(frames):,}  {fps:.1f} fps")
    return all_dets

def eval_at_conf(cached, conf_thr):
    """Score cached detections at a given conf threshold."""
    c_iou = {"tp":0,"fp":0,"fn":0}
    c_iop = {"tp":0,"fp":0,"fn":0}
    fp_cat_iou = Counter(); fp_cat_iop = Counter()
    n_drone = n_conf = conf_det = 0
    for dets, f in cached:
        filt = [(b,c) for b,c in dets if c >= conf_thr]
        gts = f.get("gts", [])
        tp_u,fp_u,fn_u = score_dets(filt, gts, "iou")
        tp_p,fp_p,fn_p = score_dets(filt, gts, "iop")
        c_iou["tp"]+=tp_u; c_iou["fp"]+=fp_u; c_iou["fn"]+=fn_u
        c_iop["tp"]+=tp_p; c_iop["fp"]+=fp_p; c_iop["fn"]+=fn_p
        fp_cat_iou[f["category"]] += fp_u
        fp_cat_iop[f["category"]] += fp_p
        if f["kind"] == "drone": n_drone += 1
        else:
            n_conf += 1
            if filt: conf_det += 1
    def m(c):
        tp,fp,fn = c["tp"],c["fp"],c["fn"]
        p = tp/(tp+fp) if tp+fp else 0.
        r = tp/(tp+fn) if tp+fn else 0.
        f1 = 2*p*r/(p+r) if p+r else 0.
        return {"TP":tp,"FP":fp,"FN":fn,"P":round(p,4),"R":round(r,4),"F1":round(f1,4)}
    return {
        "iou": m(c_iou), "iop": m(c_iop),
        "fp_cat_iou": dict(fp_cat_iou), "fp_cat_iop": dict(fp_cat_iop),
        "n_drone": n_drone, "n_confuser": n_conf,
        "confuser_det_rate": round(conf_det/n_conf,4) if n_conf else None,
    }

# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--sweep-only", action="store_true",
                    help="skip named configs, only run conf sweep on epoch2")
    ap.add_argument("--sweep-confs", type=float, nargs="+",
                    default=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Deduplicate which weight files we actually need to load
    needed_weights = {}  # path -> tag
    if not args.sweep_only:
        for cfg in args.configs:
            w, _ = CONFIGS[cfg]
            needed_weights[str(w)] = w
    needed_weights[str(W_EP2)] = W_EP2  # always need epoch2 for sweep

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*72}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'='*72}")
        frames = DATASETS[ds_name]()
        print(f"  {len(frames):,} frames")
        all_results[ds_name] = {}

        # Cache inference per unique weight file
        cache = {}
        for wpath_str, wpath in needed_weights.items():
            if wpath_str in cache: continue
            print(f"\n  Loading {wpath.name} ...")
            model = YOLO(str(wpath))
            print(f"  Running inference (conf=0.01) on {len(frames):,} frames ...")
            cache[wpath_str] = run_yolo_raw(model, frames)
            del model  # free GPU

        # Named configs
        if not args.sweep_only:
            for cfg in args.configs:
                w, conf = CONFIGS[cfg]
                print(f"\n  [{cfg}] weights={w.name}  conf={conf}")
                r = eval_at_conf(cache[str(w)], conf)
                all_results[ds_name][cfg] = r
                b = r["iop"]
                print(f"    IoP: TP={b['TP']:>8,} FP={b['FP']:>8,} FN={b['FN']:>8,}  "
                      f"P={b['P']:.4f} R={b['R']:.4f} F1={b['F1']:.4f}")
                if r["confuser_det_rate"] is not None:
                    print(f"    Confuser det rate: {r['confuser_det_rate']*100:.2f}%")

        # Confidence sweep on epoch2
        print(f"\n  CONFIDENCE SWEEP (epoch2.pt on {ds_name}):")
        sweep = {}
        for c in args.sweep_confs:
            r = eval_at_conf(cache[str(W_EP2)], c)
            sweep[str(c)] = r
            b = r["iop"]
            cdr = f"  confDet={r['confuser_det_rate']*100:.1f}%" if r["confuser_det_rate"] is not None else ""
            print(f"    conf={c:.2f}  IoP F1={b['F1']:.4f}  P={b['P']:.4f}  R={b['R']:.4f}  "
                  f"FP={b['FP']:>6,}{cdr}")
        all_results[ds_name]["sweep_epoch2"] = sweep

    # Save
    out_json = OUT_ROOT / "ablation_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_json}")

    # CSV summary
    csv_path = OUT_ROOT / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","config","conf","IoU_F1","IoP_F1","IoP_P","IoP_R",
                     "IoP_TP","IoP_FP","IoP_FN","confuser_det_rate"])
        for ds, by_cfg in all_results.items():
            for cfg, data in by_cfg.items():
                if cfg == "sweep_epoch2":
                    for c_str, r in data.items():
                        w.writerow([ds, "sweep_ep2", c_str,
                                    r["iou"]["F1"], r["iop"]["F1"],
                                    r["iop"]["P"], r["iop"]["R"],
                                    r["iop"]["TP"], r["iop"]["FP"], r["iop"]["FN"],
                                    r["confuser_det_rate"] or ""])
                else:
                    conf = CONFIGS.get(cfg, (None, "?"))[1]
                    w.writerow([ds, cfg, conf,
                                data["iou"]["F1"], data["iop"]["F1"],
                                data["iop"]["P"], data["iop"]["R"],
                                data["iop"]["TP"], data["iop"]["FP"], data["iop"]["FN"],
                                data["confuser_det_rate"] or ""])
    print(f"Saved: {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
