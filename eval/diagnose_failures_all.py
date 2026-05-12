"""Run Svanström category diagnosis + confuser test for hardneg_v3more and retrained_v2."""
import json, sys, time, csv
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
from metrics import score_detections
from datasets import ImageDataset, detect_category

MODELS = {
    "hardneg_v3more": str(REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt"),
    "retrained_v2":   str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
}
SVANSTROM_RGB = Path("G:/drone/svanstrom_paired/RGB")
CONFUSER_ROOT = Path("G:/drone/rgb_confusers_merged")
RGB_CONF = 0.25
IMGSZ = 1280

def run_svan(model, name):
    ds = ImageDataset(SVANSTROM_RGB / "images", SVANSTROM_RGB / "labels")
    imgs = ds.list_images()[::9]
    print(f"\n[{name}] Svanström {len(imgs)} frames @ 1280")
    cat_stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"n":0,"n_det":0,"fp_confs":[],"miss_sizes":[]})
    t0 = time.time()
    for idx, p in enumerate(imgs):
        f = ds.load_frame(p)
        if f is None: continue
        res = model.predict(f["img"], conf=RGB_CONF, verbose=False, imgsz=IMGSZ)
        boxes = res[0].boxes
        dets = [((float(boxes.xyxy[i][0]),float(boxes.xyxy[i][1]),float(boxes.xyxy[i][2]),float(boxes.xyxy[i][3])),float(boxes.conf[i])) for i in range(len(boxes))]
        tp,fp,fn = score_detections(dets, f["gt"], rule="iop", iop_thr=0.5)
        s = cat_stats[f["category"]]
        s["tp"]+=tp; s["fp"]+=fp; s["fn"]+=fn; s["n"]+=1
        if dets: s["n_det"]+=1
        for d,c in dets:
            if fp > 0: s["fp_confs"].append(c)
        if fn > 0 and f["gt"]:
            for g in f["gt"]:
                s["miss_sizes"].append((g[2]-g[0])*(g[3]-g[1])/(f["w"]*f["h"]))
        if (idx+1)%500==0: print(f"  {idx+1}/{len(imgs)}  {(idx+1)/(time.time()-t0):.1f} fps")
    print(f"\n[{name}] Results:")
    print(f"  {'Cat':12s} {'Frames':>6s} {'DetRate':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>6s} {'Rec':>6s} {'MedFPConf':>9s}")
    print("  "+"-"*70)
    rows = []
    for cat in sorted(cat_stats):
        s=cat_stats[cat]; p_=s["tp"]/max(s["tp"]+s["fp"],1); r_=s["tp"]/max(s["tp"]+s["fn"],1); dr=s["n_det"]/max(s["n"],1)
        mc = f"{sorted(s['fp_confs'])[len(s['fp_confs'])//2]:.3f}" if s["fp_confs"] else "—"
        med_miss = f"{sorted(s['miss_sizes'])[len(s['miss_sizes'])//2]:.4f}" if s["miss_sizes"] else ""
        print(f"  {cat:12s} {s['n']:6d} {dr:7.1%} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d} {p_:6.3f} {r_:6.3f} {mc:>9s}")
        if s["miss_sizes"]:
            ms=s["miss_sizes"]; print(f"    └─ Missed GT: min={min(ms):.4f} med={sorted(ms)[len(ms)//2]:.4f} max={max(ms):.4f}")
        rows.append({
            "model": name, "category": cat, "frames": s["n"],
            "det_rate": f"{dr:.1%}", "TP": s["tp"], "FP": s["fp"], "FN": s["fn"],
            "precision": f"{p_:.3f}", "recall": f"{r_:.3f}",
            "med_fp_conf": mc if mc != "—" else "",
            "missed_gt_med_area": med_miss
        })
    return rows

def run_confuser(model, name, split="test"):
    img_dir = CONFUSER_ROOT / "images" / split
    images = sorted(img_dir.glob("*.*"))
    print(f"\n[{name}] Confusers {split}: {len(images)} images")
    src_stats = defaultdict(lambda: {"n":0,"n_det":0,"confs":[]})
    t0 = time.time()
    for idx, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None: continue
        stem = p.stem
        if stem.startswith("airplane_"): src="airplane"
        elif stem.startswith("helicopter_"): src="helicopter"
        elif stem.startswith("bird_") or stem.startswith("raihanrsd_"): src="bird"
        elif "_BIRD_" in stem: src="svan_bird"
        elif "_AIRPLANE_" in stem: src="svan_airplane"
        elif "_HELICOPTER_" in stem: src="svan_helicopter"
        else: src="other"
        res = model.predict(img, conf=RGB_CONF, verbose=False, imgsz=IMGSZ)
        n = len(res[0].boxes)
        src_stats[src]["n"]+=1
        if n>0:
            src_stats[src]["n_det"]+=1
            for i in range(n): src_stats[src]["confs"].append(float(res[0].boxes.conf[i]))
        if (idx+1)%500==0: print(f"  {idx+1}/{len(images)}  {(idx+1)/(time.time()-t0):.1f} fps")
    print(f"\n[{name}] Confuser hallucination:")
    print(f"  {'Source':16s} {'Images':>6s} {'Halluc':>6s} {'Rate':>7s} {'AvgConf':>7s}")
    print("  "+"-"*50)
    rows = []
    for src in sorted(src_stats):
        s=src_stats[src]; r=s["n_det"]/max(s["n"],1); ac=np.mean(s["confs"]) if s["confs"] else 0
        print(f"  {src:16s} {s['n']:6d} {s['n_det']:6d} {r:7.1%} {ac:7.3f}")
        rows.append({
            "model": name, "source": src, "images": s["n"],
            "hallucinations": s["n_det"], "halluc_rate": f"{r:.1%}",
            "avg_conf": f"{ac:.3f}"
        })
    return rows

if __name__ == "__main__":
    from ultralytics import YOLO
    out_dir = EVAL_DIR / "results" / "_failure_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_svan_rows = []
    all_confuser_rows = []

    for name, weights in MODELS.items():
        print(f"\n{'='*60}\n  Loading {name}\n{'='*60}")
        m = YOLO(weights)
        all_svan_rows.extend(run_svan(m, name))
        all_confuser_rows.extend(run_confuser(m, name))

    # Write Svanstrom CSV (append to existing if baseline already there)
    svan_csv = out_dir / "svanstrom_1280_by_category.csv"
    existing_rows = []
    if svan_csv.exists():
        with open(svan_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["model"] not in MODELS]
    with open(svan_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "category", "frames", "det_rate", "TP", "FP", "FN",
                "precision", "recall", "med_fp_conf", "missed_gt_med_area"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        for r in all_svan_rows:
            w.writerow(r)
    print(f"\n[SAVED] {svan_csv}")

    # Write confuser CSV (append to existing)
    confuser_csv = out_dir / "confuser_test_hallucination.csv"
    existing_rows = []
    if confuser_csv.exists():
        with open(confuser_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["model"] not in MODELS]
    with open(confuser_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["model", "source", "images", "hallucinations", "halluc_rate", "avg_conf"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        for r in all_confuser_rows:
            w.writerow(r)
    print(f"\n[SAVED] {confuser_csv}")

    print("\n[ALL DONE]")

