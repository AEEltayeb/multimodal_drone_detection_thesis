"""Quick comparison of 3 RGB YOLO models on Anti-UAV and Svanström.
Runs eval_model.evaluate_model() for each, then prints a combined table.
"""
import sys, time, json, cv2
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import compute_prf, score_detections, compute_frame_metrics, iou_iop, classify_size, SIZE_BUCKETS, score_per_size
from datasets import load_config, resolve_path, ImageDataset, detect_category

MODELS = {
    "baseline_trained": str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
    "retrained_v2":     str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
    "ft3_1280":         str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"),
    "ft4_1280":         str(REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"),
}

DATASETS = {
    "antiuav": {
        "path": "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB",
        "stride": 85,   # ~1004 images from 85374
        "categories": ["DRONE"],
        "has_drones": True,
    },
    "svanstrom": {
        "path": "G:/drone/svanstrom_paired/RGB",
        "stride": 29,   # ~990 images from 28710
        "categories": ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"],
        "has_drones": True,
    },
    "confuser_test": {
        "path": "G:/drone/rgb_confusers_merged/images/test",
        "stride": 3,    # ~878 imgs
        "categories": [],
        "has_drones": False,
    },
    "selcom_val": {
        "path": "G:/drone/_finetune_selcom_mixed_ft2/images/val",
        "stride": 1,    # 311 images, use all
        "categories": ["DRONE"],
        "has_drones": True,
        "iop": True,    # selcom uses IoP@0.5
    },
}


def run_eval(model_name, weights, ds_name, ds_info):
    from ultralytics import YOLO
    model = YOLO(weights)

    has_drones = ds_info.get("has_drones", True)

    if has_drones:
        ds_path = Path(ds_info["path"])
        img_dir = ds_path / "images" if (ds_path / "images").exists() else ds_path
        ds = ImageDataset(img_dir)
        images = ds.list_images()[::ds_info["stride"]]
    else:
        # Confuser eval: no GT labels, just count detections
        img_dir = Path(ds_info["path"])
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        images = images[::ds_info["stride"]]

    print(f"\n  [{model_name} on {ds_name}] {len(images):,} images")

    totals = {rule: {"tp": 0, "fp": 0, "fn": 0} for rule in ("iou", "iop")}
    fp_by_cat = {rule: {} for rule in ("iou", "iop")}
    sizes = {"small": 0, "medium": 0, "large": 0}
    halluc_frames = 0
    total_dets = 0

    t0 = time.time()
    for idx, img_path in enumerate(images):
        if has_drones:
            frame = ds.load_frame(img_path)
            if frame is None:
                continue
            img, gt, w, h = frame["img"], frame["gt"], frame["w"], frame["h"]
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = []  # no GT

        res = model.predict(img, conf=0.25, verbose=False, imgsz=1280)
        boxes = res[0].boxes
        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_val = float(boxes.conf[i])
            dets.append(((float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3])), conf_val))

        total_dets += len(dets)
        if dets:
            halluc_frames += 1

        for d_box, _ in dets:
            sz = classify_size(d_box, w, h)
            sizes[sz] += 1

        for rule in ("iou", "iop"):
            tp, fp, fn = score_detections(dets, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
            totals[rule]["tp"] += tp
            totals[rule]["fp"] += fp
            totals[rule]["fn"] += fn

            # FP category attribution (only for svanstrom)
            if ds_name == "svanstrom" and fp > 0 and has_drones:
                cat = frame.get("category", "OTHER")
                if cat not in fp_by_cat[rule]:
                    fp_by_cat[rule][cat] = 0
                fp_by_cat[rule][cat] += fp

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps_rate = (idx + 1) / elapsed
            remaining = len(images) - (idx + 1)
            eta = remaining / fps_rate
            print(f"    {idx+1:>5,}/{len(images):,}  {fps_rate:.1f} fps  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  [{model_name}] Done in {elapsed:.0f}s ({len(images)/elapsed:.1f} fps)")

    return {
        "model": model_name,
        "dataset": ds_name,
        "n_images": len(images),
        "totals": totals,
        "fp_by_cat": fp_by_cat,
        "sizes": sizes,
        "halluc_frames": halluc_frames,
        "total_dets": total_dets,
        "elapsed": elapsed,
    }


def main():
    all_results = []
    
    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")
        
        for model_name, weights in MODELS.items():
            result = run_eval(model_name, weights, ds_name, ds_info)
            all_results.append(result)
    
    # Print combined tables
    print(f"\n{'='*70}")
    print("  COMBINED RESULTS")
    print(f"{'='*70}")
    
    for ds_name in DATASETS:
        ds_info = DATASETS[ds_name]
        has_drones = ds_info.get("has_drones", True)

        if not has_drones:
            # Confuser table: halluc rate only
            print(f"\n  CONFOUSER TEST (hallucination rate)")
            print(f"  {'model':<22s} {'n_imgs':>7s} {'dets':>7s} {'halluc':>7s} {'hall%':>8s}")
            print(f"  {'-'*52}")
            for r in all_results:
                if r["dataset"] != ds_name:
                    continue
                hr = r["halluc_frames"] / max(r["n_images"], 1)
                print(f"  {r['model']:<22s} {r['n_images']:>7d} {r['total_dets']:>7d} "
                      f"{r['halluc_frames']:>7d} {hr:>7.2%}")
            continue

        # Drone-bearing datasets: P/R/F1
        rule = "iop" if ds_info.get("iop") or ds_name == "svanstrom" else "iou"
        
        print(f"\n  {ds_name.upper()} ({rule.upper()} matching)")
        print(f"  {'model':<22s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
        print(f"  {'-'*62}")
        
        for r in all_results:
            if r["dataset"] != ds_name:
                continue
            t = r["totals"][rule]
            m = compute_prf(t["tp"], t["fp"], t["fn"])
            print(f"  {r['model']:<22s} {t['tp']:>6d} {t['fp']:>6d} {t['fn']:>6d} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        
        # FP by category for svanstrom
        if ds_name == "svanstrom":
            print(f"\n  FP by category ({rule.upper()}):")
            cats = set()
            for r in all_results:
                if r["dataset"] == ds_name:
                    cats.update(r["fp_by_cat"][rule].keys())
            cats = sorted(cats)
            header = f"  {'model':<22s}" + "".join(f" {c:>10s}" for c in cats) + f" {'TOTAL':>8s}"
            print(header)
            for r in all_results:
                if r["dataset"] != ds_name:
                    continue
                fp_cats = r["fp_by_cat"][rule]
                total_fp = sum(fp_cats.values())
                row = f"  {r['model']:<22s}"
                for c in cats:
                    row += f" {fp_cats.get(c, 0):>10d}"
                row += f" {total_fp:>8d}"
                print(row)
    
    # Save results
    out_dir = EVAL_DIR / "results" / "_rgb_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = []
    for r in all_results:
        save_data.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "n_images": r["n_images"],
            "iou": r["totals"]["iou"],
            "iop": r["totals"]["iop"],
            "fp_by_cat_iou": r["fp_by_cat"]["iou"],
            "fp_by_cat_iop": r["fp_by_cat"]["iop"],
            "sizes": r["sizes"],
            "elapsed_s": round(r["elapsed"], 1),
        })
    
    out_path = out_dir / "rgb_comparison.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
