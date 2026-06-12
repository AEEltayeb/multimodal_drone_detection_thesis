"""
finetune_selcom.py — Fine-tune Yolo26n_trained on the selcom TestDrone dataset.

Stages:
  1. Stage dataset  (calls build_selcom_mixed_ft1.py by default; --no-mix for
                     pure-selcom staging via build_selcom_ft1.py)
  2. Fine-tune YOLO (outputs training/<name>/weights/best.pt)
  3. Eval old vs new on dataset_rgb test split + selcom val split
  4. Print comparison table + save comparison.json

Usage:
    python "training/finetune_selcom.py"
    python "training/finetune_selcom.py" --name Yolo26n_selcom_mixed_ft1 --ratio 0.20
    python "training/finetune_selcom.py" --skip-stage --skip-train  # eval-only
    python "training/finetune_selcom.py" --no-mix                   # pure selcom (no replay)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT          = Path(__file__).resolve().parents[1]
BUILDER_PURE  = ROOT / "RGB model" / "dataset preparation" / "build_selcom_ft1.py"
BASE_MODEL    = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"
FT3_MODEL     = ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"
DATA_YAML_PURE  = Path(r"G:/drone/_finetune_selcom_ft1/data.yaml")
SELCOM_VAL_PURE  = Path(r"G:/drone/_finetune_selcom_ft1")

# Mixed dataset versions: --ft 1 -> ft1 paths, --ft 2 -> ft2 paths
BUILDERS_MIXED = {
    1: ROOT / "RGB model" / "dataset preparation" / "build_selcom_mixed_ft1.py",
    2: ROOT / "RGB model" / "dataset preparation" / "build_selcom_mixed_ft2.py",
    3: ROOT / "RGB model" / "dataset preparation" / "build_selcom_mixed_ft3.py",
    4: ROOT / "RGB model" / "dataset preparation" / "build_selcom_confuser_ft4.py",
}
DATA_YAMLS_MIXED = {
    1: Path(r"G:/drone/_finetune_selcom_mixed_ft1/data.yaml"),
    2: Path(r"G:/drone/_finetune_selcom_mixed_ft2/data.yaml"),
    3: Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3/data.yaml"),
    4: Path(r"C:/drone_cache/_finetune_selcom_confuser_ft4/data.yaml"),
}
SELCOM_VALS_MIXED = {
    1: Path(r"G:/drone/_finetune_selcom_mixed_ft1"),
    2: Path(r"G:/drone/_finetune_selcom_mixed_ft2"),
    3: Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3"),
    4: Path(r"C:/drone_cache/_finetune_selcom_confuser_ft4"),
}

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
CONF       = 0.25
IOP_THRESH = 0.5


# ── Eval helpers (mirrored from finetune_and_eval.py) ────────────────────────

def load_gt(lbl_path):
    if lbl_path is None or not lbl_path.exists():
        return []
    boxes = []
    for ln in lbl_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        if int(parts[0]) != 0:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return boxes


def iop(pred, gt):
    ix1 = max(pred[0], gt[0]); iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2]); iy2 = min(pred[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area = max(1e-9, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    return inter / pred_area


def img_iter(d, stride=1):
    if not d.exists():
        return []
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return imgs[::stride] if stride > 1 else imgs


def eval_model(model_path, model_name, datasets, out_dir):
    from ultralytics import YOLO

    print(f"\n{'='*72}")
    print(f"EVALUATING: {model_name}  ({model_path})")
    print(f"{'='*72}")

    model = YOLO(str(model_path))
    results = {}

    for ds_name, ds_cfg in datasets.items():
        img_dir   = ds_cfg["images"]
        lbl_dir   = ds_cfg.get("labels")
        has_drones = ds_cfg.get("has_drones", True)
        stride     = ds_cfg.get("stride", 1)
        imgsz      = ds_cfg.get("imgsz", 640)

        imgs = img_iter(img_dir, stride)
        if not imgs:
            print(f"  [{ds_name}] no images found at {img_dir}")
            continue

        print(f"\n  [{ds_name}] {len(imgs)} images  stride={stride}  imgsz={imgsz}")
        tp = fp = fn = 0
        total_dets = frames_with_det = 0
        t0 = time.perf_counter()

        for i, img_path in enumerate(imgs):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            r = model.predict(frame, conf=CONF, iou=0.45, imgsz=imgsz,
                              verbose=False, device=0)[0]
            preds = []
            if r.boxes is not None:
                for j in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                    preds.append((x1/w, y1/h, x2/w, y2/h, float(r.boxes.conf[j])))

            total_dets += len(preds)
            if preds:
                frames_with_det += 1

            if has_drones and lbl_dir:
                gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt"))
                matched_gt = set()
                for px1, py1, px2, py2, pc in preds:
                    best_iop, best_j = 0, -1
                    for j, gb in enumerate(gt_boxes):
                        s = iop((px1, py1, px2, py2), gb)
                        if s > best_iop:
                            best_iop, best_j = s, j
                    if best_iop >= IOP_THRESH and best_j not in matched_gt:
                        tp += 1
                        matched_gt.add(best_j)
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(matched_gt)
            else:
                fp += len(preds)

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(imgs)} ({time.perf_counter()-t0:.0f}s)")

        elapsed = time.perf_counter() - t0
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        det_rate = frames_with_det / max(len(imgs), 1)

        r_dict = dict(
            n_images=len(imgs), tp=tp, fp=fp, fn=fn,
            precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
            total_dets=total_dets, frames_with_det=frames_with_det,
            any_det_rate=round(det_rate, 4), elapsed_s=round(elapsed, 1),
        )
        results[ds_name] = r_dict
        print(f"    P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
              f"TP={tp}  FP={fp}  FN={fn}  det%={det_rate:.2%}  ({elapsed:.0f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {out_file}")
    return results


def print_comparison(all_results, out_dir):
    print(f"\n{'='*72}\nCOMPARISON TABLE\n{'='*72}")
    models = list(all_results.keys())
    datasets = list(next(iter(all_results.values())).keys())
    for ds in datasets:
        print(f"\n  {ds}:")
        print(f"    {'Model':<20s} {'P':>7s} {'R':>7s} {'F1':>7s} "
              f"{'TP':>7s} {'FP':>7s} {'FN':>7s} {'det%':>8s}")
        print(f"    {'-'*75}")
        for m in models:
            r = all_results[m].get(ds, {})
            print(f"    {m:<20s} {r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
                  f"{r.get('f1',0):>7.4f} {r.get('tp',0):>7d} {r.get('fp',0):>7d} "
                  f"{r.get('fn',0):>7d} {r.get('any_det_rate',0):>7.2%}")

    comp_file = out_dir / "comparison.json"
    comp_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full comparison: {comp_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft",         type=int, default=2, choices=[1, 2, 3, 4],
                    help="Which selcom dataset version: 1 (TestDrone footage), 2 (full labeled, val=selcom-only), 3 (full labeled, val=50/50 baseline+selcom), 4 (ft3 + confuser hard-negs)")
    ap.add_argument("--name",       default=None,
                    help="Output run name (default: Yolo26n_selcom_mixed_ft<N>[_1280])")
    ap.add_argument("--imgsz",      type=int, default=640,
                    help="Train + eval image size. Use 1280 for small-drone CCTV.")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch",      type=int,   default=8)
    ap.add_argument("--freeze",     type=int,   default=10)
    ap.add_argument("--lr0",        type=float, default=1e-5)
    ap.add_argument("--workers",    type=int,   default=2)
    ap.add_argument("--ratio",      type=float, default=0.20,
                    help="Selcom fraction of training set (default 0.20 = 20%% selcom)")
    ap.add_argument("--no-mix",     action="store_true",
                    help="Use pure selcom dataset (no replay mixing)")
    ap.add_argument("--skip-stage", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval",  action="store_true")
    ap.add_argument("--dataset-rgb-stride", type=int, default=5,
                    help="Stride over dataset_rgb test split. Default 5 (fast). Use 1 for full eval.")
    ap.add_argument("--skip-old-eval", action="store_true",
                    help="Skip evaluating the old_baseline (only eval the fine-tuned model)")
    ap.add_argument("--clean",      action="store_true",
                    help="Wipe staged dataset before re-staging")
    args = ap.parse_args()

    use_mixed  = not args.no_mix
    _imgsz_suffix = "" if args.imgsz == 640 else f"_{args.imgsz}"
    if args.ft == 4:
        name = args.name or f"Yolo26n_selcom_confuser_ft4{_imgsz_suffix}"
    else:
        name = args.name or f"Yolo26n_selcom_mixed_ft{args.ft}{_imgsz_suffix}"
    DATA_YAML  = DATA_YAMLS_MIXED[args.ft]  if use_mixed else DATA_YAML_PURE
    SELCOM_VAL = SELCOM_VALS_MIXED[args.ft] if use_mixed else SELCOM_VAL_PURE
    # ft4 fine-tunes from ft3_1280, not from baseline
    base_model = FT3_MODEL if args.ft == 4 else BASE_MODEL
    new_model_path = ROOT / "RGB model" / name / "weights" / "best.pt"

    # ── Stage ────────────────────────────────────────────────────────────────
    if not args.skip_stage:
        print("="*72)
        builder = BUILDERS_MIXED[args.ft] if use_mixed else BUILDER_PURE
        print(f"PHASE 1 — Stage {'mixed' if use_mixed else 'pure selcom'} dataset")
        print("="*72)
        import subprocess, os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [sys.executable, str(builder)]
        if args.clean:
            cmd.append("--clean")
        if use_mixed:
            cmd += ["--ratio", str(args.ratio)]
        p = subprocess.run(cmd, env=env)
        if p.returncode != 0:
            print(f"[fatal] staging failed (exit {p.returncode})")
            sys.exit(p.returncode)
    else:
        print("[skip] dataset staging")

    if not DATA_YAML.exists():
        print(f"[fatal] {DATA_YAML} missing — run without --skip-stage first")
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────────────────
    if not args.skip_train:
        from ultralytics import YOLO

        print("\n" + "="*72)
        print("PHASE 2 — Fine-tune")
        print("="*72)

        if not base_model.exists():
            print(f"[fatal] base model missing: {base_model}")
            sys.exit(1)

        print(f"Base model: {base_model}")
        model = YOLO(str(base_model))
        train_kwargs = dict(
            data=str(DATA_YAML),
            epochs=args.epochs,
            patience=2 if args.ft == 4 else 3,
            batch=args.batch,
            imgsz=args.imgsz,
            device=0,
            amp=True,
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=0.01,
            freeze=args.freeze,
            cos_lr=True,
            close_mosaic=2,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
            save_period=1,
            workers=args.workers,
            cache=False,
            plots=args.workers > 0,  # disable plots when workers=0 (saves RAM on 4GB GPU)
            project=str(ROOT / "RGB model"),
            name=name,
            pretrained=True,
            exist_ok=True,
            verbose=True,
        )
        print("Training config:")
        for k, v in train_kwargs.items():
            print(f"  {k} = {v}")
        model.train(**train_kwargs)
        print(f"\nBest checkpoint → {new_model_path}")
    else:
        print("[skip] training")

    # ── Eval ─────────────────────────────────────────────────────────────────
    if not args.skip_eval:
        print("\n" + "="*72)
        print("PHASE 3 — Evaluate (old vs new)")
        print("="*72)

        if not new_model_path.exists():
            print(f"[fatal] trained model not found: {new_model_path}")
            sys.exit(1)

        out_dir = ROOT / "runs" / "rgb_finetune_eval" / name

        datasets = {
            # General RGB regression gate (stride configurable; 5=fast, 1=full)
            "dataset_rgb": dict(
                images=Path(r"G:/drone/dataset/dataset/images/test"),
                labels=Path(r"G:/drone/dataset/dataset/labels/test"),
                has_drones=True,
                stride=args.dataset_rgb_stride,
                imgsz=args.imgsz,
            ),
            # Sanity check: did the model learn the selcom distribution?
            "selcom_val": dict(
                images=SELCOM_VAL / "images" / "val",
                labels=SELCOM_VAL / "labels" / "val",
                has_drones=True,
                stride=1,
                imgsz=args.imgsz,
            ),
        }

        all_results = {}
        if not args.skip_old_eval:
            old_path = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt"
            if not old_path.exists():
                old_path = BASE_MODEL
            all_results["old_baseline"] = eval_model(old_path, "old_baseline", datasets, out_dir)
        all_results[name]           = eval_model(new_model_path, name, datasets, out_dir)

        print_comparison(all_results, out_dir)

        # Regression gate (only meaningful when both models were evaluated)
        old_f1  = all_results.get("old_baseline", {}).get("dataset_rgb", {}).get("f1", 0)
        new_f1  = all_results[name].get("dataset_rgb", {}).get("f1", 0)
        delta   = new_f1 - old_f1
        gate_ok = delta >= -0.01 if "old_baseline" in all_results else True
        print(f"\n{'='*72}")
        print(f"REGRESSION GATE  dataset_rgb F1:  old={old_f1:.4f}  new={new_f1:.4f}  Δ={delta:+.4f}")
        print(f"Result: {'PASS ✓' if gate_ok else 'FAIL ✗  — do NOT update production stack'}")
        print(f"{'='*72}")
    else:
        print("[skip] evaluation")

    print("\n" + "="*72 + "\nALL DONE.\n" + "="*72)


if __name__ == "__main__":
    main()
