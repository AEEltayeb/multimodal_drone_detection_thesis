"""
Evaluate all IR model versions on the IR_dset_final dataset.
Produces a comparison table: mAP50, P, R, F1 for each version.

Usage:
    python eval/eval_ir_versions.py [--split test] [--imgsz 640]
"""

import argparse
import json
import csv
import os
from pathlib import Path
from datetime import datetime

# Try to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    exit(1)


# ── Model registry ──────────────────────────────────────────────────
BASE = Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection")

MODELS = {
    "V3":           BASE / "runs" / "IR_FT_dsetV3_aug0_s0" / "weights" / "best.pt",
    "V4":           BASE / "models" / "IR_dsetV4_300ep" / "best.pt",
    "V5":           BASE / "models" / "IR_dsetV5_269ep" / "best.pt",
    "V6":           BASE / "models" / "IR_dsetV6_118ep" / "best.pt",
    "Final":        BASE / "models" / "IR_final_cleaned" / "weights" / "best.pt",
    "v3b (prod)":   BASE / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt",
}

DATASET_YAML = Path(r"G:\drone\IR_dset_final\dataset.yaml")

# ── Output ───────────────────────────────────────────────────────────
OUTDIR = BASE.parent / "es_drone_detection" / "eval" / "results" / "ir_version_comparison"


def run_eval(model_path: Path, data_yaml: Path, split: str, imgsz: int):
    """Run YOLO val and return metrics dict."""
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=16,
        verbose=False,
        plots=False,
    )
    
    # Extract metrics
    metrics = {
        "mAP50":     round(float(results.box.map50), 4),
        "mAP50-95":  round(float(results.box.map), 4),
        "Precision":  round(float(results.box.mp), 4),
        "Recall":     round(float(results.box.mr), 4),
        "F1":         round(2 * float(results.box.mp) * float(results.box.mr) / 
                      max(float(results.box.mp) + float(results.box.mr), 1e-9), 4),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate all IR model versions")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference resolution (default: 640)")
    args = parser.parse_args()

    print(f"═══ IR Model Version Comparison ═══")
    print(f"Dataset:    {DATASET_YAML}")
    print(f"Split:      {args.split}")
    print(f"Imgsz:      {args.imgsz}")
    print(f"Models:     {len(MODELS)}")
    print()

    # Verify all models exist
    missing = []
    for name, path in MODELS.items():
        if not path.exists():
            missing.append(name)
            print(f"  ✗ {name}: {path} NOT FOUND")
        else:
            print(f"  ✓ {name}: {path.name}")
    
    if missing:
        print(f"\n⚠ {len(missing)} models missing: {missing}")
        print("Proceeding with available models...\n")

    # Run evaluations
    results_all = {}
    for name, path in MODELS.items():
        if name in missing:
            continue
        
        print(f"\n{'─'*60}")
        print(f"Evaluating: {name}")
        print(f"  Weights: {path}")
        print(f"{'─'*60}")
        
        try:
            metrics = run_eval(path, DATASET_YAML, args.split, args.imgsz)
            results_all[name] = metrics
            print(f"  mAP50={metrics['mAP50']:.3f}  P={metrics['Precision']:.3f}  "
                  f"R={metrics['Recall']:.3f}  F1={metrics['F1']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results_all[name] = {"error": str(e)}

    # ── Print summary table ──────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  IR Model Comparison — {args.split} split @ imgsz={args.imgsz}")
    print(f"{'═'*70}")
    print(f"  {'Model':<15} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'─'*15} {'─'*7} {'─'*9} {'─'*7} {'─'*7} {'─'*7}")
    
    for name in MODELS:
        if name in missing or "error" in results_all.get(name, {}):
            print(f"  {name:<15} {'—':>7} {'—':>9} {'—':>7} {'—':>7} {'—':>7}")
            continue
        m = results_all[name]
        print(f"  {name:<15} {m['mAP50']:>7.3f} {m['mAP50-95']:>9.3f} "
              f"{m['Precision']:>7.3f} {m['Recall']:>7.3f} {m['F1']:>7.3f}")
    
    print(f"{'═'*70}")

    # ── Save results ─────────────────────────────────────────────
    OUTDIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    # JSON
    out_json = OUTDIR / f"ir_comparison_{args.split}_{args.imgsz}_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump({
            "split": args.split,
            "imgsz": args.imgsz,
            "dataset": str(DATASET_YAML),
            "timestamp": timestamp,
            "results": results_all,
        }, f, indent=2)
    
    # CSV
    out_csv = OUTDIR / f"ir_comparison_{args.split}_{args.imgsz}_{timestamp}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "mAP50", "mAP50-95", "Precision", "Recall", "F1"])
        for name in MODELS:
            if name in missing or "error" in results_all.get(name, {}):
                writer.writerow([name, "", "", "", "", ""])
            else:
                m = results_all[name]
                writer.writerow([name, m["mAP50"], m["mAP50-95"], m["Precision"], m["Recall"], m["F1"]])
    
    print(f"\nResults saved:")
    print(f"  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")


if __name__ == "__main__":
    main()
