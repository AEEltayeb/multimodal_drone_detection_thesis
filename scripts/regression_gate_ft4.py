"""
regression_gate_ft4.py — Multi-surface regression gate for confuser fine-tune.

Compares ft4 metrics against the baseline snapshot (from baseline_snapshot_ft3.py)
and reports PASS/FAIL per surface.

Gate rules:
  - selcom_val F1:        Δ >= -0.01
  - dataset_rgb_test F1:  Δ >= -0.01
  - svanstrom DRONE R:    Δ >= -0.01
  - antiuav F1:           Δ >= -0.005
  - confuser_test halluc: must decrease (Δ < 0)

Usage:
    python scripts/regression_gate_ft4.py
    python scripts/regression_gate_ft4.py --weights "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_snapshot_ft3 import DATASETS, eval_surface, CONF


def main():
    ap = argparse.ArgumentParser(description="Regression gate for ft4")
    ap.add_argument("--weights", default=str(ROOT / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"))
    ap.add_argument("--baseline", default=str(ROOT / "scripts" / "baseline_snapshot.json"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--output", default=str(ROOT / "scripts" / "regression_gate_results.json"))
    args = ap.parse_args()

    # Load baseline
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"[fatal] Baseline snapshot not found: {baseline_path}")
        print("  Run baseline_snapshot_ft3.py first.")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text())
    baseline_surfaces = baseline["surfaces"]

    # Load ft4 model
    ft4_path = Path(args.weights)
    if not ft4_path.exists():
        print(f"[fatal] ft4 model not found: {ft4_path}")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Loading ft4 model: {ft4_path}")
    model = YOLO(str(ft4_path))

    # Eval ft4 on all surfaces
    ft4_results = {}
    for ds_name in DATASETS:
        if ds_name not in baseline_surfaces:
            print(f"  [{ds_name}] SKIP — not in baseline snapshot")
            continue
        result = eval_surface(model, ds_name, DATASETS[ds_name], args.imgsz)
        ft4_results[ds_name] = result

    # ── Regression gate ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"REGRESSION GATE: ft4 vs ft3 baseline")
    print(f"{'='*72}")

    gate_rules = [
        # (surface, metric_path, direction, threshold, label)
        ("selcom_val", "f1", ">=", -0.01, "Selcom val F1"),
        ("dataset_rgb_test", "f1", ">=", -0.01, "Dataset RGB F1"),
        ("svanstrom", "by_category.DRONE.recall", ">=", -0.01, "Svanström DRONE R"),
        ("antiuav", "f1", ">=", -0.005, "Anti-UAV F1"),
        ("confuser_test", "halluc_rate", "<", 0.0, "Confuser halluc (target)"),
    ]

    # Also check per-category Svanström confuser halluc
    for cat in ["BIRD", "AIRPLANE", "HELICOPTER"]:
        gate_rules.append(
            ("svanstrom", f"by_category.{cat}.halluc_rate", "<=", 0.02,
             f"Svanström {cat} halluc")
        )

    all_pass = True
    gate_results = []

    for surface, metric_path, direction, threshold, label in gate_rules:
        if surface not in baseline_surfaces or surface not in ft4_results:
            gate_results.append({
                "label": label, "status": "SKIP", "reason": "missing data"
            })
            continue

        # Navigate metric path (supports dotted paths like "by_category.DRONE.recall")
        def _get(d, path):
            for key in path.split("."):
                if isinstance(d, dict) and key in d:
                    d = d[key]
                else:
                    return None
            return d

        base_val = _get(baseline_surfaces[surface], metric_path)
        ft4_val = _get(ft4_results[surface], metric_path)

        if base_val is None or ft4_val is None:
            gate_results.append({
                "label": label, "status": "SKIP",
                "reason": f"metric not found (base={base_val}, ft4={ft4_val})"
            })
            continue

        delta = ft4_val - base_val

        if direction == ">=":
            passed = delta >= threshold
        elif direction == "<":
            passed = delta < threshold
        elif direction == "<=":
            passed = delta <= threshold
        else:
            passed = False

        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False

        gate_results.append({
            "label": label,
            "baseline": round(base_val, 4),
            "ft4": round(ft4_val, 4),
            "delta": round(delta, 4),
            "threshold": threshold,
            "direction": direction,
            "status": status,
        })

    # Print results table
    print(f"\n  {'Check':<30s} {'Baseline':>9s} {'ft4':>9s} {'Δ':>8s} {'Gate':>8s} {'Result':>8s}")
    print(f"  {'-'*75}")
    for g in gate_results:
        if g["status"] == "SKIP":
            print(f"  {g['label']:<30s} {'---':>9s} {'---':>9s} {'---':>8s} {'---':>8s} {'SKIP':>8s}")
        else:
            print(f"  {g['label']:<30s} {g['baseline']:>9.4f} {g['ft4']:>9.4f} "
                  f"{g['delta']:>+8.4f} {g['threshold']:>+8.4f} {g['status']:>8s}")

    print(f"\n  {'='*75}")
    if all_pass:
        print(f"  OVERALL: PASS ✓ — ft4 is safe to ship")
    else:
        print(f"  OVERALL: FAIL ✗ — DO NOT update production stack")
        failed = [g["label"] for g in gate_results if "FAIL" in g.get("status", "")]
        print(f"  Failed checks: {', '.join(failed)}")
    print(f"  {'='*75}")

    # Save results
    out = {
        "ft4_weights": str(ft4_path),
        "baseline_snapshot": str(baseline_path),
        "all_pass": all_pass,
        "gate_results": gate_results,
        "ft4_surfaces": ft4_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
