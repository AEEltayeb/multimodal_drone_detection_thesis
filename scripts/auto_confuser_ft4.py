"""
auto_confuser_ft4.py — Automated confuser fine-tune loop with regression gating.

Iterates through a schedule of (n_hardnegs, epochs, freeze, lr0, min_conf)
configs.  For each:
  1. Rebuild ONLY the confuser portion of the dataset (fast — ~5 seconds)
  2. Train from ft3_1280 (always fresh — never chain from a failed attempt)
  3. Run the multi-surface regression gate against baseline_snapshot.json
  4. If PASS → stop and report.  If FAIL → try next config.

The search schedule is ordered from *most aggressive FP suppression* to *most
conservative*, so the first config that passes is the one that reduces the
most confuser hallucinations while staying within the regression budget.

Results for every attempt are logged to scripts/auto_ft4_log.json.

Usage:
    python scripts/auto_confuser_ft4.py
    python scripts/auto_confuser_ft4.py --dry-run          # preview the schedule
    python scripts/auto_confuser_ft4.py --start-from 3     # resume from config #3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Paths ────────────────────────────────────────────────────────────
FT3_WEIGHTS    = ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"
BASELINE_JSON  = ROOT / "scripts" / "baseline_snapshot.json"
BUILDER_SCRIPT = ROOT / "RGB model" / "dataset preparation" / "build_selcom_confuser_ft4.py"
FINETUNE_SCRIPT = ROOT / "RGB model" / "finetune_selcom.py"
LOG_FILE       = ROOT / "scripts" / "auto_ft4_log.json"
FT4_DATASET    = Path(r"C:/drone_cache/_finetune_selcom_confuser_ft4")
FT4_RUN_DIR    = ROOT / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280"

# ── Search schedule ──────────────────────────────────────────────────
# Each config is a dict of hyperparameters.
# n_extra_positives: extra drone-positive images sampled from the general pool
#                    to keep the confuser ratio balanced when increasing confusers.
SEARCH_SCHEDULE = [
    # ---- Phase 1: Original parameter search (configs 0-7) ----
    # Round 1: original recipe but fewer epochs (peaked at epoch 3 anyway)
    {"n_hardnegs": 600, "min_conf": 0.0, "epochs": 3, "freeze": 12, "lr0": 5e-6,
     "n_extra_positives": 0, "label": "R1_600hn_3ep_f12_lr5e6"},

    # Round 2: fewer hard-negs (300 = ~3%), 3 epochs
    {"n_hardnegs": 300, "min_conf": 0.0, "epochs": 3, "freeze": 12, "lr0": 5e-6,
     "n_extra_positives": 0, "label": "R2_300hn_3ep_f12_lr5e6"},

    # Round 3: 300 hard-negs, MORE frozen backbone (freeze=15) -- WINNER
    {"n_hardnegs": 300, "min_conf": 0.0, "epochs": 3, "freeze": 15, "lr0": 5e-6,
     "n_extra_positives": 0, "label": "R3_300hn_3ep_f15_lr5e6"},

    # Round 4: only the highest-confidence FPs (conf>0.7), 300 images
    {"n_hardnegs": 300, "min_conf": 0.7, "epochs": 3, "freeze": 12, "lr0": 5e-6,
     "n_extra_positives": 0, "label": "R4_300hn_minc07_3ep_f12"},

    # Round 5: very small dose -- 150 hardnegs (~1.7%), high conf only
    {"n_hardnegs": 150, "min_conf": 0.7, "epochs": 3, "freeze": 15, "lr0": 3e-6,
     "n_extra_positives": 0, "label": "R5_150hn_minc07_3ep_f15_lr3e6"},

    # Round 6: ultra-conservative -- 100 hardnegs, 2 epochs, very frozen
    {"n_hardnegs": 100, "min_conf": 0.7, "epochs": 2, "freeze": 18, "lr0": 2e-6,
     "n_extra_positives": 0, "label": "R6_100hn_minc07_2ep_f18_lr2e6"},

    # Round 7: 500 hard-negs but half LR and more freeze
    {"n_hardnegs": 500, "min_conf": 0.5, "epochs": 3, "freeze": 15, "lr0": 3e-6,
     "n_extra_positives": 0, "label": "R7_500hn_minc05_3ep_f15_lr3e6"},

    # Round 8: 200 hardnegs, balanced -- medium of everything
    {"n_hardnegs": 200, "min_conf": 0.5, "epochs": 3, "freeze": 12, "lr0": 4e-6,
     "n_extra_positives": 0, "label": "R8_200hn_minc05_3ep_f12_lr4e6"},

    # ---- Phase 2: Ratio ablation (configs 8-11) ----
    # All use freeze=15 (proven safe in R3) and same lr/epochs.
    # Tests whether scaling positives alongside confusers preserves quality.
    #
    # Base: 8,825 images.  R3 ratio = 300/9125 = 3.28%
    #
    # A1: 600 confusers, no extra pos -> ratio 6.37%  (pure freeze=15 test)
    {"n_hardnegs": 600, "min_conf": 0.0, "epochs": 3, "freeze": 15, "lr0": 5e-6,
     "n_extra_positives": 0, "label": "A1_600hn_f15_noextra"},

    # A2: 600 confusers + 4000 extra pos -> ratio ~4.47%  (midpoint)
    {"n_hardnegs": 600, "min_conf": 0.0, "epochs": 3, "freeze": 15, "lr0": 5e-6,
     "n_extra_positives": 4000, "label": "A2_600hn_f15_4000xp"},

    # A3: 600 confusers + 8800 extra pos -> ratio ~3.28%  (match R3 ratio)
    {"n_hardnegs": 600, "min_conf": 0.0, "epochs": 3, "freeze": 15, "lr0": 5e-6,
     "n_extra_positives": 8800, "label": "A3_600hn_f15_8800xp"},

    # A4: 900 confusers + 17600 extra pos -> ratio ~3.28%  (3x confusers, ratio-matched)
    {"n_hardnegs": 900, "min_conf": 0.0, "epochs": 3, "freeze": 15, "lr0": 5e-6,
     "n_extra_positives": 17600, "label": "A4_900hn_f15_17600xp"},
]

# ── Regression gate thresholds ───────────────────────────────────────
GATE_RULES = [
    # (surface, metric_path, direction, threshold, label)
    ("selcom_val",       "f1",                             ">=", -0.01,  "Selcom val F1"),
    ("dataset_rgb_test", "f1",                             ">=", -0.01,  "Dataset RGB F1"),
    ("svanstrom",        "by_category.DRONE.recall",       ">=", -0.01,  "Svanström DRONE R"),
    ("antiuav",          "f1",                             ">=", -0.005, "Anti-UAV F1"),
    ("confuser_test",    "halluc_rate",                    "<",   0.0,   "Confuser halluc down"),
    ("svanstrom",        "by_category.BIRD.halluc_rate",   "<=",  0.02,  "Svanström BIRD halluc"),
    ("svanstrom",        "by_category.AIRPLANE.halluc_rate","<=", 0.02,  "Svanström AIRPLANE halluc"),
    ("svanstrom",        "by_category.HELICOPTER.halluc_rate","<=",0.02, "Svanström HELI halluc"),
]

# Which surfaces to evaluate
DEFAULT_SURFACES = ["selcom_val", "dataset_rgb_test", "confuser_test",
                    "svanstrom", "antiuav"]
# Fast surfaces for quick-reject (run these first; if they fail, skip the rest)
FAST_REJECT_SURFACES = ["selcom_val", "dataset_rgb_test", "confuser_test"]


def _get_nested(d: dict, path: str):
    """Navigate a dotted path like 'by_category.DRONE.recall'."""
    for key in path.split("."):
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d


def load_baseline() -> dict:
    if not BASELINE_JSON.exists():
        print(f"[fatal] Baseline snapshot not found: {BASELINE_JSON}")
        print("  Run baseline_snapshot_ft3.py first.")
        sys.exit(1)
    return json.loads(BASELINE_JSON.read_text())


def run_gate(ft4_surfaces: dict, baseline_surfaces: dict) -> tuple[bool, list[dict]]:
    """Evaluate regression gate rules.  Returns (all_pass, results_list)."""
    all_pass = True
    results = []

    for surface, metric_path, direction, threshold, label in GATE_RULES:
        if surface not in baseline_surfaces or surface not in ft4_surfaces:
            results.append({"label": label, "status": "SKIP", "reason": "missing"})
            continue

        base_val = _get_nested(baseline_surfaces[surface], metric_path)
        ft4_val  = _get_nested(ft4_surfaces[surface], metric_path)

        if base_val is None or ft4_val is None:
            results.append({"label": label, "status": "SKIP",
                            "reason": f"metric missing (base={base_val}, ft4={ft4_val})"})
            continue

        delta = ft4_val - base_val

        if   direction == ">=": passed = delta >= threshold
        elif direction == "<":  passed = delta < threshold
        elif direction == "<=": passed = delta <= threshold
        else:                   passed = False

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        results.append({
            "label": label,
            "baseline": round(base_val, 4),
            "ft4": round(ft4_val, 4),
            "delta": round(delta, 4),
            "threshold": threshold,
            "status": status,
        })

    return all_pass, results


def print_gate_table(results: list[dict], all_pass: bool):
    print(f"\n  {'Check':<30s} {'Baseline':>9s} {'ft4':>9s} {'Delta':>8s} {'Gate':>8s} {'Result':>8s}")
    print(f"  {'-'*75}")
    for g in results:
        if g["status"] == "SKIP":
            print(f"  {g['label']:<30s} {'---':>9s} {'---':>9s} {'---':>8s} {'---':>8s} {'SKIP':>8s}")
        else:
            print(f"  {g['label']:<30s} {g['baseline']:>9.4f} {g['ft4']:>9.4f} "
                  f"{g['delta']:>+8.4f} {g['threshold']:>+8.4f} {g['status']:>8s}")
    print(f"\n  {'='*75}")
    if all_pass:
        print(f"  OVERALL: PASS [OK] -- this config is safe to ship")
    else:
        failed = [g["label"] for g in results if g.get("status") == "FAIL"]
        print(f"  OVERALL: FAIL [X] -- failed: {', '.join(failed)}")
    print(f"  {'='*75}")


def ensure_base_dataset() -> bool:
    """Make sure the base (non-confuser) images exist in the ft4 dataset dir.
    If not, run a full build with 0 confusers just to stage the base."""
    train_img_dir = FT4_DATASET / "images" / "train"
    val_img_dir   = FT4_DATASET / "images" / "val"

    # Check if base images exist (any non-confuser image)
    base_train_ok = False
    base_val_ok   = False
    if train_img_dir.exists():
        non_confuser = [p for p in train_img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                        and not p.name.startswith("confuser_")]
        base_train_ok = len(non_confuser) > 8000  # expect ~8825
    if val_img_dir.exists():
        non_confuser = [p for p in val_img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                        and not p.name.startswith("confuserval_")]
        base_val_ok = len(non_confuser) > 500  # expect ~622

    if base_train_ok and base_val_ok:
        print(f"  Base dataset already staged at {FT4_DATASET}")
        return True

    print(f"\n  Base dataset not found or incomplete — running full build...")
    cmd = [
        sys.executable, str(BUILDER_SCRIPT),
        "--n-hardnegs", "0",       # No confusers — just the base
        "--n-confuser-val", "0",
        "--clean",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def build_dataset(cfg: dict) -> bool:
    """Rebuild the variable portion of the ft4 dataset (confusers + extra positives)."""
    n_xp = cfg.get("n_extra_positives", 0)
    print(f"\n{'='*72}")
    print(f"  DATASET BUILD: n_hardnegs={cfg['n_hardnegs']}  min_conf={cfg['min_conf']}  extra_pos={n_xp}")
    print(f"{'='*72}")
    cmd = [
        sys.executable, str(BUILDER_SCRIPT),
        "--n-hardnegs", str(cfg["n_hardnegs"]),
        "--min-conf", str(cfg["min_conf"]),
        "--n-extra-positives", str(n_xp),
        "--confusers-only",   # Only swap variable images, keep base intact
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def train_model(cfg: dict) -> Path | None:
    """Train from ft3 with given config.  Returns weights path or None."""
    label = cfg["label"]

    print(f"\n{'='*72}")
    print(f"  TRAINING: {label}")
    print(f"  epochs={cfg['epochs']}  freeze={cfg['freeze']}  lr0={cfg['lr0']}")
    print(f"{'='*72}")

    # Force garbage collection before training to free RAM
    gc.collect()

    cmd = [
        sys.executable, str(FINETUNE_SCRIPT),
        "--ft", "4",
        "--imgsz", "1280",
        "--epochs", str(cfg["epochs"]),
        "--batch", "8",
        "--freeze", str(cfg["freeze"]),
        "--lr0", str(cfg["lr0"]),
        "--skip-stage",       # we already built the dataset
        "--skip-eval",        # we do our own multi-surface eval
        "--workers", "0",     # single-process data loading to avoid OOM
    ]
    # Set env var to disable YOLO plotting (saves RAM on 4GB GPU)
    env = os.environ.copy()
    env["YOLO_PLOTS"] = "0"

    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"  [!] Training failed (exit {result.returncode})")
        return None

    weights = FT4_RUN_DIR / "weights" / "best.pt"
    if not weights.exists():
        print(f"  [!] Weights not found after training: {weights}")
        return None

    return weights


def eval_surfaces(weights_path: Path, surfaces: list[str], imgsz: int = 1280,
                  fast_reject: bool = True, baseline_surfaces: dict | None = None) -> dict:
    """Evaluate ft4 on selected surfaces.  With fast_reject, runs a subset first
    and bails early if those fail the gate."""
    # Force GC before loading model
    gc.collect()

    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline_snapshot_ft3 import DATASETS, eval_surface

    from ultralytics import YOLO
    print(f"\n  Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    ft4_results = {}

    # Determine evaluation order
    if fast_reject and baseline_surfaces:
        # Run fast-reject surfaces first
        ordered = [s for s in FAST_REJECT_SURFACES if s in surfaces]
        ordered += [s for s in surfaces if s not in ordered]
    else:
        ordered = list(surfaces)

    for ds_name in ordered:
        if ds_name not in DATASETS:
            print(f"  [{ds_name}] UNKNOWN — skipping")
            continue
        result = eval_surface(model, ds_name, DATASETS[ds_name], imgsz)
        ft4_results[ds_name] = result

        # Fast reject: after each fast-reject surface, check if we've already
        # failed a critical gate
        if fast_reject and baseline_surfaces and ds_name in FAST_REJECT_SURFACES:
            _pass, _res = run_gate(ft4_results, baseline_surfaces)
            critical_fails = [r for r in _res
                              if r.get("status") == "FAIL"
                              and r["label"] in ("Selcom val F1", "Dataset RGB F1")]
            if critical_fails:
                print(f"\n  >> FAST REJECT after {ds_name}: "
                      f"{', '.join(r['label'] for r in critical_fails)} already failed.")
                print(f"     Skipping remaining surfaces to save time.")
                return ft4_results

    # Release model memory
    del model
    gc.collect()

    return ft4_results


def main():
    ap = argparse.ArgumentParser(
        description="Automated confuser fine-tune loop with regression gating")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the schedule without running anything")
    ap.add_argument("--start-from", type=int, default=0,
                    help="Skip configs before this index (0-based)")
    ap.add_argument("--run-all", action="store_true",
                    help="Run ALL configs (don't stop on first PASS). "
                         "Use for ablation studies where you want all results.")
    ap.add_argument("--surfaces", nargs="*", default=None,
                    help="Override which surfaces to evaluate")
    ap.add_argument("--no-fast-reject", action="store_true",
                    help="Disable fast-reject (always eval all surfaces)")
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    surfaces = args.surfaces or DEFAULT_SURFACES

    # ── Print schedule ───────────────────────────────────────────────
    print("="*72)
    print("CONFUSER FT4 AUTO-TRAINING LOOP")
    print(f"  Configs to try: {len(SEARCH_SCHEDULE)}")
    print(f"  Starting from:  #{args.start_from}")
    print(f"  Surfaces:       {surfaces}")
    print(f"  Fast reject:    {not args.no_fast_reject}")
    print(f"  Run all:        {args.run_all}")
    print("="*72)

    for i, cfg in enumerate(SEARCH_SCHEDULE):
        marker = " << START" if i == args.start_from else ""
        skip = " [SKIP]" if i < args.start_from else ""
        xp = cfg.get("n_extra_positives", 0)
        xp_str = f"  xp={xp:<5d}" if xp > 0 else "        "
        print(f"  #{i:>2d}: {cfg['label']:<40s}  "
              f"hn={cfg['n_hardnegs']:<4d}  mc={cfg['min_conf']:<4.1f}  "
              f"ep={cfg['epochs']}  fr={cfg['freeze']:<2d}  lr={cfg['lr0']:.0e}"
              f"{xp_str}{skip}{marker}")

    if args.dry_run:
        print("\n[dry-run] Exiting without running.")
        return

    # ── Load baseline ────────────────────────────────────────────────
    baseline = load_baseline()
    baseline_surfaces = baseline["surfaces"]
    print(f"\n  Baseline model: {baseline['model']}")
    print(f"  Baseline timestamp: {baseline['timestamp']}")

    # ── Ensure base dataset exists ───────────────────────────────────
    print(f"\n{'='*72}")
    print("CHECKING BASE DATASET")
    print(f"{'='*72}")
    if not ensure_base_dataset():
        print("[fatal] Could not stage base dataset. Exiting.")
        sys.exit(1)

    # ── Log setup ────────────────────────────────────────────────────
    # Load existing log if resuming
    log = {"started": datetime.now().isoformat(), "attempts": [], "winner": None}
    if LOG_FILE.exists() and args.start_from > 0:
        try:
            existing = json.loads(LOG_FILE.read_text())
            log["attempts"] = existing.get("attempts", [])
            print(f"  Loaded {len(log['attempts'])} previous attempts from log")
        except Exception:
            pass

    # ── Main loop ────────────────────────────────────────────────────
    winner = None
    for i, cfg in enumerate(SEARCH_SCHEDULE):
        if i < args.start_from:
            continue

        attempt = {
            "config_index": i,
            "label": cfg["label"],
            "config": {k: v for k, v in cfg.items() if k != "label"},
            "started": datetime.now().isoformat(),
        }

        print(f"\n\n{'#'*72}")
        print(f"# CONFIG #{i}: {cfg['label']}")
        print(f"{'#'*72}")

        t0 = time.time()

        # Step 1: Build dataset (confusers-only — fast)
        if not build_dataset(cfg):
            attempt["status"] = "BUILD_FAILED"
            attempt["elapsed_s"] = round(time.time() - t0, 1)
            log["attempts"].append(attempt)
            _save_log(log)
            print(f"\n  [!] Dataset build failed — skipping to next config")
            continue

        # Step 2: Train
        weights = train_model(cfg)
        if weights is None:
            attempt["status"] = "TRAIN_FAILED"
            attempt["elapsed_s"] = round(time.time() - t0, 1)
            log["attempts"].append(attempt)
            _save_log(log)
            print(f"\n  [!] Training failed — skipping to next config")
            continue

        # Step 3: Evaluate
        ft4_results = eval_surfaces(
            weights, surfaces, args.imgsz,
            fast_reject=not args.no_fast_reject,
            baseline_surfaces=baseline_surfaces,
        )

        # Step 4: Regression gate
        all_pass, gate_results = run_gate(ft4_results, baseline_surfaces)
        print_gate_table(gate_results, all_pass)

        attempt["gate_results"] = gate_results
        attempt["all_pass"] = all_pass
        attempt["ft4_surfaces"] = ft4_results
        attempt["weights"] = str(weights)
        attempt["elapsed_s"] = round(time.time() - t0, 1)
        attempt["status"] = "PASS" if all_pass else "FAIL"
        log["attempts"].append(attempt)
        _save_log(log)

        if all_pass:
            if winner is None:
                winner = attempt  # first passing config
            if not args.run_all:
                break
            else:
                print(f"\n  [run-all] Config PASSED but continuing to run remaining configs...")

        # Archive weights for this config (pass or fail)
        archive_dir = FT4_RUN_DIR / f"weights_{cfg['label']}"
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True)
        archived = archive_dir / "best.pt"
        if not archived.exists():
            shutil.copy2(weights, archived)
        print(f"\n  Weights archived to: {archive_dir}")
        if not all_pass:
            print(f"  Moving on to next config...\n")

    # ── Final report ─────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("AUTO-TRAINING COMPLETE")
    print(f"{'='*72}")
    print(f"  Configs tried: {len([a for a in log['attempts'] if a.get('config_index', -1) >= args.start_from])}")

    if winner:
        log["winner"] = winner["label"]
        print(f"\n  [PASS] WINNER: {winner['label']}")
        print(f"     Weights: {winner['weights']}")
        print(f"     Elapsed: {winner['elapsed_s']:.0f}s")
        print(f"\n  Gate results:")
        print_gate_table(winner["gate_results"], True)
    else:
        print(f"\n  [FAIL] NO CONFIG PASSED the regression gate.")
        print(f"     Consider:")
        print(f"     - Relaxing gate thresholds")
        print(f"     - Adding more configs to the search schedule")
        print(f"     - Investigating why confuser injection hurts drone recall")

        # Print a summary of all failures
        print(f"\n  Failure summary:")
        for a in log["attempts"]:
            status = a.get("status", "?")
            fails = [g["label"] for g in a.get("gate_results", [])
                     if g.get("status") == "FAIL"]
            print(f"    #{a['config_index']} {a['label']:<40s} {status}  "
                  f"{'| '.join(fails) if fails else ''}")

    log["finished"] = datetime.now().isoformat()
    _save_log(log)
    print(f"\n  Full log: {LOG_FILE}")
    print(f"{'='*72}")


def _save_log(log: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
