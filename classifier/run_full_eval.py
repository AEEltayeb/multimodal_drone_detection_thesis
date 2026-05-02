"""
run_full_eval.py — One command. Three phases, sequential.

  Phase 1: rebuild YOLO cache for Anti-UAV + Svanström with current RGB+IR
           weights from fusion_settings.json.
  Phase 2: eval_six_configs.py on both datasets.
  Phase 3: eval_youtube_rgb_filter.py on the YouTube RGB confuser corpus.

Total wall: ~3-5 hours on 1050 Ti. Fire and forget.

If any phase fails, the rest are skipped and the failure printed.

Usage:
  python classifier/run_full_eval.py
  python classifier/run_full_eval.py --skip cache       # cache already fresh
  python classifier/run_full_eval.py --skip youtube     # skip Phase 3
  python classifier/run_full_eval.py --rgb-conf 0.25    # override sweep point
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PY = sys.executable

PHASES = ["cache", "six_configs", "youtube"]


def run_step(label: str, cmd: list[str]) -> bool:
    print()
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print(">>", " ".join(str(c) for c in cmd))
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"\n[FAIL] {label} exited with {p.returncode} after {dt/60:.1f} min")
        return False
    print(f"\n[ok] {label} done in {dt/60:.1f} min")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], choices=PHASES,
                    help="phases to skip")
    ap.add_argument("--rgb-conf", type=float, default=0.30)
    ap.add_argument("--ir-conf",  type=float, default=0.40)
    ap.add_argument("--stride",   type=int,   default=3,
                    help="YouTube eval stride (every Nth frame)")
    args = ap.parse_args()

    overall_t0 = time.time()
    results = {}

    # ── Phase 1: rebuild cache ──────────────────────────────────
    if "cache" not in args.skip:
        ok = run_step("Phase 1/3 - Rebuild YOLO cache (Anti-UAV + Svanström)",
                      [PY, str(SCRIPT_DIR / "rebuild_yolo_cache.py"),
                       "--dataset", "both"])
        results["cache"] = ok
        if not ok:
            print("\nAborting: cache rebuild failed; downstream phases skipped.")
            sys.exit(1)
    else:
        print("[skip] Phase 1 (cache)")
        results["cache"] = "skipped"

    # ── Phase 2: eval_six_configs ───────────────────────────────
    if "six_configs" not in args.skip:
        ok = run_step("Phase 2/3 - System-level eval (eval_six_configs)",
                      [PY, str(SCRIPT_DIR / "eval_six_configs.py"),
                       "--dataset", "both",
                       "--rgb-conf", str(args.rgb_conf),
                       "--ir-conf",  str(args.ir_conf)])
        results["six_configs"] = ok
        if not ok:
            print("\n[warn] Phase 2 failed — Phase 3 will still run "
                  "since it's independent.")
    else:
        print("[skip] Phase 2 (six_configs)")
        results["six_configs"] = "skipped"

    # ── Phase 3: YouTube RGB confuser eval ──────────────────────
    if "youtube" not in args.skip:
        ok = run_step("Phase 3/3 - YouTube RGB confuser eval",
                      [PY, str(SCRIPT_DIR / "eval_youtube_rgb_filter.py"),
                       "--stride", str(args.stride)])
        results["youtube"] = ok
    else:
        print("[skip] Phase 3 (youtube)")
        results["youtube"] = "skipped"

    # ── Summary ─────────────────────────────────────────────────
    dt = time.time() - overall_t0
    print()
    print("=" * 72)
    print(f"  ALL DONE in {dt/60:.1f} min")
    print("=" * 72)
    for phase, status in results.items():
        marker = ("OK" if status is True
                  else "FAIL" if status is False
                  else "skip")
        print(f"  [{marker}] {phase}")
    print()
    print("Outputs:")
    print(f"  cache:        classifier/runs/raw_detections.json (+ svanstrom)")
    print(f"  six_configs:  classifier/runs/eval_six_configs/")
    print(f"  youtube:      classifier/runs/eval_youtube_rgb/")


if __name__ == "__main__":
    main()
