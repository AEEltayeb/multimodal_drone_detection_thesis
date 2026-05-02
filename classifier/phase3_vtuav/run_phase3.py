"""
run_phase3.py — Orchestrate the VTUAV FPPI experiment.

Steps:
  1. Run inference_vtuav.py (with --resume so rerunning is safe)
  2. Build vtuav_frame_dataset.csv
  3. Evaluate baseline + sup-trained classifiers

Usage:
    python run_phase3.py
    python run_phase3.py --subsample 20     # faster, fewer frames
    python run_phase3.py --flip-ir          # invert IR polarity before inference
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"


def run(cmd, step_name):
    print()
    print("=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"\n  ERROR: '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=10)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip YOLO inference if JSON already exists")
    parser.add_argument("--flip-ir", action="store_true",
                        help="Invert IR polarity (black-hot → white-hot)")
    args = parser.parse_args()

    # File paths: separate files for flipped variant
    suffix = "_flipped" if args.flip_ir else ""
    det_json = str(RUNS_DIR / f"vtuav_detections{suffix}.json")
    csv_file = str(RUNS_DIR / f"vtuav_frame_dataset{suffix}.csv")

    # Step 1: Inference
    if not args.skip_inference:
        inf_cmd = [sys.executable, "run_inference_vtuav.py",
                   "--resume", "--subsample", str(args.subsample)]
        if args.flip_ir:
            inf_cmd.append("--flip-ir")
        run(inf_cmd, f"Run YOLO inference on VTUAV{' [IR FLIPPED]' if args.flip_ir else ''}")

    # Step 2: Build CSV
    run([sys.executable, "build_vtuav_frame_csv.py",
         "--input", det_json, "--output", csv_file],
        "Build VTUAV frame CSV")

    # Step 3: Evaluate
    run([sys.executable, "eval_on_vtuav.py",
         "--csv", csv_file],
        "Evaluate classifiers on VTUAV")


if __name__ == "__main__":
    main()

