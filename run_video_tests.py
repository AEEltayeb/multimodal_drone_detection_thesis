"""
Drive eval_full_pipeline_singlepass.py over all 19 drone-detection video test
clips. Run from the repo root:

    python run_video_tests.py
    python run_video_tests.py --redo            # ignore existing summaries
    python run_video_tests.py --only drone      # drone clips only
    python run_video_tests.py --only confuser   # confuser-only clips
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

DRONE_CLIPS = [
    "video_drone_drone_and_bird_sky_and_trees_short",
    "video_drone_drone_attacked_by_bird_mountain_side_view",
    "video_drone_drone_over_mountain_attacked_by_birds",
    "video_drone_drone_seagull_attack",
    "video_drone_drone_takeoff_from_ground_and_not_hand_short",
    "video_drone_drone_takeoff_short",
    "video_drone_drone_takeoff_short_trees_background_dji_air_3s_take_off_sho",
    "video_drone_flock_of_seagulls_attack_drone_beach",
    "video_drone_two_birds_drone",
]
CONFUSER_CLIPS = [
    "video_birds_birds_flying_overhead_various_sizes_short",
    "video_birds_birds_in_slow_motion_flying_various_sizes_compilation",
    "video_birds_distant_birds_flying_in_the_sky_short",
    "video_birds_flock_of_birds_flying_short",
    "video_birds_flock_of_birds_flying_sunset",
    "video_airplanes_airplanes_compilation",
    "video_airplanes_distant_airplane_over_head_flying_away",
    "video_helicopters_helicopter_compilation",
    "video_helicopters_helicopter_overhead_short",
    "video_helicopters_helicopter_overhead_very_small_airplane_in_background",
]


def run_clip(clip: str, redo: bool) -> int:
    cmd = [
        sys.executable, str(REPO / "eval" / "eval_full_pipeline_singlepass.py"),
        "--dataset", clip,
        "--rgb-detectors", "baseline", "retrained_v2", "selcom_1280",
        "--ir-detectors", "ir_grayscale",
        "--classifiers", "sa32",
    ]
    if redo:
        cmd.append("--redo")
    proc = subprocess.run(cmd, cwd=str(REPO))
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["drone", "confuser", "all"], default="all")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    if args.only == "drone":
        clips = DRONE_CLIPS
    elif args.only == "confuser":
        clips = CONFUSER_CLIPS
    else:
        clips = DRONE_CLIPS + CONFUSER_CLIPS

    n_total = len(clips)
    t0 = time.time()
    failures: list[str] = []
    for i, clip in enumerate(clips, start=1):
        print(f"\n=== [{i}/{n_total}] {clip} ===", flush=True)
        rc = run_clip(clip, redo=args.redo)
        if rc != 0:
            failures.append(clip)
            print(f"  FAILED on {clip} (rc={rc}) -- continuing", flush=True)
    dt = time.time() - t0
    print(f"\nAll done in {dt:.0f}s. {len(failures)} failures.")
    for f in failures:
        print(f"  failed: {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
