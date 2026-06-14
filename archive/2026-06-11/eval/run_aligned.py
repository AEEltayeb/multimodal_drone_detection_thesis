#!/usr/bin/env python3
"""Train + validate the aligned (gray-harvested) thermal verifier. OFFLINE,
GPU-gated (waits for the GPU so it runs after current jobs; no OOM), sequential.

  1. train_aligned  — thermal drones + per-modality-z gray confusers -> mlp_aligned.pt
  2. holdout        — validate on HELD-OUT CBAM (main_class=1) + thermal drone
                      surfaces + sea/road. CBAM was NOT in training, so this tests
                      whether gray-harvested confusers generalize to novel thermal
                      aerial confusers, at ~0 thermal-drone recall cost.
"""
from __future__ import annotations
import os, subprocess, sys, time, datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
V3B = "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
ALIGNED = "models/verifiers/ir_aligned/mlp_aligned.pt"
IR_PATCH = "models/patches/confuser_filter4_ir_v2_backup.pt"
CBAM = "G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images:main_class=1"
IRDSET = "G:/drone/IR_dset_final/test/images:stride=5,max=2000"
IRVID = "G:/drone/IR_video_ir_dataset/test/images:prefixes=IR_DRONE_,stride=2,max=1500"
ANTIUAV = "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images:stride=20,max=2000"
SEA = "G:/drone/roboflow_infrared_sea_ships_dataset.ir.v1i.yolo26/train/images:max=2000"
ROAD = "G:/drone/road_dog_person_truck_Thermal.v11i.yolo26/test/images"

LOG = REPO / "mri" / "results" / f"aligned_run_{dt.datetime.now():%Y%m%d_%H%M%S}"
LOG.mkdir(parents=True, exist_ok=True)


def log(m):
    line = f"[{dt.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    (LOG / "run.log").open("a", encoding="utf-8").write(line + "\n")


def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(thr=1600, stable=3, interval=30, timeout=4 * 3600):
    log(f"waiting for GPU (<{thr} MB, {stable}x)...")
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        u = gpu_used()
        if u is None:
            log("  no nvidia-smi; proceeding"); return
        ok = ok + 1 if u < thr else 0
        if ok >= stable:
            log(f"  GPU free (used={u}); go"); return
        time.sleep(interval)
    log("  timeout; proceeding")


def step(name, args):
    wait_for_gpu()
    log(f"START {name}: {' '.join(str(a) for a in args)}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    with open(LOG / f"{name}.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(args, cwd=str(REPO), env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    log(f"DONE {name}: rc={rc} ({(time.time()-t0)/60:.1f} min)")
    return rc


def main():
    log(f"== run_aligned ==  log={LOG}")
    step("1_train_aligned", [PY, "mri/train_aligned.py"])
    step("2_holdout_aligned", [PY, "-m", "mri", "--yolo", V3B,
        "--holdout-eval", ALIGNED, "--patch", IR_PATCH,
        "--pos", CBAM, IRDSET, IRVID, ANTIUAV, "--neg", SEA, ROAD,
        "--conf", "0.40", "--mlp-thr", "0.05",
        "--out", "mri/results/ir_aligned_holdout"])
    log("== done ==  see mri/results/ir_aligned_holdout/holdout.json")


if __name__ == "__main__":
    main()
