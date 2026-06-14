#!/usr/bin/env python3
"""Re-train the aligned verifier (now dual-saves thermal + grayscale deploy
checkpoints) and evaluate it on BOTH modes. OFFLINE, GPU-gated, sequential.

  1. train_aligned     -> mlp_aligned.pt (thermal) + mlp_aligned_gray.pt (gray)
  2. holdout GRAY       -> can it replace the grayscale MLP filter?
                          (gray ckpt, --grayscale-input, held-out RGB->gray)
  3. holdout THERMAL    -> ir_dset_final (full-ish) + held-out CBAM + thermal
                          drone surfaces + sea/road. (thermal ckpt)
"""
from __future__ import annotations
import os, subprocess, sys, time, datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
V3B = "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
AL_T = "models/verifiers/ir_aligned/mlp_aligned.pt"
AL_G = "models/verifiers/ir_aligned/mlp_aligned_gray.pt"
IR_PATCH = "models/patches/confuser_filter4_ir_v2_backup.pt"
LOG = REPO / "mri" / "results" / f"aligned_full_{dt.datetime.now():%Y%m%d_%H%M%S}"
LOG.mkdir(parents=True, exist_ok=True)


def log(m):
    line = f"[{dt.datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    (LOG / "run.log").open("a", encoding="utf-8").write(line + "\n")


def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(thr=1600, stable=3, interval=30, timeout=5 * 3600):
    log(f"waiting for GPU (<{thr} MB, {stable}x)..."); t0, ok = time.time(), 0
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
    wait_for_gpu(); log(f"START {name}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    with open(LOG / f"{name}.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(args, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    log(f"DONE {name}: rc={rc} ({(time.time()-t0)/60:.1f} min)")


def main():
    log(f"== run_aligned_full ==  log={LOG}")
    step("1_train_aligned", [PY, "mri/train_aligned.py"])
    step("2_holdout_GRAY", [PY, "-m", "mri", "--yolo", V3B, "--grayscale-input",
        "--holdout-eval", AL_G,
        "--pos", "G:/drone/dataset/dataset/images/test:max=1500",
        "--neg", "G:/drone/rgb_confusers_merged/images/test:stride=2,max=2500",
        "--conf", "0.25", "--mlp-thr", "0.25",
        "--out", "mri/results/ir_aligned_holdout_GRAY"])
    step("3_holdout_THERMAL", [PY, "-m", "mri", "--yolo", V3B, "--patch", IR_PATCH,
        "--holdout-eval", AL_T,
        "--pos",
        "G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images:main_class=1",
        "G:/drone/IR_dset_final/test/images:stride=2,max=4000",
        "G:/drone/IR_video_ir_dataset/test/images:prefixes=IR_DRONE_,stride=2,max=1500",
        "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images:stride=20,max=2000",
        "--neg", "G:/drone/roboflow_infrared_sea_ships_dataset.ir.v1i.yolo26/train/images:max=2000",
        "G:/drone/road_dog_person_truck_Thermal.v11i.yolo26/test/images",
        "--conf", "0.40", "--mlp-thr", "0.05",
        "--out", "mri/results/ir_aligned_holdout_THERMAL"])
    log("== done == GRAY: mri/results/ir_aligned_holdout_GRAY ; THERMAL: mri/results/ir_aligned_holdout_THERMAL")


if __name__ == "__main__":
    main()
