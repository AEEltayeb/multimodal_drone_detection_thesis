#!/usr/bin/env python3
"""
overnight_ir.py — unattended IR eval batch. OFFLINE (local weights+data only;
no downloads). Waits for the GPU to free (so it runs AFTER the current job with
no OOM), then runs steps SEQUENTIALLY (each fully exits before the next, so the
GPU is never shared). Continue-on-error; everything logged.

Launch and leave:
    py eval/overnight_ir.py            # runs in foreground/background; safe to detach

Steps (informed by the 2026-05-31 modality probe: grayscale harvest does NOT
transfer to thermal, but grayscale-mode hallucinates heavily + data is abundant):
  1. grayscale-mode V5 verifier — TRAIN on RGB->gray drones+confusers (fused 517-D)
  2. grayscale verifier — HOLDOUT eval on held-out RGB->gray drone/confuser surfaces
  3. thermal detector — HALLUC on clean held-out thermal ground confusers (sea/road)
  4. thermal drone-diversity — RE-MINE+TRAIN fused MLP (+corrective 30k drones)
  5. drone-diversity verifier — HOLDOUT (new) vs (old) on held-out thermal surfaces
"""
from __future__ import annotations
import os, subprocess, sys, time, json, datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
V3B = "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
MLP_IR = "eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt"
IR_PATCH = "models/patches/confuser_filter4_ir_v2_backup.pt"
SEA = "G:/drone/roboflow_infrared_sea_ships_dataset.ir.v1i.yolo26/train/images:max=2500"
ROAD = "G:/drone/road_dog_person_truck_Thermal.v11i.yolo26/test/images"
RGB_DRONE_TEST = "G:/drone/dataset/dataset/images/test:max=1500"
RGB_CONF_TEST = "G:/drone/rgb_confusers_merged/images/test:stride=2,max=2500"
IRDSET_TEST = "G:/drone/IR_dset_final/test/images:stride=5,max=2000"
IRVID_DRONE_TEST = "G:/drone/IR_video_ir_dataset/test/images:prefixes=IR_DRONE_,stride=2,max=1500"
ANTIUAV_TEST = "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images:stride=20,max=2000"

RES = REPO / "mri" / "results"
LOGDIR = RES / f"overnight_{dt.datetime.now():%Y%m%d_%H%M%S}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "overnight.log"


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(MASTER, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_used_mb():
    """MB in use on GPU 0, or None if nvidia-smi unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(threshold_mb=1600, stable=3, interval=30, timeout=5 * 3600):
    """Block until GPU usage < threshold for `stable` consecutive reads."""
    log(f"waiting for GPU (< {threshold_mb} MB used, {stable}x stable)...")
    t0 = time.time(); ok = 0
    while time.time() - t0 < timeout:
        u = gpu_used_mb()
        if u is None:
            log("  nvidia-smi unavailable; proceeding without GPU gate"); return
        ok = ok + 1 if u < threshold_mb else 0
        if ok == 1 or u >= threshold_mb:
            log(f"  GPU used={u} MB (need <{threshold_mb}, stable {ok}/{stable})")
        if ok >= stable:
            log(f"  GPU free (used={u} MB) — proceeding"); return
        time.sleep(interval)
    log("  WARNING: GPU-free timeout; proceeding anyway")


def run_step(name, args, needs=None):
    if needs and not (REPO / needs).exists():
        log(f"SKIP {name}: missing prerequisite {needs}"); return ("skipped", 0)
    wait_for_gpu()
    log(f"START {name}")
    log("  cmd: " + " ".join(str(a) for a in args))
    step_log = LOGDIR / f"{name}.log"
    env = {**os.environ, "MPLBACKEND": "Agg", "YOLO_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    try:
        with open(step_log, "w", encoding="utf-8") as lf:
            rc = subprocess.run(args, cwd=str(REPO), env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        dur = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        log(f"DONE  {name}: {status}  ({dur/60:.1f} min)  -> {step_log.name}")
        return (status, dur)
    except Exception as e:
        log(f"ERROR {name}: {e}")
        return (f"ERROR({e})", time.time() - t0)


def mri(*a):
    return [PY, "-m", "mri", "--yolo", V3B, *a]


STEPS = [
    ("1_grayscale_train", mri(
        "--config", "mri/configs/ir_grayscale_train.yaml", "--grayscale-input",
        "--train-mlp", "--feature-set", "fused", "--conf", "0.25",
        "--no-examples", "--out", "mri/results/ir_grayscale_v5"), None),
    ("2_grayscale_holdout", mri(
        "--grayscale-input", "--holdout-eval", "mri/results/ir_grayscale_v5/mlp.pt",
        "--pos", RGB_DRONE_TEST, "--neg", RGB_CONF_TEST,
        "--conf", "0.25", "--mlp-thr", "0.25",
        "--out", "mri/results/ir_grayscale_holdout"), "mri/results/ir_grayscale_v5/mlp.pt"),
    ("3_thermal_groundconf", mri(
        "--holdout-eval", MLP_IR, "--patch", IR_PATCH,
        "--neg", SEA, ROAD, "--conf", "0.40", "--mlp-thr", "0.05",
        "--out", "mri/results/ir_thermal_groundconf"), MLP_IR),
    ("4_dronediv_train", mri(
        "--config", "mri/configs/ir_dronediv_train.yaml",
        "--train-mlp", "--feature-set", "fused", "--conf", "0.40",
        "--no-examples", "--out", "mri/results/ir_dronediv"), None),
    ("5a_dronediv_holdout_NEW", mri(
        "--holdout-eval", "mri/results/ir_dronediv/mlp.pt", "--patch", IR_PATCH,
        "--pos", IRDSET_TEST, IRVID_DRONE_TEST, ANTIUAV_TEST, "--neg", SEA, ROAD,
        "--conf", "0.40", "--mlp-thr", "0.05",
        "--out", "mri/results/ir_dronediv_holdout_new"), "mri/results/ir_dronediv/mlp.pt"),
    ("5b_dronediv_holdout_OLD", mri(
        "--holdout-eval", MLP_IR, "--patch", IR_PATCH,
        "--pos", IRDSET_TEST, IRVID_DRONE_TEST, ANTIUAV_TEST, "--neg", SEA, ROAD,
        "--conf", "0.40", "--mlp-thr", "0.05",
        "--out", "mri/results/ir_dronediv_holdout_old"), MLP_IR),
]


def main():
    log(f"== overnight_ir start ==  logdir={LOGDIR}")
    log(f"  python={PY}")
    results = {}
    for name, args, needs in STEPS:
        results[name] = run_step(name, args, needs)
    log("== overnight_ir SUMMARY ==")
    for name, (status, dur) in results.items():
        log(f"  {name:28} {status:14} {dur/60:6.1f} min")
    (LOGDIR / "summary.json").write_text(json.dumps(
        {k: {"status": v[0], "minutes": round(v[1]/60, 1)} for k, v in results.items()},
        indent=2))
    log(f"== done. outputs under mri/results/ ; summary {LOGDIR/'summary.json'} ==")


if __name__ == "__main__":
    main()
