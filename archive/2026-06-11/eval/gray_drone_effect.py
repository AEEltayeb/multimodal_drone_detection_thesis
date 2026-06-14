"""
gray_drone_effect.py — what does grayscale conversion do to DRONE detection?

We characterized grayscale CONFUSER hallucination (37%/img, verifier cuts 96%).
The missing half: does v3b actually DETECT drones on grayscale-RGB, and how does
recall compare to raw-RGB and (where paired) real thermal? If grayscale drone
recall is poor, the grayscale-verifier track is moot — you can't filter what the
detector never finds.

Bare-detector drone recall per surface x mode {raw_rgb, grayscale, thermal}:
  svan / antiuav are paired -> all three modes; rgb_dataset / rgb_video are RGB
  -> raw vs gray only. OFFLINE (local weights+data). Waits for GPU (no OOM).
"""
from __future__ import annotations
import sys, time, json, subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO))
from metrics import score_detections, compute_prf       # noqa: E402
from mri.scan import _read_gt                            # noqa: E402
from mri.datasets import resolve_labels_dir, is_image    # noqa: E402

V3B = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
CONF = 0.40

# name -> dict(rgb=dir, ir=dir|None, imgsz, rule, prefix)
SURFACES = {
    "svan":        dict(rgb="G:/drone/svanstrom_paired/RGB/images",
                        ir="G:/drone/svanstrom_paired/IR/images",
                        imgsz=1280, rule="iop", prefix="IR_DRONE_", stride=6, cap=900),
    "antiuav":     dict(rgb="G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images",
                        ir="G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images",
                        imgsz=640, rule="iou", prefix="", stride=8, cap=900),
    "rgb_dataset": dict(rgb="G:/drone/dataset/dataset/images/test",
                        ir=None, imgsz=640, rule="iou", prefix="", stride=1, cap=900),
    "rgb_video":   dict(rgb="G:/drone/RGB_video_rgb_dataset/test/images",
                        ir=None, imgsz=640, rule="iou", prefix="V_DRONE_", stride=2, cap=900),
}


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(thr=1600, stable=3, interval=30, timeout=2 * 3600):
    print(f"waiting for GPU (<{thr} MB, {stable}x)...", flush=True)
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        u = gpu_used_mb()
        if u is None:
            print("  no nvidia-smi; proceeding"); return
        ok = ok + 1 if u < thr else 0
        if ok >= stable:
            print(f"  GPU free (used={u}); go", flush=True); return
        time.sleep(interval)
    print("  timeout; proceeding anyway")


def imgs_of(d, prefix, stride):
    p = Path(d)
    if not p.exists():
        return []
    xs = sorted(q for q in p.iterdir() if is_image(q)
                and (not prefix or q.name.startswith(prefix)))
    return xs[::stride]


def eval_mode(yolo, img_dir, label_dir, imgsz, rule, prefix, stride, cap, grayscale):
    tp = fp = fn = n = 0
    for ip in imgs_of(img_dir, prefix, stride):
        if n >= cap:
            break
        im = cv2.imread(str(ip))
        if im is None:
            continue
        if grayscale:
            im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        ih, iw = im.shape[:2]
        gt = _read_gt(label_dir, ip.stem, ih, iw, 0) if label_dir else []
        if not gt:
            continue  # only score frames with a GT drone (recall surface)
        n += 1
        r = yolo.predict(im, imgsz=imgsz, conf=CONF, verbose=False, device=0)[0]
        dets = []
        if r.boxes is not None:
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                dets.append(((float(x1), float(y1), float(x2), float(y2)),
                             float(r.boxes.conf[i])))
        t, f, n_ = score_detections(dets, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
        tp += t; fp += f; fn += n_
    prf = compute_prf(tp, fp, fn)
    return {"n_gt_frames": n, "tp": tp, "fp": fp, "fn": fn, **prf}


def main():
    wait_for_gpu()
    yolo = YOLO(str(V3B))
    out = REPO / "mri" / "results" / "gray_drone_effect"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'surface':12} {'mode':10} {'nGT':>5} {'TP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 64)
    report = {}
    for name, s in SURFACES.items():
        modes = [("raw_rgb", s["rgb"], False), ("grayscale", s["rgb"], True)]
        if s["ir"]:
            modes.append(("thermal", s["ir"], False))
        report[name] = {}
        for mode, d, gray in modes:
            if not Path(d).exists():
                print(f"{name:12} {mode:10}  (missing {d})"); continue
            ld = resolve_labels_dir(Path(d))
            m = eval_mode(yolo, d, ld, s["imgsz"], s["rule"], s["prefix"],
                          s["stride"], s["cap"], gray)
            report[name][mode] = m
            print(f"{name:12} {mode:10} {m['n_gt_frames']:>5} {m['tp']:>5} {m['fn']:>5} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
        # grayscale effect = gray recall - raw recall (same RGB frames)
        if "grayscale" in report[name] and "raw_rgb" in report[name]:
            d = report[name]["grayscale"]["recall"] - report[name]["raw_rgb"]["recall"]
            report[name]["gray_minus_raw_recall"] = round(d, 3)
            print(f"{name:12} {'>> gray-raw recall':22} {d:+.3f}")
    (out / "gray_drone_effect.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out/'gray_drone_effect.json'}")
    print("Reading: grayscale viable for drones iff gray recall is usable AND not far "
          "below raw/thermal. If gray recall collapses, the grayscale-verifier track is moot.")


if __name__ == "__main__":
    main()
