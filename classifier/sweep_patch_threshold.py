"""
sweep_patch_threshold.py — Sweep patch verifier veto threshold across
Anti-UAV, Svanstrom (cached per_det.jsonl) and YouTube (live IR YOLO + patch).

For each candidate threshold T in {0.50..0.95}, report:
  - Anti-UAV / Svanstrom: ir_filter F1 (IoP rule) + drone TP retention
  - YouTube IR: per-category any-det suppression, drone passthrough

Picks the best T across the trade between confuser suppression and drone
preservation.

Usage:
  python classifier/sweep_patch_threshold.py
  python classifier/sweep_patch_threshold.py --no-youtube  # skip live YouTube
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS = SCRIPT_DIR / "runs"
REPO = SCRIPT_DIR.parent

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

RGB_CONF = 0.25
IR_CONF  = 0.40

# ── Anti-UAV / Svanstrom (cached) ─────────────────────────────────

def sweep_paired_cached():
    """Sweep on antiuav + svanstrom from per_det.jsonl, IoP rule for svanstrom,
    IoU for antiuav. Reports ir_filter F1 + drone TP retention."""
    out = {}
    for ds, rule, conf in [("antiuav",   "iop", IR_CONF),
                            ("svanstrom", "iop", IR_CONF)]:
        midx = 2 if rule == "iou" else 3
        # Use the OLD-pipeline cache (per_det records are independent of classifier
        # since we're looking at IR-side patch behavior).
        # Actually use NEW32 since that has the latest patch verifier scores cached.
        jsonl = RUNS / "eval_six_configs_v3more_32feat" / ds / "per_det.jsonl"
        if not jsonl.exists():
            print(f"[skip] {ds}: missing {jsonl}")
            continue

        ir_probs = []  # (patch_prob, matched_iop, has_gt)
        rgb_probs = []
        with jsonl.open() as fh:
            for ln in fh:
                r = json.loads(ln)
                ir_n_gt = r["ir_n_gt"]; rgb_n_gt = r["rgb_n_gt"]
                for d in r.get("ir", []):
                    if d[0] >= IR_CONF:
                        ir_probs.append((d[1], d[midx], ir_n_gt > 0))
                for d in r.get("rgb", []):
                    if d[0] >= RGB_CONF:
                        rgb_probs.append((d[1], d[midx], rgb_n_gt > 0))

        out[ds] = {"rule": rule, "ir": [], "rgb": []}
        for T in THRESHOLDS:
            for tag, dets in [("ir", ir_probs), ("rgb", rgb_probs)]:
                tp = fp = fn = 0
                # patch survival: pred=accept iff patch_prob < T
                # label=1 if matched a drone GT (matched_iop)
                for prob, matched, has_gt in dets:
                    accept = (prob < T)
                    if matched == 1:
                        if accept: tp += 1
                        else:      fn += 1
                    else:
                        if accept: fp += 1
                        # if not accept and label==0 -> TN (correct rejection)
                p = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * p * rec / max(1e-9, p + rec)
                out[ds][tag].append({"T": T, "TP": tp, "FP": fp, "FN": fn,
                                       "P": p, "R": rec, "F1": f1})
        print(f"\n  [{ds}] IR-side patch sweep (IoP rule):")
        print(f"    {'T':>5}  {'TP':>7}  {'FP':>6}  {'FN':>6}  {'P':>7}  {'R':>7}  {'F1':>7}")
        for r in out[ds]["ir"]:
            print(f"    {r['T']:>5.2f}  {r['TP']:>7,}  {r['FP']:>6,}  {r['FN']:>6,}  "
                  f"{r['P']:>7.4f}  {r['R']:>7.4f}  {r['F1']:>7.4f}")
    return out


# ── YouTube IR (live) ─────────────────────────────────────────────

DRONE_QUALITY = {
    "yt_zFu7hAi5mIc.mp4": "CLEAN",
    "yt_oA8Bfc_bjFk.mp4": "LABELS",
    "yt_Y0epqCI7muk.mp4": "LABELS",
    "yt_nqk0NsTBlFI.mp4": "LABELS",
}


def sweep_youtube_live():
    """Live IR YOLO + patch verifier on YouTube videos. Dump per-det probs
    once, sweep thresholds in memory."""
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO / "ir_gui"))
    from patch_verifier import PatchVerifier
    from ultralytics import YOLO

    settings = json.loads((REPO / "ir_gui" / "fusion_settings.json").read_text())
    yolo = YOLO(settings["ir_model"])
    patch = PatchVerifier(str(SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"),
                           device="cuda:0")

    from eval_youtube_ir_filter import VIDEO_LABELS, DEMO_OUT  # noqa
    video_dir = DEMO_OUT

    cache_path = RUNS / "eval_youtube_ir" / "patch_sweep_perdet.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # If cache exists, reuse
    if cache_path.exists():
        print(f"  [cache] using existing {cache_path.name}")
    else:
        print(f"  [live] generating per-det dump (one pass)...")
        with cache_path.open("w") as out_fh:
            for vname, cat in VIDEO_LABELS.items():
                vpath = video_dir / vname
                if not vpath.exists():
                    continue
                quality = DRONE_QUALITY.get(vname, "")
                cap = cv2.VideoCapture(str(vpath))
                if not cap.isOpened():
                    continue
                t0 = time.time()
                idx = 0
                n_dets = 0
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    res = yolo.predict(frame, conf=IR_CONF, iou=0.45, imgsz=640,
                                        verbose=False, device=0, max_det=300)[0]
                    if res.boxes is None or len(res.boxes) == 0:
                        # Still record empty-frame for det-rate denom
                        out_fh.write(json.dumps({
                            "video": vname, "category": cat, "quality": quality,
                            "frame": idx, "n_dets": 0, "probs": []
                        }) + "\n")
                        idx += 1; continue
                    xy = res.boxes.xyxy.cpu().numpy()
                    boxes = [(float(xy[i,0]), float(xy[i,1]), float(xy[i,2]), float(xy[i,3]))
                             for i in range(len(xy))]
                    probs = patch.predict_boxes(frame, boxes).tolist()
                    out_fh.write(json.dumps({
                        "video": vname, "category": cat, "quality": quality,
                        "frame": idx, "n_dets": len(boxes), "probs": probs
                    }) + "\n")
                    n_dets += len(boxes)
                    idx += 1
                cap.release()
                dt = time.time() - t0
                print(f"    {vname} ({cat}{'/'+quality if quality else ''}): "
                      f"{idx} frames, {n_dets} dets, {dt:.0f}s")

    # ── sweep thresholds ─────────────────────────────────────────
    # For each (category, quality) group, count: total_frames, ir_only_det_frames,
    # ir_filter_det_frames(T) for each T in THRESHOLDS.
    groups = defaultdict(lambda: {
        "total": 0, "ir_only": 0,
        "ir_filter": {T: 0 for T in THRESHOLDS},
    })
    with cache_path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            cat = r["category"]; q = r["quality"]
            # Group key:
            #   confusers: just category
            #   drones: DRONE_<quality>
            if cat == "DRONE":
                gk = f"DRONE_{q}" if q else "DRONE"
            else:
                gk = cat
            g = groups[gk]
            g["total"] += 1
            n = r["n_dets"]
            probs = r["probs"]
            if n > 0:
                g["ir_only"] += 1
                # For each threshold, count frames where ANY det survives (prob < T)
                for T in THRESHOLDS:
                    if any(p < T for p in probs):
                        g["ir_filter"][T] += 1

    # Aggregate ALL_CONFUSERS across AIRPLANE+BIRD+HELICOPTER
    confuser_keys = {"AIRPLANE", "BIRD", "HELICOPTER"}
    total_c = sum(g["total"]    for k, g in groups.items() if k in confuser_keys)
    iro_c   = sum(g["ir_only"]  for k, g in groups.items() if k in confuser_keys)
    irf_c = {T: sum(g["ir_filter"][T] for k, g in groups.items() if k in confuser_keys)
             for T in THRESHOLDS}
    groups["ALL_CONFUSERS"] = {"total": total_c, "ir_only": iro_c, "ir_filter": irf_c}

    print(f"\n  YouTube IR — per-threshold survival rates")
    print(f"  (lower = better for confusers; higher = better for drones)")
    print()
    rows = []
    for gk in ["ALL_CONFUSERS", "AIRPLANE", "BIRD", "HELICOPTER",
              "DRONE_CLEAN", "DRONE_LABELS"]:
        g = groups.get(gk)
        if not g or g["total"] == 0: continue
        n = g["total"]; ir_only = g["ir_only"]
        ir_only_rate = ir_only / max(1, n)
        row = {"group": gk, "total": n, "ir_only_rate": ir_only_rate,
               "ir_filter_rate": {T: g["ir_filter"][T] / max(1, n) for T in THRESHOLDS}}
        rows.append(row)
        print(f"  {gk}  n={n}  ir_only={ir_only_rate:.2%}")
        for T in THRESHOLDS:
            rate = row["ir_filter_rate"][T]
            supp = (ir_only - g["ir_filter"][T]) / max(1, ir_only) if ir_only > 0 else 0
            print(f"    T={T:.2f}  ir_filter={rate:.2%}  suppression={supp:.2%}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-youtube", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("  ANTI-UAV + SVANSTROM IR-side patch sweep (cached)")
    print("=" * 70)
    paired = sweep_paired_cached()

    yt = None
    if not args.no_youtube:
        print("\n" + "=" * 70)
        print("  YOUTUBE IR patch sweep (live, one pass)")
        print("=" * 70)
        yt = sweep_youtube_live()

    out_path = RUNS / "patch_threshold_sweep.json"
    out_path.write_text(json.dumps({"paired": paired, "youtube": yt}, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
