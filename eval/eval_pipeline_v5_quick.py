#!/usr/bin/env python3
"""
Quick 500-image pipeline eval with speed comparison.

For each surface, runs FT4 R3 once and evaluates 5 verifier configurations
on the SAME detections (so latency comparisons are like-for-like):

    bare_ft4         -- no verifier (baseline)
    patch_v2_pf      -- patch verifier on every detection (per-frame)
    patch_v2_ag      -- patch verifier only on alert frames (alert-gated)
    v5_mlp_pf        -- V5 MLP on every detection (per-frame)
    v5_mlp_ag        -- V5 MLP only on alert frames (alert-gated)

"Alert frame" semantics: any frame where the bare detector produced >=1
detection with confidence >= ALERT_CONF_THR (default 0.4). Without sa32 in
this quick harness, this is a simple high-conf-detection proxy; in production
the alert gate is the trust classifier's vote, but the verifier-stage compute
implication is identical.

Speed measurement: per-detection inference latency for patch_v2 vs V5 MLP,
measured with torch.cuda.synchronize() barriers, plus per-frame total
pipeline latency for each branch.

Surfaces: svanstrom, confuser_test, antiuav, selcom_val, rgb_dataset_test.
500 imgs each (configurable via --n-images). Per-surface imgsz matches the
production pick.

Usage:
    python eval/eval_pipeline_v5_quick.py
    python eval/eval_pipeline_v5_quick.py --n-images 200 --datasets svanstrom,selcom_val
    python eval/eval_pipeline_v5_quick.py --alert-conf-thr 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))
sys.path.insert(0, str(REPO / "eval"))

from patch_verifier import PatchVerifier  # noqa: E402
from metrics import compute_prf, score_detections  # noqa: E402
from distill_v5_p3p5_ft4 import (  # noqa: E402
    DetectInputHook, _extract_detection_features, INPUT_DIM,
)
from eval_v4_vs_patch import MLPv4Verifier, load_gt_boxes, is_jpg  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────
FT4_WEIGHTS = REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"
PATCH_V2_WEIGHTS = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
MLP_V5_WEIGHTS = REPO / "eval" / "results" / "_v5_selcom_pure_1x8" / "classifiers" / "mlp_v5.pt"

OUT_DIR = REPO / "eval" / "results" / "_v5_pipeline_quick"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Surface registry: (img_dir, has_drones, scoring_rule, imgsz)
DATASETS = {
    "svanstrom":        (Path("G:/drone/svanstrom_paired/RGB/images"), True, "iop", 1280),
    "confuser_test":    (Path("G:/drone/rgb_confusers_merged/images/test"), False, "iou", 640),
    "antiuav":          (Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"), True, "iou", 640),
    "selcom_val":       (Path("G:/drone/_finetune_selcom_mixed_ft2/images/val"), True, "iop", 1280),
    "rgb_dataset_test": (Path("G:/drone/dataset/dataset/images/test"), True, "iou", 640),
}

CONF_THR = 0.25
IOU_THR = 0.5
IOP_THR = 0.5
PATCH_THR = 0.5
MLP_THR = 0.5
DEFAULT_ALERT_CONF_THR = 0.4


# ── Timing helpers ──────────────────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class LatencyAccumulator:
    """Tracks per-call timings and exposes mean/median/p95 + total."""
    def __init__(self):
        self.times_ms = []  # one entry per call

    def add(self, t_seconds: float):
        self.times_ms.append(t_seconds * 1000.0)

    def summary(self) -> dict:
        if not self.times_ms:
            return {"n_calls": 0, "total_ms": 0.0, "mean_ms": 0.0,
                    "median_ms": 0.0, "p95_ms": 0.0}
        a = np.asarray(self.times_ms, dtype=np.float64)
        return {
            "n_calls": int(len(a)),
            "total_ms": float(a.sum()),
            "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
        }


# ── Per-surface eval ────────────────────────────────────────────────────────

def eval_one_surface(
    ds_name: str, img_dir: Path, has_drones: bool, rule: str, imgsz: int,
    n_images: int, yolo: YOLO, hook: DetectInputHook,
    patch_vf: PatchVerifier, mlp_vf: MLPv4Verifier,
    alert_conf_thr: float,
) -> dict:
    """Score 5 verifier branches on the same N images, accumulate timings.
    Uses stride-sampling so the N images span the dataset evenly (avoids
    the alphabetic-first-N bias that hides drone-positive frames behind
    category prefixes like BIRD_/HELICOPTER_)."""
    all_files = sorted(p for p in img_dir.iterdir() if is_jpg(p))
    n_total = len(all_files)
    stride = max(1, n_total // n_images)
    sampled = all_files[::stride][:n_images]
    print(f"\n  {ds_name}: {len(sampled)} images (from {n_total} total at "
          f"stride={stride}, imgsz={imgsz}, rule={rule})")

    branches = ["bare_ft4", "patch_v2_pf", "patch_v2_ag", "v5_mlp_pf", "v5_mlp_ag"]
    counts = {b: {"tp": 0, "fp": 0, "fn": 0,
                  "n_kept": 0, "n_vetoed": 0, "n_alerts": 0}
              for b in branches}

    # Latency trackers
    lat = {
        "yolo_per_frame": LatencyAccumulator(),
        "patch_v2_per_detection": LatencyAccumulator(),
        "v5_mlp_per_detection": LatencyAccumulator(),
        "pipeline_per_frame": {b: LatencyAccumulator() for b in branches},
    }

    n_dets_total = 0
    n_frames_with_alert = 0
    n_frames_processed = 0
    t_surface0 = time.perf_counter()

    for img_path in sampled:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        n_frames_processed += 1
        ih, iw = img_bgr.shape[:2]
        gt_boxes = load_gt_boxes(img_path, ih, iw) if has_drones else []

        # ── YOLO forward ────────────────────────────────────────────────
        hook.clear()
        cuda_sync()
        t0 = time.perf_counter()
        results = yolo.predict(img_bgr, imgsz=imgsz, conf=CONF_THR,
                               verbose=False, device="cuda")
        cuda_sync()
        t_yolo = time.perf_counter() - t0
        lat["yolo_per_frame"].add(t_yolo)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # No detections -> all GT are FN across every branch
            if has_drones and gt_boxes:
                for b in branches:
                    counts[b]["fn"] += len(gt_boxes)
            for b in branches:
                lat["pipeline_per_frame"][b].add(t_yolo)
            continue

        det_boxes = [tuple(boxes.xyxy[i].cpu().numpy().tolist())
                     for i in range(len(boxes))]
        det_confs = [float(boxes.conf[i]) for i in range(len(boxes))]
        n_dets_total += len(det_boxes)

        # Alert-gate decision: does any detection cross ALERT_CONF_THR?
        is_alert = max(det_confs) >= alert_conf_thr
        if is_alert:
            n_frames_with_alert += 1
        for b in branches:
            if "ag" in b and is_alert:
                counts[b]["n_alerts"] += 1
            elif "pf" in b:
                counts[b]["n_alerts"] += 1  # per-frame always "alerts"

        # ── Patch v2 inference (always run to get timing parity) ────────
        cuda_sync()
        t0 = time.perf_counter()
        patch_probs = patch_vf.predict_boxes(img_bgr, det_boxes)
        cuda_sync()
        t_patch = time.perf_counter() - t0
        # Per-detection patch latency
        for _ in det_boxes:
            lat["patch_v2_per_detection"].add(t_patch / len(det_boxes))

        # ── V5 MLP inference (always run for timing parity) ─────────────
        feats = np.stack([
            _extract_detection_features(hook, db, (ih, iw), dc)
            for db, dc in zip(det_boxes, det_confs)
        ])
        cuda_sync()
        t0 = time.perf_counter()
        mlp_drone_probs = mlp_vf.predict_drone_probs(feats)
        cuda_sync()
        t_mlp = time.perf_counter() - t0
        for _ in det_boxes:
            lat["v5_mlp_per_detection"].add(t_mlp / len(det_boxes))

        # ── Verdicts per branch ─────────────────────────────────────────
        # bare_ft4: keep all
        bare_mask = np.ones(len(det_boxes), dtype=bool)
        # patch_v2_pf: keep if not confuser
        patch_pf_mask = patch_probs < PATCH_THR
        # patch_v2_ag: only veto on alert frames; pass through on non-alert
        patch_ag_mask = patch_pf_mask if is_alert else bare_mask.copy()
        # v5_mlp_pf: keep if drone-prob high
        v5_pf_mask = mlp_drone_probs >= MLP_THR
        # v5_mlp_ag: only veto on alert frames
        v5_ag_mask = v5_pf_mask if is_alert else bare_mask.copy()

        masks = {
            "bare_ft4": bare_mask,
            "patch_v2_pf": patch_pf_mask,
            "patch_v2_ag": patch_ag_mask,
            "v5_mlp_pf": v5_pf_mask,
            "v5_mlp_ag": v5_ag_mask,
        }

        # Per-branch pipeline timing (YOLO + verifier cost for that branch)
        branch_extra_cost = {
            "bare_ft4":    0.0,
            "patch_v2_pf": t_patch,
            "patch_v2_ag": t_patch if is_alert else 0.0,
            "v5_mlp_pf":   t_mlp,
            "v5_mlp_ag":   t_mlp if is_alert else 0.0,
        }
        for b in branches:
            lat["pipeline_per_frame"][b].add(t_yolo + branch_extra_cost[b])

        # Score per branch
        for b in branches:
            kept = [(db, dc) for db, dc, k in
                    zip(det_boxes, det_confs, masks[b]) if k]
            counts[b]["n_kept"] += int(masks[b].sum())
            counts[b]["n_vetoed"] += int((~masks[b]).sum())
            if has_drones:
                tp, fp, fn = score_detections(
                    kept, gt_boxes, rule=rule,
                    iou_thr=IOU_THR, iop_thr=IOP_THR)
                counts[b]["tp"] += tp
                counts[b]["fp"] += fp
                counts[b]["fn"] += fn
            else:
                counts[b]["fp"] += len(kept)

    t_surface = time.perf_counter() - t_surface0

    # ── Assemble output ────────────────────────────────────────────────
    out = {
        "dataset": ds_name,
        "n_images": n_frames_processed,
        "n_dets_total": n_dets_total,
        "n_frames_with_alert": n_frames_with_alert,
        "alert_fraction": round(n_frames_with_alert / max(n_frames_processed, 1), 3),
        "imgsz": imgsz,
        "rule": rule,
        "alert_conf_thr": alert_conf_thr,
        "elapsed_s": round(t_surface, 1),
        "branches": {},
        "latency": {
            "yolo_per_frame": lat["yolo_per_frame"].summary(),
            "patch_v2_per_detection": lat["patch_v2_per_detection"].summary(),
            "v5_mlp_per_detection": lat["v5_mlp_per_detection"].summary(),
            "pipeline_per_frame": {b: lat["pipeline_per_frame"][b].summary()
                                    for b in branches},
        },
    }
    for b in branches:
        t = counts[b]
        if has_drones:
            prf = compute_prf(t["tp"], t["fp"], t["fn"])
            out["branches"][b] = {**t, **{k: prf[k] for k in ("precision","recall","f1")},
                                   "halluc_per_image": round(t["fp"] / max(n_frames_processed, 1), 4)}
        else:
            out["branches"][b] = {**t,
                                   "halluc_per_image": round(t["fp"] / max(n_frames_processed, 1), 4)}
    return out


# ── Output formatting ───────────────────────────────────────────────────────

def write_comparison_md(all_results: dict, out_path: Path):
    lines = []
    lines.append("# V5 MLP pipeline quick eval vs patch v2\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Detector: FT4 R3 (Yolo26n_selcom_confuser_ft4_1280)\n")
    lines.append(f"V5 MLP: production pure_1x8\n")

    # ── Latency summary table (cross-surface aggregated) ───────────────
    lines.append("\n## Latency comparison (per detection, ms)\n")
    lines.append("| Surface | YOLO/frame | patch_v2/det | V5 MLP/det | "
                  "patch v2 / V5 ratio |")
    lines.append("|---|---|---|---|---|")
    for ds_name, r in all_results.items():
        y = r["latency"]["yolo_per_frame"]
        p = r["latency"]["patch_v2_per_detection"]
        m = r["latency"]["v5_mlp_per_detection"]
        ratio = (p["mean_ms"] / m["mean_ms"]) if m["mean_ms"] > 0 else 0
        lines.append(
            f"| {ds_name} | {y['mean_ms']:.2f} | {p['mean_ms']:.3f} | "
            f"{m['mean_ms']:.3f} | **{ratio:.1f}×** |")

    # ── Per-surface deploy metrics ─────────────────────────────────────
    lines.append("\n## Per-surface metrics\n")
    for ds_name, r in all_results.items():
        lines.append(f"\n### {ds_name}  (n={r['n_images']}, "
                     f"alert_frames={r['n_frames_with_alert']} = "
                     f"{r['alert_fraction']*100:.1f}%, rule={r['rule']}, "
                     f"imgsz={r['imgsz']})\n")
        has_drones = "precision" in next(iter(r["branches"].values()))
        if has_drones:
            lines.append("| Branch | TP | FP | FN | P | R | F1 | Halluc/img | Pipeline/frame (ms) |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for b, m in r["branches"].items():
                pl = r["latency"]["pipeline_per_frame"][b]["mean_ms"]
                lines.append(
                    f"| {b} | {m['tp']} | {m['fp']} | {m['fn']} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                    f"{m['halluc_per_image']:.4f} | {pl:.2f} |")
        else:
            lines.append("| Branch | FP (halluc) | Kept | Vetoed | Halluc/img | Pipeline/frame (ms) |")
            lines.append("|---|---|---|---|---|---|")
            for b, m in r["branches"].items():
                pl = r["latency"]["pipeline_per_frame"][b]["mean_ms"]
                lines.append(
                    f"| {b} | {m['fp']} | {m['n_kept']} | {m['n_vetoed']} | "
                    f"{m['halluc_per_image']:.4f} | {pl:.2f} |")

    # ── Per-frame vs Alert-gated comparison ─────────────────────────────
    lines.append("\n## V5 MLP per-frame vs alert-gated (does alert-gating buy anything?)\n")
    lines.append("| Surface | V5 PF F1 | V5 AG F1 | Δ F1 | "
                  "V5 PF pipeline ms | V5 AG pipeline ms | Δ ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds_name, r in all_results.items():
        if "precision" not in r["branches"].get("v5_mlp_pf", {}):
            continue
        pf = r["branches"]["v5_mlp_pf"]["f1"]
        ag = r["branches"]["v5_mlp_ag"]["f1"]
        pf_ms = r["latency"]["pipeline_per_frame"]["v5_mlp_pf"]["mean_ms"]
        ag_ms = r["latency"]["pipeline_per_frame"]["v5_mlp_ag"]["mean_ms"]
        lines.append(
            f"| {ds_name} | {pf:.4f} | {ag:.4f} | {ag-pf:+.4f} | "
            f"{pf_ms:.2f} | {ag_ms:.2f} | {ag_ms-pf_ms:+.2f} |")

    lines.append("\n## Patch v2 per-frame vs alert-gated (reference)\n")
    lines.append("| Surface | Patch PF F1 | Patch AG F1 | Δ F1 | "
                  "Patch PF pipeline ms | Patch AG pipeline ms | Δ ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds_name, r in all_results.items():
        if "precision" not in r["branches"].get("patch_v2_pf", {}):
            continue
        pf = r["branches"]["patch_v2_pf"]["f1"]
        ag = r["branches"]["patch_v2_ag"]["f1"]
        pf_ms = r["latency"]["pipeline_per_frame"]["patch_v2_pf"]["mean_ms"]
        ag_ms = r["latency"]["pipeline_per_frame"]["patch_v2_ag"]["mean_ms"]
        lines.append(
            f"| {ds_name} | {pf:.4f} | {ag:.4f} | {ag-pf:+.4f} | "
            f"{pf_ms:.2f} | {ag_ms:.2f} | {ag_ms-pf_ms:+.2f} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global MLP_V5_WEIGHTS, OUT_DIR
    parser = argparse.ArgumentParser(description="Quick pipeline eval with V5 vs patch v2 + speed")
    parser.add_argument("--datasets", type=str,
                        default="svanstrom,confuser_test,antiuav,selcom_val,rgb_dataset_test",
                        help="Comma-separated subset of DATASETS")
    parser.add_argument("--n-images", type=int, default=500,
                        help="Cap per surface (default 500)")
    parser.add_argument("--alert-conf-thr", type=float, default=DEFAULT_ALERT_CONF_THR,
                        help="Detection confidence above which the frame is 'alert' (default 0.4)")
    parser.add_argument("--mlp-weights", type=str, default=str(MLP_V5_WEIGHTS),
                        help="Path to the MLP verifier checkpoint (default: production pure_1x8)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for the output dir (avoids overwriting the default run)")
    args = parser.parse_args()

    MLP_V5_WEIGHTS = Path(args.mlp_weights)
    if args.out_suffix:
        OUT_DIR = REPO / "eval" / "results" / f"_v5_pipeline_quick{args.out_suffix}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_subset = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print("=" * 72)
    print("  V5 MLP pipeline quick eval (with speed + alert-gate ablation)")
    print("=" * 72)
    print(f"  Detector:    {FT4_WEIGHTS}")
    print(f"  Patch v2:    {PATCH_V2_WEIGHTS}")
    print(f"  V5 MLP:      {MLP_V5_WEIGHTS}")
    print(f"  Surfaces:    {ds_subset}")
    print(f"  N images per surface: {args.n_images}")
    print(f"  Alert conf threshold: {args.alert_conf_thr}")
    print("=" * 72)

    for p, label in [(FT4_WEIGHTS, "FT4 detector"),
                      (PATCH_V2_WEIGHTS, "patch v2"),
                      (MLP_V5_WEIGHTS, "V5 MLP")]:
        if not Path(p).exists():
            print(f"FATAL: {label} not found at {p}")
            sys.exit(1)

    print("\n  Loading models...")
    yolo = YOLO(str(FT4_WEIGHTS))
    hook = DetectInputHook()
    handle = hook.register(yolo)
    patch_vf = PatchVerifier(str(PATCH_V2_WEIGHTS))
    mlp_vf = MLPv4Verifier(Path(MLP_V5_WEIGHTS))
    print(f"    V5 schema: {mlp_vf.feature_schema}, CV F1={mlp_vf.cv_f1:.4f}")

    all_results = {}
    for ds_name in ds_subset:
        if ds_name not in DATASETS:
            print(f"  SKIP unknown dataset {ds_name}")
            continue
        img_dir, has_drones, rule, imgsz = DATASETS[ds_name]
        if not img_dir.exists():
            print(f"  SKIP {ds_name}: {img_dir} not found")
            continue
        r = eval_one_surface(ds_name, img_dir, has_drones, rule, imgsz,
                              args.n_images, yolo, hook, patch_vf, mlp_vf,
                              args.alert_conf_thr)
        all_results[ds_name] = r
        out_p = OUT_DIR / f"{ds_name}_summary.json"
        out_p.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"    Saved: {out_p}")

    md_path = OUT_DIR / "comparison.md"
    write_comparison_md(all_results, md_path)
    print(f"\n  Comparison: {md_path}")
    handle.remove()
    print("\n── Done ──")


if __name__ == "__main__":
    main()
