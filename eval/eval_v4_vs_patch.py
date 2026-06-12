#!/usr/bin/env python3
"""
Head-to-head: V4 MLP-on-fused-features vs production patch verifier v2_backup.

Both branches share the same upstream stack:
    FT4 R3 detector  ->  (single YOLO forward, hook captures p3+p5)
Each branch then applies a different post-detection verifier:
    Branch A (patch_v2): MobileNet-V3-Small on 224x224 image crop.
    Branch B (mlp_v4):   MLP on fused p3+p5 features pooled per detection.

The harness scores both branches against the same GT with identical scoring
rules per dataset. Output: per-branch summary JSON + a comparison.md table.

Eval surfaces:
    svanstrom      -- IoP@0.5, the primary discriminating benchmark.
    confuser_test  -- halluc count only (no GT).
    antiuav        -- IoU@0.5, saturated; sanity floor.

Usage:
    python eval/eval_v4_vs_patch.py                       # all 3 surfaces
    python eval/eval_v4_vs_patch.py --datasets svanstrom  # one surface
    python eval/eval_v4_vs_patch.py --quick               # larger strides
    python eval/eval_v4_vs_patch.py --patch-thr 0.9       # match prod sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))
sys.path.insert(0, str(REPO / "eval"))

from patch_verifier import PatchVerifier  # noqa: E402

from metrics import compute_prf, score_detections  # noqa: E402

# Reuse hook + ROI pool + multi-scale extractor from V5 (works for both
# V4 and V5 weight loads — feature shape determined by V5 globals).
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import (  # noqa: E402
    DetectInputHook,
    roi_pool,
    extract_box_metadata,
    _extract_detection_features,
    INPUT_DIM as _V5_INPUT_DIM,
)

# ── Paths ────────────────────────────────────────────────────────────────────
FT4_WEIGHTS = REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"
PATCH_V2_WEIGHTS = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
MLP_V4_WEIGHTS = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "classifiers" / "mlp_v5.pt"

OUT_ROOT = REPO / "eval" / "results" / "_v5_head_to_head"
# OUT_DIR is set per-run inside main() based on --out-suffix so ablation
# variants don't overwrite each other.

# Eval dataset registry: (img_dir, has_drones, scoring_rule, default_stride)
DATASETS = {
    # (path, has_drones, rule, stride, imgsz)
    # Svanstrom: 1280 because 640x480 native makes drones unresolvable at 640.
    # Everything else: 640, the production default.
    "svanstrom":     (Path("G:/drone/svanstrom_paired/RGB/images"), True,  "iop",  9, 1280),
    "confuser_test": (Path("G:/drone/rgb_confusers_merged/images/test"), False, "iou", 1, 640),
    "antiuav":       (Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"), True, "iou", 5, 640),
    # selcom_val: production CCTV eval surface, IoP@0.5. V5 trains on
    # selcom_TRAIN (not val) so this stays clean for fair V5 vs patch_v2.
    "selcom_val":    (Path("G:/drone/_finetune_selcom_mixed_ft2/images/val"), True, "iop", 1, 1280),
    # rgb_dataset_test: general RGB benchmark, IoU@0.5, stride=34 per ledger
    # row 3.x. V5 trains on train+val splits so test is clean. Required for
    # EVIDENCE_LEDGER regression gate #2 (baseline F1=0.9177).
    "rgb_dataset_test": (Path("G:/drone/dataset/dataset/images/test"), True, "iou", 34, 640),
}
CONF_THR = 0.25
IOU_THR = 0.5
IOP_THR = 0.5


# ── MLP V5 inference wrapper ─────────────────────────────────────────────────

class MLPv4Verifier:
    """Loads an MLP-verifier checkpoint (mlp_v4.pt or mlp_v5.pt) and exposes a
    per-detection drone-vs-confuser verdict.

    Architecture reproduced from the checkpoint's saved hyperparameters:
        sequence of (Linear -> [BatchNorm1d] -> ReLU -> Dropout) with a final
        Linear(., 1). BatchNorm is included iff ckpt['use_batchnorm'] is True
        (V5 default; V4 had no BN).
    """

    def __init__(self, weights_path: Path, device: str = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        self.input_dim = int(ckpt["input_dim"])
        self.hidden_dims = list(ckpt["hidden_dims"])
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.cv_f1 = float(ckpt.get("cv_f1", -1.0))
        self.feature_schema = ckpt.get("feature_schema", "unknown")
        self.metadata_order = ckpt.get("metadata_order",
                                       ["conf", "log_area", "aspect", "rel_cx", "rel_cy"])
        use_bn = bool(ckpt.get("use_batchnorm", False))
        dropout = float(ckpt.get("dropout", 0.2))
        self.p3_grid = tuple(ckpt.get("p3_grid", (1, 1)))
        self.p5_grid = tuple(ckpt.get("p5_grid", (1, 1)))

        dims = [self.input_dim, *self.hidden_dims, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers).to(self.device).eval()
        self.net.load_state_dict(ckpt["state_dict"])

        sm, ss = ckpt["scaler_mean"], ckpt["scaler_scale"]
        if isinstance(sm, torch.Tensor):
            self.scaler_mean = sm.to(self.device).float()
            self.scaler_scale = ss.to(self.device).float()
        else:
            self.scaler_mean = torch.from_numpy(
                np.asarray(sm, dtype=np.float32)).to(self.device)
            self.scaler_scale = torch.from_numpy(
                np.asarray(ss, dtype=np.float32)).to(self.device)

        # Sanity check: the V5 _extract_detection_features feeds INPUT_DIM-D
        # vectors. If the checkpoint expects a different dim, fail loud.
        if self.input_dim != _V5_INPUT_DIM:
            raise ValueError(
                f"Checkpoint input_dim={self.input_dim} but the harness's "
                f"V5 feature extractor produces {_V5_INPUT_DIM}-D vectors. "
                f"Train a V5 checkpoint with eval/distill_v5_p3p5_ft4.py.")

    @torch.no_grad()
    def predict_drone_probs(self, feats_np: np.ndarray) -> np.ndarray:
        """feats_np: (n, input_dim) raw features (NOT pre-scaled)."""
        if len(feats_np) == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(feats_np.astype(np.float32)).to(self.device)
        x = (x - self.scaler_mean) / self.scaler_scale
        self.net.eval()
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_jpg(p: Path) -> bool:
    return p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")


def load_gt_boxes(img_path: Path, ih: int, iw: int) -> list:
    """Resolve labels for both YOLO layouts:
        Layout A (Svan, AntiUAV, RGB_video): .../<split>/{images,labels}/file
        Layout B (Selcom, rgb_dataset):      .../{images,labels}/<split>/file
    """
    # Try Layout A first: parent_of_images is split, so labels is sibling
    layout_a = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    layout_b = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    if layout_a.exists():
        lbl_path = layout_a
    elif layout_b.exists():
        lbl_path = layout_b
    else:
        return []
    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and int(parts[0]) == 0:
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2) * iw
            y1 = (yc - bh / 2) * ih
            x2 = (xc + bw / 2) * iw
            y2 = (yc + bh / 2) * ih
            boxes.append((x1, y1, x2, y2))
    return boxes


def eval_one_surface(
    ds_name: str,
    img_dir: Path,
    stride: int,
    has_drones: bool,
    rule: str,
    imgsz: int,
    yolo: YOLO,
    hook: DetectInputHook,
    patch_vf: PatchVerifier,
    mlp_vf,
    patch_thr: float,
    mlp_thresholds: list[float],
    proto_vf=None,
    proto_thresholds: list | None = None,
) -> dict:
    """Score patch v2 + MLP V5 (+ optional prototype) on the same dataset.

    Branch keys:
        patch_v2_thr_{patch_thr}     -- patch verifier veto if P(confuser) >= thr
        mlp_thr_{t}                   -- MLP keep if P(drone) >= t
        proto_thr_{t}                 -- prototype keep if score >= t (only if proto_vf is given)
    """
    images = sorted(p for p in img_dir.iterdir() if is_jpg(p))[::stride]
    print(f"\n  {ds_name}: {len(images)} images (stride={stride}, rule={rule})")

    branches: dict[str, dict] = {
        # No-veto baseline: shows what FT4 R3 alone delivers
        "bare_ft4": {"tp": 0, "fp": 0, "fn": 0,
                     "n_kept": 0, "n_vetoed": 0},
        f"patch_v2_thr_{patch_thr}": {"tp": 0, "fp": 0, "fn": 0,
                                       "n_kept": 0, "n_vetoed": 0},
    }
    for t in mlp_thresholds:
        branches[f"mlp_thr_{t}"] = {"tp": 0, "fp": 0, "fn": 0,
                                       "n_kept": 0, "n_vetoed": 0}
    if proto_vf is not None and proto_thresholds:
        for t in proto_thresholds:
            branches[f"proto_thr_{t}"] = {"tp": 0, "fp": 0, "fn": 0,
                                          "n_kept": 0, "n_vetoed": 0}

    n_dets_total = 0
    t0 = time.time()
    fn_unmatched = 0  # for has_drones=False, FN doesn't apply

    for img_idx, img_path in enumerate(images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        ih, iw = img_bgr.shape[:2]
        gt_boxes = load_gt_boxes(img_path, ih, iw) if has_drones else []

        # Single YOLO forward (hook captures p3 + p5 simultaneously)
        hook.clear()
        results = yolo.predict(img_bgr, imgsz=imgsz, conf=CONF_THR,
                               verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # No detections: all GT are FN
            if has_drones and gt_boxes:
                for k in branches:
                    branches[k]["fn"] += len(gt_boxes)
            continue

        det_boxes = []
        det_confs = []
        for i in range(len(boxes)):
            det_boxes.append(tuple(boxes.xyxy[i].cpu().numpy().tolist()))
            det_confs.append(float(boxes.conf[i]))
        n_dets_total += len(det_boxes)

        # --- Patch v2 verdict (per detection: P(confuser)) ---
        patch_probs = patch_vf.predict_boxes(img_bgr, det_boxes)

        # --- YOLO feature extract (shared by MLP + prototype branches) ---
        feats = np.stack([
            _extract_detection_features(hook, db, (ih, iw), dc)
            for db, dc in zip(det_boxes, det_confs)
        ])
        mlp_drone_probs = mlp_vf.predict_drone_probs(feats)
        if proto_vf is not None:
            proto_scores = proto_vf.predict_drone_probs(feats)
        else:
            proto_scores = None

        # --- Score each branch ---
        for branch_key in branches:
            if branch_key == "bare_ft4":
                kept_mask = np.ones(len(det_boxes), dtype=bool)  # no veto
            elif branch_key.startswith("patch_v2_thr"):
                thr = patch_thr
                kept_mask = patch_probs < thr  # keep if NOT confuser
            elif branch_key.startswith("proto_thr"):
                thr = float(branch_key.rsplit("_", 1)[-1])
                kept_mask = proto_scores >= thr  # keep if drone-prototype-close
            else:
                thr = float(branch_key.rsplit("_", 1)[-1])
                kept_mask = mlp_drone_probs >= thr  # keep if drone-prob high

            kept_dets = [(db, dc)
                          for db, dc, k in zip(det_boxes, det_confs, kept_mask)
                          if k]
            branches[branch_key]["n_kept"] += int(kept_mask.sum())
            branches[branch_key]["n_vetoed"] += int((~kept_mask).sum())

            if has_drones:
                tp, fp, fn = score_detections(
                    kept_dets, gt_boxes, rule=rule,
                    iou_thr=IOU_THR, iop_thr=IOP_THR)
                branches[branch_key]["tp"] += tp
                branches[branch_key]["fp"] += fp
                branches[branch_key]["fn"] += fn
            else:
                # Confuser surface: every kept det is FP
                branches[branch_key]["fp"] += len(kept_dets)

    elapsed = time.time() - t0
    fps = len(images) / max(elapsed, 0.01)
    print(f"    Done: {fps:.2f} fps  ({elapsed:.1f}s total)")

    # Compute per-branch metrics
    out = {
        "dataset": ds_name,
        "n_images": len(images),
        "n_dets_total": n_dets_total,
        "stride": stride,
        "rule": rule,
        "imgsz": imgsz,
        "conf_thr": CONF_THR,
        "elapsed_s": round(elapsed, 1),
        "branches": {},
    }
    for branch_key, t in branches.items():
        if has_drones:
            prf = compute_prf(t["tp"], t["fp"], t["fn"])
            out["branches"][branch_key] = {
                **t,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "halluc_per_image": round(t["fp"] / max(len(images), 1), 4),
            }
        else:
            out["branches"][branch_key] = {
                **t,
                "halluc_per_image": round(t["fp"] / max(len(images), 1), 4),
                "n_fp_total": t["fp"],
            }

    return out


def build_comparison_table(all_results: dict, patch_thr: float,
                            mlp_thresholds: list[float]) -> str:
    """Render a markdown comparison table from all eval results."""
    lines = []
    lines.append("# V4 MLP vs Patch v2 — Head-to-head\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Per-surface metrics\n")

    for ds_name, r in all_results.items():
        lines.append(f"\n### {ds_name}  (n_images={r['n_images']}, "
                     f"rule={r['rule']}, stride={r['stride']})\n")
        has_drones = "precision" in next(iter(r["branches"].values()))
        if has_drones:
            lines.append("| Branch | TP | FP | FN | P | R | F1 | Halluc/img |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for branch_key, m in r["branches"].items():
                lines.append(
                    f"| {branch_key} | {m['tp']} | {m['fp']} | {m['fn']} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                    f"{m['halluc_per_image']:.4f} |"
                )
        else:
            lines.append("| Branch | FP (halluc) | Kept | Vetoed | Halluc/img |")
            lines.append("|---|---|---|---|---|")
            for branch_key, m in r["branches"].items():
                lines.append(
                    f"| {branch_key} | {m['fp']} | {m['n_kept']} | "
                    f"{m['n_vetoed']} | {m['halluc_per_image']:.4f} |"
                )

    # Decision-gate summary across surfaces
    lines.append("\n## Decision gate\n")
    lines.append(f"Patch v2 threshold: {patch_thr}.  MLP V5 thresholds: {mlp_thresholds}.\n")
    lines.append("\nRules:")
    lines.append("- Gate uses F1 (for drone surfaces) and halluc/img (confuser-only).")
    lines.append("- Recall trade-off is reflected in F1; check the per-surface table for raw R.")
    lines.append("- MLP wins F1 or halluc on >=3 of 5 surfaces with <=1pp F1 loss elsewhere -> swap.")
    lines.append("- MLP within +/-1pp on all surfaces -> swap (latency win).")
    lines.append("- Otherwise -> keep patch v2 in production.\n")

    # Auto-tally
    pkey = f"patch_v2_thr_{patch_thr}"
    for mlp_thr in mlp_thresholds:
        mkey = f"mlp_thr_{mlp_thr}"
        wins, losses, ties = [], [], []

        def cmp(metric_name: str, patch_val: float, mlp_val: float,
                higher_better: bool = True):
            delta = mlp_val - patch_val if higher_better else patch_val - mlp_val
            if delta > 0.01:
                wins.append(f"{metric_name} (+{delta:.3f})")
            elif delta < -0.01:
                losses.append(f"{metric_name} ({delta:.3f})")
            else:
                ties.append(f"{metric_name} ({delta:+.3f})")

        for ds_name, r in all_results.items():
            if pkey not in r["branches"] or mkey not in r["branches"]:
                continue
            p, m = r["branches"][pkey], r["branches"][mkey]
            # Headline metrics for the gate:
            #   - drone surfaces: F1 (captures precision-recall trade-off)
            #   - confuser-only / general:  halluc/img
            # Recall is tracked secondarily (printed below the verdict, not used
            # in gate logic) — if F1 is up, recall change is already accounted
            # for; only flag it if F1 is also down.
            if "recall" in p:
                cmp(f"{ds_name}.F1", p["f1"], m["f1"])
            cmp(f"{ds_name}.halluc/img",
                p["halluc_per_image"], m["halluc_per_image"],
                higher_better=False)

        lines.append(f"\n### MLP @ thr={mlp_thr} vs patch v2 @ thr={patch_thr}\n")
        lines.append(f"- **Wins**: {', '.join(wins) if wins else '(none)'}")
        lines.append(f"- **Losses**: {', '.join(losses) if losses else '(none)'}")
        lines.append(f"- **Ties**: {', '.join(ties) if ties else '(none)'}")
        if len(wins) >= 3 and not any(
            float(s.split("(")[-1].rstrip(")")) < -0.01 for s in losses
        ):
            verdict = "**SWAP candidate**"
        elif not losses:
            verdict = "**SWAP (within +/-1pp)**"
        else:
            verdict = "**KEEP patch v2**"
        lines.append(f"- **Verdict**: {verdict}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4 MLP vs patch v2 head-to-head verifier comparison")
    parser.add_argument("--datasets", type=str,
                        default="svanstrom,confuser_test,antiuav,selcom_val,rgb_dataset_test",
                        help="Comma-separated subset of: svanstrom,confuser_test,antiuav,selcom_val,rgb_dataset_test")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Append to output dir name (avoids overwriting between ablation runs)")
    parser.add_argument("--quick", action="store_true",
                        help="2x larger strides for smoke testing")
    parser.add_argument("--patch-thr", type=float, default=0.5,
                        help="Patch verifier confuser-probability threshold")
    parser.add_argument("--mlp-thrs", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated MLP drone-probability thresholds to sweep")
    parser.add_argument("--mlp-weights", type=str, default=str(MLP_V4_WEIGHTS),
                        help="Path to mlp_v5.pt (default: V5 distill output)")
    parser.add_argument("--patch-weights", type=str, default=str(PATCH_V2_WEIGHTS),
                        help="Path to patch verifier .pt (default: confuser_filter4_rgb_v2_backup.pt)")
    parser.add_argument("--prototype-weights", type=str, default="",
                        help="(Optional) Path to prototype_v1.pt — if set, "
                             "runs a third 'proto_thr_{t}' branch alongside "
                             "MLP and patch v2 on the same images.")
    parser.add_argument("--proto-thrs", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated prototype score thresholds")
    args = parser.parse_args()

    ds_subset = [d.strip() for d in args.datasets.split(",") if d.strip()]
    # Per-run output dir so ablation variants don't overwrite each other.
    out_dir = OUT_ROOT if not args.out_suffix else (
        OUT_ROOT.with_name(OUT_ROOT.name + "_" + args.out_suffix.strip("_")))
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_thresholds = [float(t) for t in args.mlp_thrs.split(",")]
    proto_thresholds = [float(t) for t in args.proto_thrs.split(",")] if args.prototype_weights else None

    print("=" * 72)
    print("  Verifier Head-to-Head: patch v2 + MLP" + (" + prototype" if args.prototype_weights else ""))
    print("=" * 72)
    print(f"  Detector:    {FT4_WEIGHTS}")
    print(f"  Patch v2:    {args.patch_weights}  (thr={args.patch_thr})")
    print(f"  MLP V5:      {args.mlp_weights}  (thrs={mlp_thresholds})")
    if args.prototype_weights:
        print(f"  Prototype:   {args.prototype_weights}  (thrs={proto_thresholds})")
    print(f"  Datasets:    {ds_subset}")
    print("=" * 72)

    # Existence checks
    required = [(FT4_WEIGHTS, "FT4 detector"),
                (Path(args.patch_weights), "Patch v2"),
                (Path(args.mlp_weights), "MLP V5")]
    if args.prototype_weights:
        required.append((Path(args.prototype_weights), "Prototype"))
    for path, label in required:
        if not Path(path).exists():
            print(f"\n  FATAL: {label} not found at {path}")
            if label == "MLP V5":
                print(f"  (Run `python eval/distill_v5_p3p5_ft4.py` first)")
            elif label == "Prototype":
                print(f"  (Run `python eval/build_prototype_verifier.py` first)")
            sys.exit(1)

    # Load models once
    print("\n  Loading models...")
    yolo = YOLO(str(FT4_WEIGHTS))
    hook = DetectInputHook()
    handle = hook.register(yolo)
    patch_vf = PatchVerifier(args.patch_weights)
    mlp_vf = MLPv4Verifier(Path(args.mlp_weights))
    print(f"    MLP V5 schema: {mlp_vf.feature_schema}, CV F1={mlp_vf.cv_f1:.4f}")
    proto_vf = None
    if args.prototype_weights:
        from prototype_verifier import PrototypeVerifier
        proto_vf = PrototypeVerifier(Path(args.prototype_weights))
        print(f"    Prototype schema: {proto_vf.feature_schema}")
        print(f"    Prototype tau={proto_vf.tau:.3f}, scale={proto_vf.scale:.3f}")

    # Eval each surface
    all_results: dict = {}
    for ds_name in ds_subset:
        if ds_name not in DATASETS:
            print(f"  SKIP unknown dataset {ds_name}")
            continue
        img_dir, has_drones, rule, default_stride, imgsz = DATASETS[ds_name]
        stride = default_stride * (2 if args.quick else 1)
        if not img_dir.exists():
            print(f"  SKIP {ds_name}: {img_dir} not found")
            continue
        r = eval_one_surface(
            ds_name, img_dir, stride, has_drones, rule, imgsz,
            yolo, hook, patch_vf, mlp_vf,
            patch_thr=args.patch_thr,
            mlp_thresholds=mlp_thresholds,
            proto_vf=proto_vf,
            proto_thresholds=proto_thresholds,
        )
        all_results[ds_name] = r

        # Persist per-surface JSON
        out_path = out_dir /f"{ds_name}_summary.json"
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"    Saved: {out_path}")

    # Write comparison.md
    md = build_comparison_table(all_results, args.patch_thr, mlp_thresholds)
    md_path = out_dir /"comparison.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n  Comparison table: {md_path}")

    handle.remove()
    print("\n── Done ──")


if __name__ == "__main__":
    main()
