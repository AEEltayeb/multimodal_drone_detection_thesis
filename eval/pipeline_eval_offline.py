"""pipeline_eval_offline.py — Phase B of the offline full-pipeline eval.

ZERO GPU. Loads the per-surface caches from Phase A (pipeline_cache.py) and replays the
verifier matrix per modality, scoring with the exact same score_detections used elsewhere:

  RGB  surfaces : bare | patch_v2@0.5 | mlp_v5@0.25
  IR   surfaces : bare | ir_patch@0.5 | aligned_thermal@0.05
  gray surfaces : bare | patch@0.5     | aligned_gray@0.05

MLP/aligned probs are computed on the cached 517-D features (CPU). Emits a per-surface
markdown table (TP/FP/FN/P/R/F1/halluc) + JSON. Auto-runs after Phase A when chained.

  py -u eval/pipeline_eval_offline.py
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections           # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                   # noqa: E402

CACHE = REPO / "eval" / "results" / "_offline_pipeline" / "cache"
OUT = REPO / "eval" / "results" / "_offline_pipeline"
import os
# env-overridable (defaults = shipped); set THESIS_* to score the offline cache with candidate filters
MLP_V5 = os.environ.get("THESIS_MLP_V5", str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt"))
ALIGNED_THR = os.environ.get("THESIS_ALIGNED", str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"))
ALIGNED_GRAY = os.environ.get("THESIS_ALIGNED_GRAY", str(REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt"))

# modality -> list of (label, kind, arg, thr). kind: bare|patch|mlp
VARIANTS = {
    "rgb":  [("bare", "bare", None, None), ("patch_v2@0.5", "patch", None, 0.5), ("mlp_v5@0.25", "mlp", MLP_V5, 0.25)],
    "ir":   [("bare", "bare", None, None), ("ir_patch@0.5", "patch", None, 0.5), ("aligned_thr@0.05", "mlp", ALIGNED_THR, 0.05)],
    "gray": [("bare", "bare", None, None), ("patch@0.5", "patch", None, 0.5), ("aligned@0.05", "mlp", ALIGNED_THR, 0.05)],
}
_mlp_cache: dict = {}


def get_mlp(path):
    if path not in _mlp_cache:
        _mlp_cache[path] = MLPv4Verifier(Path(path), device="cpu")
    return _mlp_cache[path]


def mlp_probs_per_frame(frames, mlp):
    """Run MLP once over all dets, split back per frame."""
    counts = [len(f["feats"]) for f in frames]
    if sum(counts) == 0:
        return [np.zeros(0, np.float32) for _ in frames]
    allf = np.concatenate([f["feats"] for f in frames if len(f["feats"])], axis=0)
    p = mlp.predict_drone_probs(allf)
    out, i = [], 0
    for c in counts:
        out.append(p[i:i + c]); i += c
    return out


def eval_surface(pkl: Path):
    d = pickle.load(open(pkl, "rb"))
    meta, frames = d["meta"], d["frames"]
    modality, rule, has_drones = meta["modality"], meta["rule"], meta["has_drones"]
    variants = VARIANTS.get(modality, VARIANTS["rgb"])

    # precompute mlp probs per needed checkpoint
    mlp_probs = {}
    for (_, kind, arg, _) in variants:
        if kind == "mlp" and arg not in mlp_probs:
            try:
                mlp_probs[arg] = mlp_probs_per_frame(frames, get_mlp(arg))
            except Exception as e:
                print(f"    [mlp-load-fail {Path(arg).name}: {e}] -> skipping that variant")
                mlp_probs[arg] = None

    rows = {}
    for (label, kind, arg, thr) in variants:
        if kind == "mlp" and mlp_probs.get(arg) is None:
            continue
        tp = fp = fn = nk = nv = 0
        for fi, fr in enumerate(frames):
            n = len(fr["confs"])
            if kind == "bare":
                keep = np.ones(n, bool)
            elif kind == "patch":
                keep = fr["patch"] < thr if n else np.zeros(0, bool)
            else:
                keep = mlp_probs[arg][fi] >= thr if n else np.zeros(0, bool)
            nk += int(keep.sum()); nv += int((~keep).sum())
            kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
            gtb = [tuple(g) for g in fr["gt_boxes"]]
            if has_drones:
                t, f_, n_ = score_detections(kept, gtb, rule=rule, iou_thr=0.5, iop_thr=0.5)
                tp += t; fp += f_; fn += n_
            else:
                fp += len(kept)
        rec = {"tp": tp, "fp": fp, "fn": fn, "n_kept": nk, "n_vetoed": nv,
               "halluc_per_image": round(fp / max(meta["n_images"], 1), 4)}
        if has_drones:
            rec.update(compute_prf(tp, fp, fn))
        rows[label] = rec
    return meta, rows


def main():
    pkls = sorted(CACHE.glob("*.pkl"))
    if not pkls:
        print(f"No caches in {CACHE} — run pipeline_cache.py (Phase A) first."); return
    print(f"Phase B: replaying {len(pkls)} cached surfaces (offline, CPU)\n")
    all_results, lines = {}, ["# Offline Full-Pipeline Eval — Verifier Matrix\n",
                              f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    for pkl in pkls:
        try:
            meta, rows = eval_surface(pkl)
        except Exception as e:
            print(f"  [ERR {pkl.name}: {e}]"); continue
        all_results[meta["name"]] = {"meta": meta, "rows": rows}
        hd = meta["has_drones"]
        print(f"== {meta['name']} ({meta['modality']}, {meta['n_images']} imgs, rule={meta['rule']}) ==")
        lines.append(f"\n## {meta['name']}  ({meta['modality']}, n={meta['n_images']}, rule={meta['rule']})\n")
        if hd:
            lines.append("| variant | TP | FP | FN | P | R | F1 | halluc/img |")
            lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| variant | FP (halluc) | n_kept | halluc/img |")
            lines.append("|---|---|---|---|")
        for lab, m in rows.items():
            if hd:
                print(f"   {lab:<18} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} halluc={m['halluc_per_image']}")
                lines.append(f"| {lab} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['halluc_per_image']} |")
            else:
                print(f"   {lab:<18} FP={m['fp']} halluc/img={m['halluc_per_image']}")
                lines.append(f"| {lab} | {m['fp']} | {m['n_kept']} | {m['halluc_per_image']} |")

    json.dump(all_results, open(OUT / "offline_eval_results.json", "w"), indent=2)
    (OUT / "offline_eval_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPhase B done. -> {OUT/'offline_eval_results.md'} + .json")


if __name__ == "__main__":
    main()
