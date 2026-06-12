"""
thesis_eval/temporal_replay.py — SEGMENT-LEVEL (2-of-3) replay on the CONSECUTIVE video caches.

Reads thesis_eval/cache/{video_drone,video_confuser}.pkl (kind=grayrgb_paired: ft4 on the RGB frame +
v3b on gray(RGB), the production RGB-video / GUI no-thermal regime, is_grayscale=1) and applies the
documented segment semantics (eval/temporal_ablation.py: per clip, sliding 3-frame windows, window
fires iff >=2 of 3 frames fire, window is drone-positive iff >=2 of 3 frames have GT).

Cells: bare | filt_mlp (mlp_v5 @0.25 RGB + aligned_gray @0.25 on the gray channel, per frame) |
filt_patch (RGB-content patch CNN @0.70, per-frame approximation of the old alert-gated veto) |
clf[router] | clf->filt[router] for router in {robust8, robust6, sa32}. clf->filt_patch[sa32] is the
closest replica of the OLD stack (sa32 + patch) for the before/after comparison; note the old
tab:cascade_segment rows used baseline RGB + alert-gating, so the detector and the gate point differ.

This kills fig:mlp_pipeline_placeholder: production robust8+mlp, segment grain, operational footage.

  py -u thesis_eval/temporal_replay.py
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "thesis_eval"))
from pipeline_eval_unified import (load_classifiers, load_verifiers, batch_labels, batch_probs,  # noqa: E402
                                   patch_mask, RGB_THR_MLP, GRAY_THR_MLP)

CACHE_DIR = REPO / "thesis_eval" / "cache"
OUT_DIR = REPO / "thesis_eval" / "results"
PATCH_THR_VIDEO = 0.70           # the real-video production patch threshold (old cascade table)
WIN, NEED = 3, 2                 # sliding window / votes required (matches eval/temporal_ablation.py)


def frame_decisions(meta, frames, clfs, verifs):
    """Per-frame binary fire decision per cell (any surviving detection in a trusted channel)."""
    F8, F32 = meta["F8"], meta["F32"]
    F8m = np.stack([f["f8_all"] for f in frames])
    F32m = np.stack([f["f32_all"] for f in frames])
    labels = {c: batch_labels(clf, F8m, F32m, F8, F32) for c, clf in clfs.items()}
    rp = batch_probs(frames, "rgb", verifs["mlp_v5"]) if "mlp_v5" in verifs else None
    ip = batch_probs(frames, "ir", verifs["aligned_gray"]) if "aligned_gray" in verifs else None
    n = len(frames)
    rgb_any = np.array([len(f["rgb"]["confs"]) > 0 for f in frames])
    ir_any = np.array([len(f["ir"]["confs"]) > 0 for f in frames])
    rgb_mlp = np.array([bool((rp[i] >= RGB_THR_MLP).any()) if rp is not None else rgb_any[i] for i in range(n)])
    ir_mlp = np.array([bool((ip[i] >= GRAY_THR_MLP).any()) if ip is not None else ir_any[i] for i in range(n)])
    rgb_pch = np.array([bool(patch_mask(f["rgb"], PATCH_THR_VIDEO).any()) for f in frames])
    ir_pch = np.array([bool(patch_mask(f["ir"], PATCH_THR_VIDEO).any()) for f in frames])

    cells = {"bare": rgb_any | ir_any,
             "filt_mlp": rgb_mlp | ir_mlp,
             "filt_patch": rgb_pch | ir_pch}
    for cn, lab in labels.items():
        trgb, tir = np.isin(lab, [1, 3]), np.isin(lab, [2, 3])
        cells[f"clf[{cn}]"] = (trgb & rgb_any) | (tir & ir_any)
        cells[f"clf->filt[{cn}]"] = (trgb & rgb_mlp) | (tir & ir_mlp)
    if "sa32" in labels:        # OLD-stack replica: sa32 routing + patch veto (per-frame approx)
        trgb, tir = np.isin(labels["sa32"], [1, 3]), np.isin(labels["sa32"], [2, 3])
        cells["clf->filt_patch[sa32]"] = (trgb & rgb_pch) | (tir & ir_pch)
    return cells


def window_vote(alert, pos, seqs):
    """Sliding 3-frame windows per clip: window fires iff >=2 fire; positive iff >=2 GT frames."""
    wa, wp, order = [], [], {}
    for i, s in enumerate(seqs):
        order.setdefault(s, []).append(i)
    for s, idxs in order.items():
        if len(idxs) < WIN:
            continue
        a, p = alert[idxs], pos[idxs]
        for j in range(len(idxs) - WIN + 1):
            wa.append(a[j:j + WIN].sum() >= NEED)
            wp.append(p[j:j + WIN].sum() >= NEED)
    return np.asarray(wa, bool), np.asarray(wp, bool)


def prf(alert, pos):
    tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum()); fn = int((~alert & pos).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return round(p, 4), round(r, 4), round(2 * p * r / max(p + r, 1e-9), 4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clfs, verifs = load_classifiers(), load_verifiers()
    results = {}
    L = ["# Temporal / segment-level replay — real-video clips, production stack",
         f"{time.strftime('%Y-%m-%d %H:%M')} | grayrgb_paired regime (ft4 + v3b-on-gray, is_grayscale=1) | "
         f"window {NEED}-of-{WIN} per clip | patch_thr={PATCH_THR_VIDEO} (per-frame approx of alert gate)",
         "Old tab:cascade_segment rows used baseline RGB + ALERT-gated patch — detector & gate point "
         "differ; compare directions, not decimals.\n"]

    for name in ("video_drone", "video_confuser"):
        pkl = CACHE_DIR / f"{name}.pkl"
        if not pkl.exists():
            print(f"  [skip {name}: cache not built yet — run pipeline_cache_unified.py --only {name}]")
            continue
        d = pickle.load(open(pkl, "rb")); meta, frames = d["meta"], d["frames"]
        seqs = [f["seq"] for f in frames]
        pos = np.array([len(f["rgb_gt"]) > 0 for f in frames])
        cells = frame_decisions(meta, frames, clfs, verifs)
        n_clips = len(set(seqs))
        res = {"meta": {"n": meta["n"], "clips": n_clips, "has_drones": meta["has_drones"]}}
        L.append(f"\n## {name}  (n={meta['n']} frames, {n_clips} clips, consecutive)\n")
        if meta["has_drones"]:
            L.append("| cell | frame P/R/F1 | window P/R/F1 (2-of-3) | ΔR (win−frame) |\n|---|---|---|---|")
            for cell, al in cells.items():
                fp_, fr_, ff = prf(al, pos)
                wa, wp = window_vote(al, pos, seqs)
                wpp, wrr, wff = prf(wa, wp)
                res[cell] = {"frame": [fp_, fr_, ff], "window": [wpp, wrr, wff]}
                L.append(f"| {cell} | {fp_}/{fr_}/{ff} | {wpp}/{wrr}/{wff} | {wrr - fr_:+.3f} |")
        else:
            L.append("| cell | frame fire | window fire (2-of-3) |\n|---|---|---|")
            for cell, al in cells.items():
                wa, _ = window_vote(al, pos, seqs)
                f_rate = round(float(al.mean()), 4)
                w_rate = round(float(wa.mean()), 4) if len(wa) else None
                res[cell] = {"frame_fire": f_rate, "window_fire": w_rate}
                L.append(f"| {cell} | {f_rate} | {w_rate} |")
        results[name] = res
        print(f"  [{name}] n={meta['n']} clips={n_clips}")

    (OUT_DIR / "temporal_segment_results.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(OUT_DIR / "temporal_results.json", "w"), indent=2, default=float)
    print(f"\nDONE -> {OUT_DIR/'temporal_segment_results.md'}  +  temporal_results.json")


if __name__ == "__main__":
    main()
