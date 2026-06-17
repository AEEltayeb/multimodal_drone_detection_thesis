"""ir_native_vs_aligned_offline.py — does the grayscale-harvested ALIGNED IR
filter beat / match / hurt the older THERMAL-NATIVE filter on the THERMAL path?
And is either filter even doing anything on thermal (redundancy)?  ZERO-GPU.

Trust-aware / own-GT scoring only. A filter only removes detections; we measure
(i) own-GT drone recall on thermal drone surfaces and (ii) confuser detections
kept on the thermal confuser surface, swept over P(drone) thresholds so native
vs aligned are compared on the SAME recall-vs-confuser-fire tradeoff (not one
arbitrary operating point).

Filters (both v3b 517-D distillations, fed RAW cached feats; each applies its own
internal scaler):
  - native  = eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt   (thermal-only)
  - aligned = models/verifiers/ir_aligned/mlp_aligned.pt              (thermal + grayscale-harvested confusers, z-aligned)

Reuses pipeline_eval_offline.get_mlp / mlp_probs_per_frame and the shared
metrics.score_detections so scoring matches the canonical replay exactly.

  py eval/ir_native_vs_aligned_offline.py
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections                 # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE  # noqa: E402

NATIVE = str(REPO / "eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt")
ALIGNED = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
BALANCED = str(REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt")  # v2 (thermal-confuser balanced)
OUT_MD = REPO / "eval/results/_offline_pipeline/ir_native_vs_aligned.md"
OUT_JSON = REPO / "eval/results/_offline_pipeline/ir_native_vs_aligned.json"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
OUT_PNG = IMG / "2026-06-17_ir_native_vs_aligned.png"

THERMAL = ["antiuav_ir", "ir_dset_final", "svanstrom_ir", "ir_video", "ir_confusers"]
THRS = [0.002, 0.01, 0.05, 0.10, 0.15, 0.25, 0.50]
FILTERS = {"native": NATIVE, "aligned": ALIGNED, "balanced": BALANCED}


def load(name):
    p = CACHE / f"{name}.pkl"
    if not p.exists():
        return None
    return pickle.load(open(p, "rb"))


def surface_probs(frames, path):
    return mlp_probs_per_frame(frames, get_mlp(path))


def score_surface(d, probs, thr):
    """Return dict with own-GT recall (drone surfaces) or kept FP (confuser)."""
    meta, frames = d["meta"], d["frames"]
    rule, has_drones = meta["rule"], meta["has_drones"]
    tp = fp = fn = nk = ndet = 0
    for fi, fr in enumerate(frames):
        n = len(fr["confs"])
        ndet += n
        keep = (probs[fi] >= thr) if (probs is not None and n) else (np.ones(n, bool) if n else np.zeros(0, bool))
        nk += int(keep.sum())
        kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
        gtb = [tuple(g) for g in fr["gt_boxes"]]
        if has_drones:
            t, f_, n_ = score_detections(kept, gtb, rule=rule, iou_thr=0.5, iop_thr=0.5)
            tp += t; fp += f_; fn += n_
        else:
            fp += len(kept)
    rec = {"n_images": meta["n_images"], "has_drones": has_drones,
           "n_dets": ndet, "n_kept": nk}
    if has_drones:
        rec.update(compute_prf(tp, fp, fn)); rec.update({"tp": tp, "fp": fp, "fn": fn})
    else:
        rec.update({"fp": fp, "kept_frac": nk / max(ndet, 1), "halluc_per_image": fp / max(meta["n_images"], 1)})
    return rec


def main():
    caches = {n: load(n) for n in THERMAL}
    caches = {n: d for n, d in caches.items() if d is not None}
    print(f"thermal surfaces present: {list(caches)}")

    # precompute probs per filter per surface
    probs = {f: {} for f in FILTERS}
    for n, d in caches.items():
        for f, path in FILTERS.items():
            probs[f][n] = surface_probs(d["frames"], path)

    results = {}  # results[variant][surface] = rec
    # bare
    results["bare"] = {n: score_surface(d, None, 0.0) for n, d in caches.items()}
    for f in FILTERS:
        for thr in THRS:
            results[f"{f}@{thr:.2f}"] = {n: score_surface(d, probs[f][n], thr) for n, d in caches.items()}

    drone_surfaces = [n for n, d in caches.items() if d["meta"]["has_drones"]]
    conf_surfaces = [n for n, d in caches.items() if not d["meta"]["has_drones"]]

    def mean_recall(variant):
        rs = [results[variant][n]["recall"] for n in drone_surfaces]
        return float(np.mean(rs)) if rs else float("nan")

    def conf_kept(variant):
        if not conf_surfaces:
            return float("nan")
        return float(np.mean([results[variant][n]["kept_frac"] for n in conf_surfaces]))

    # ── markdown ─────────────────────────────────────────────────────────
    lines = ["# IR filter: thermal-native vs grayscale-aligned (offline, own-GT)\n",
             f"Thermal surfaces: {', '.join(drone_surfaces)} (drones) + {', '.join(conf_surfaces)} (confuser)\n",
             "\n## Per-surface own-GT drone recall (R) and confuser kept-fraction\n",
             "| variant | " + " | ".join(drone_surfaces) + " | mean R | conf kept |",
             "|" + "---|" * (len(drone_surfaces) + 3)]
    for variant in ["bare"] + [f"{f}@{thr:.2f}" for f in FILTERS for thr in THRS]:
        cells = [f"{results[variant][n]['recall']:.3f}" for n in drone_surfaces]
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {mean_recall(variant):.3f} | {conf_kept(variant):.3f} |")
    lines.append("\n## Confuser surface detail (lower kept = stronger filter)\n")
    for n in conf_surfaces:
        lines.append(f"\n### {n} ({results['bare'][n]['n_dets']} confuser dets over {results['bare'][n]['n_images']} imgs)\n")
        lines.append("| variant | kept | kept_frac | halluc/img |")
        lines.append("|---|---|---|---|")
        for variant in ["bare"] + [f"{f}@{thr:.2f}" for f in FILTERS for thr in THRS]:
            r = results[variant][n]
            lines.append(f"| {variant} | {r['n_kept']} | {r['kept_frac']:.3f} | {r['halluc_per_image']:.3f} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # ── figure: recall vs confuser-kept tradeoff ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for f, color in (("native", "tab:blue"), ("aligned", "tab:red"), ("balanced", "tab:green")):
        xs = [conf_kept(f"{f}@{thr:.2f}") for thr in THRS]
        ys = [mean_recall(f"{f}@{thr:.2f}") for thr in THRS]
        ax.plot(xs, ys, "-o", color=color, label=f)
        for thr, xx, yy in zip(THRS, xs, ys):
            ax.annotate(f"{thr:.2f}", (xx, yy), fontsize=7, color=color, xytext=(3, 3), textcoords="offset points")
    ax.scatter([conf_kept("bare")], [mean_recall("bare")], marker="*", s=200, color="k", label="bare (no filter)", zorder=5)
    ax.set_xlabel("confuser detections kept (fraction)  ← better")
    ax.set_ylabel("mean own-GT drone recall  → better")
    ax.set_title("IR filter tradeoff on the THERMAL path\nnative (thermal-only) vs aligned (thermal+grayscale)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    # ── console summary ──────────────────────────────────────────────────
    print(f"\nbare: mean R={mean_recall('bare'):.3f}  conf kept={conf_kept('bare'):.3f}")
    for f in FILTERS:
        print(f"-- {f} --")
        for thr in THRS:
            v = f"{f}@{thr:.2f}"
            print(f"   thr {thr:.2f}: mean R={mean_recall(v):.3f}  conf kept={conf_kept(v):.3f}")
    print(f"\nsaved {OUT_MD}\nsaved {OUT_JSON}\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
