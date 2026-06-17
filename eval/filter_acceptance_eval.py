"""filter_acceptance_eval.py — PASS/FAIL acceptance for a re-trained filter vs the
SHIPPED one, ZERO-GPU (cached feats, own-GT). Both filters score the SAME cached
517-D detections, so this isolates the filter change.

RGB (--mode rgb), candidate = mlp_v5_balanced:
  PASS iff  rgb_dataset_test recall UP  AND  svanstrom/selcom_val/antiuav_rgb recall
  not down (> -tol)  AND  rgb_confuser/rgb_bird_confuser FP held (<= shipped*(1+tol)).
IR (--mode ir), candidate = ir_aligned_balanced (thermal head; _gray head for gray surfaces):
  PASS iff  thermal drone recall held (every surface > -tol)  AND  ir_confusers FP DOWN
  (suppression up)  AND  grayscale path not regressed (gray_svan recall, gray_confuser FP).

  py eval/filter_acceptance_eval.py --mode rgb --candidate eval/results/_v5_balanced_remine/classifiers/mlp_v5_balanced.pt
  py eval/filter_acceptance_eval.py --mode ir  --candidate mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections          # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE  # noqa: E402

SHIP = {
    "rgb":  str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt"),
    "ir":   str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"),
    "gray": str(REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt"),
}
# mode -> list of (cache, head)  ; head selects which weight (thermal/gray)
SURF = {
    "rgb": [("rgb_dataset_test", "rgb"), ("svanstrom", "rgb"), ("selcom_val", "rgb"),
            ("antiuav_rgb", "rgb"), ("rgb_confuser", "rgb"), ("rgb_bird_confuser", "rgb")],
    "ir":  [("antiuav_ir", "ir"), ("ir_dset_final", "ir"), ("svanstrom_ir", "ir"),
            ("ir_video", "ir"), ("ir_confusers", "ir"),
            ("gray_svan", "gray"), ("gray_confuser", "gray")],
}
THR = {"rgb": 0.25, "ir": 0.05, "gray": 0.05}


def score(cache_name, weight, thr):
    import pickle
    p = CACHE / f"{cache_name}.pkl"
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb")); meta, frames = d["meta"], d["frames"]
    rule, has = meta["rule"], meta["has_drones"]
    probs = mlp_probs_per_frame(frames, get_mlp(weight)) if weight else None
    tp = fp = fn = 0
    for fi, fr in enumerate(frames):
        n = len(fr["confs"])
        keep = (probs[fi] >= thr) if (probs is not None and n) else np.ones(n, bool)
        kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
        if has:
            t, f_, n_ = score_detections(kept, [tuple(g) for g in fr["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
            tp += t; fp += f_; fn += n_
        else:
            fp += len(kept)
    r = {"has_drones": has, "fp": fp, "n_images": meta["n_images"]}
    if has:
        r.update(compute_prf(tp, fp, fn))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rgb", "ir"], required=True)
    ap.add_argument("--candidate", required=True, help="path to the new filter weight")
    ap.add_argument("--tol", type=float, default=0.01, help="recall no-regression tolerance (default 1pp)")
    a = ap.parse_args()
    cand = Path(a.candidate)
    if not cand.exists():
        print(f"CANDIDATE NOT FOUND: {cand}\n(run the retrain first; then re-run this.)"); return
    cand_gray = cand.parent / "mlp_aligned_gray.pt"   # IR gray head sibling

    print(f"== acceptance [{a.mode}] ==  candidate {cand}\n  shipped: ship={SHIP[a.mode]}\n")
    rows, fails, drone_target = [], [], "rgb_dataset_test"
    for cache_name, head in SURF[a.mode]:
        thr = THR[head]
        ship_w = SHIP[head]
        cand_w = str(cand_gray) if head == "gray" else str(cand)
        if head == "gray" and not cand_gray.exists():
            print(f"  [skip {cache_name}: no candidate gray head {cand_gray.name}]"); continue
        s = score(cache_name, ship_w, thr); c = score(cache_name, cand_w, thr)
        bare = score(cache_name, None, thr)
        if s is None or c is None:
            print(f"  [skip {cache_name}: cache missing]"); continue
        rows.append((cache_name, s, c, bare))

    print(f"{'surface':<18} {'metric':<9} {'bare':>8} {'shipped':>8} {'cand':>8} {'delta':>8}  verdict")
    for name, s, c, bare in rows:
        if s["has_drones"]:
            d = c["recall"] - s["recall"]
            if name == drone_target:
                ok = d > 0; tag = "MUST-UP"
            else:
                ok = d > -a.tol; tag = "no-regress"
            if not ok: fails.append(f"{name} recall {d:+.3f} ({tag})")
            print(f"{name:<18} {'recall':<9} {bare['recall']:>8.3f} {s['recall']:>8.3f} {c['recall']:>8.3f} {d:>+8.3f}  {'PASS' if ok else 'FAIL'} [{tag}]")
        else:
            d = c["fp"] - s["fp"]
            if a.mode == "ir":
                ok = c["fp"] < s["fp"]; tag = "suppress-UP"
            else:
                ok = c["fp"] <= s["fp"] * (1 + a.tol) + 1; tag = "held"
            if not ok: fails.append(f"{name} FP {d:+d} ({tag})")
            print(f"{name:<18} {'FP':<9} {bare['fp']:>8d} {s['fp']:>8d} {c['fp']:>8d} {d:>+8d}  {'PASS' if ok else 'FAIL'} [{tag}]")

    print("\n" + ("ACCEPTED -- all criteria met" if not fails else "REJECTED -- " + "; ".join(fails)))


if __name__ == "__main__":
    main()
