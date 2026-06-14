"""
thesis_eval/scan_routers.py — ZERO-GPU scan of every trust-router contender on a unified cache.

Loads a unified-cache .pkl (written by pipeline_cache_unified.py) and, for EVERY router .joblib under
models/routers/ that is replayable on the cached features (its feature set is a subset of the cached
f8 or f32 vectors), applies the router and scores the classifier-only trust-aware cell clf[router]
(no verifier; per-modality, NO union — same scoring as pipeline_eval_unified Part B). Ranks by RECALL
(the requested objective), with precision/F1 alongside and the bare/no-classifier ceiling on top.

Routers needing features absent from the cache (lean10/13/17/19, control40, all56, ...) are SKIPPED
with a printed reason — scoring them would need a GPU feature-extraction pass. Decision rule per
router reproduces the shipped harness: robust8* uses a tau on P(trust_rgb); everything else argmax.

  py -u thesis_eval/scan_routers.py --cache thesis_eval/cache/dut_antiuav_960.pkl
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np, joblib

REPO = Path(__file__).resolve().parent.parent
import sys
for sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / sub))

from metrics import score_trust_aware, compute_prf            # noqa: E402
from pipeline_eval_unified import batch_labels, dets2, gts, _sum_ta  # noqa: E402

ROUTERS_DIR = REPO / "models" / "routers"


def load_router(path):
    raw = joblib.load(path)
    if isinstance(raw, dict):
        return raw.get("model", raw), raw.get("features"), raw.get("tau"), raw.get("feat_key")
    return raw, None, None, None


def feat_space(model, features, feat_key, F8, F32):
    """Return (space, feats, named) or (None, ...) if not replayable on this cache."""
    if features:
        if set(features) <= set(F8):  return "f8", list(features), True
        if set(features) <= set(F32): return "f32", list(features), True
        return None, None, None
    if feat_key in ("f8", "f32"):
        return feat_key, None, False
    n = getattr(model, "n_features_in_", None)
    if n == len(F8):  return "f8", None, False
    if n == len(F32): return "f32", None, False
    return None, None, None


def score_router(labels, frames, rule):
    tp = fp = fn = 0
    for i, fr in enumerate(frames):
        s = score_trust_aware(int(labels[i]), dets2(fr["rgb"]), dets2(fr["ir"]),
                              gts(fr["rgb_gt"]), gts(fr["ir_gt"]),
                              1920, 1080, 1920, 1080, is_paired=True, rule=rule)
        t, f, n = _sum_ta(s); tp += t; fp += f; fn += n
    return compute_prf(tp, fp, fn)


def tau_for(stem, default_tau, dict_tau):
    if "tau0.10" in stem: return 0.10
    if "robust8" in stem: return default_tau
    return dict_tau            # None -> argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tau-robust8", type=float, default=0.20)
    args = ap.parse_args()

    d = pickle.load(open(args.cache, "rb")); meta, frames = d["meta"], d["frames"]
    F8, F32, rule = meta["F8"], meta["F32"], meta["rule"]
    F8mat = np.stack([fr["f8_all"] for fr in frames])
    F32mat = np.stack([fr["f32_all"] for fr in frames])
    print(f"cache={Path(args.cache).name}  n={meta['n']}  kind={meta['kind']}  rule={rule}  "
          f"is_gray={meta['is_grayscale']}  |F8|={len(F8)} |F32|={len(F32)}")

    rows = [("BARE (no classifier — label=3 always)",
             score_router(np.full(len(frames), 3, int), frames, rule), "ceiling")]

    for jp in sorted(ROUTERS_DIR.rglob("*.joblib")):
        rel = str(jp.relative_to(ROUTERS_DIR))
        try:
            model, features, dtau, fkey = load_router(jp)
            space, feats, named = feat_space(model, features, fkey, F8, F32)
            if space is None:
                need = features or getattr(model, "n_features_in_", "?")
                print(f"  [skip] {rel:<46} needs {need} (not a subset of cached f8/f32)")
                continue
            tau = tau_for(jp.stem, args.tau_robust8, dtau)
            clf = {"model": model, "features": feats, "feat_key": space, "tau": tau}
            labels = batch_labels(clf, F8mat, F32mat, F8, F32)
            prf = score_router(labels, frames, rule)
            uniq = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
            nf = len(feats) if feats else getattr(model, "n_features_in_", "?")
            note = f"{space}{'*named' if named else '*pos'} nf={nf} {'tau='+str(tau) if tau is not None else 'argmax'} lbl={uniq}"
            rows.append((rel, prf, note))
        except Exception as e:
            print(f"  [err]  {rel:<46} {type(e).__name__}: {e}")

    rows.sort(key=lambda r: (r[1]["recall"], r[1]["f1"]), reverse=True)
    print(f"\n{'router':<46} {'P':>6} {'R':>7} {'F1':>6}  note")
    print("-" * 110)
    for name, prf, note in rows:
        print(f"{name:<46} {prf['precision']:>6} {prf['recall']:>7} {prf['f1']:>6}  {note}")
    print("\nNOTE: recall ceiling = BARE (label=3). Any router can only match or reduce it; the "
          "recall-max router is the one that rejects/down-routes least. '*pos' = features applied "
          "positionally (assumes FCOLS32/F8 order); '*named' = mapped by feature name (order-safe).")


if __name__ == "__main__":
    main()
