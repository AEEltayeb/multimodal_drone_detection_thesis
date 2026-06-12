"""temporal_ablation.py — 2-of-3 temporal voting on the SAME 5k cached samples.

Offline (CPU). Reuses the overnight_ablation_full caches (per-frame decisions) and recovers
frame order/sequence by re-running the (fast) iterators and aligning by index (verified by
length match — no detector re-run). Then applies a 2-of-3 sliding window per sequence and
reports per-FRAME vs per-WINDOW P/R/F1 for each ablation cell. Temporal applies only to
sequence/video surfaces (antiuav, svanstrom, svanstrom_gray); photo surfaces are skipped.

  py eval/temporal_ablation.py
"""
from __future__ import annotations
import pickle, re
from pathlib import Path
import numpy as np, joblib, sys
REPO = Path(__file__).resolve().parent.parent
for p in ("classifier", "eval"):
    sys.path.insert(0, str(REPO / p))
from generate_retrained_v2_data import FEATURE_COLS
from rebuild_yolo_cache import iter_antiuav_pairs, iter_svanstrom_pairs

CACHE = REPO / "eval/results/_overnight_ablation_full/cache"
SA32 = joblib.load(REPO / "models/routers/scene_aware_v3more_32feat/model.joblib")
ROB6 = joblib.load(REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib")
TARGET = 5000
RGBDS_DIR = Path("G:/drone/dataset/dataset/images/test")
SURF = [("antiuav", iter_antiuav_pairs), ("svanstrom", iter_svanstrom_pairs),
        ("svanstrom_gray", iter_svanstrom_pairs),
        ("rgb_dataset_test", "DIR")]   # sparse-sampled frames grouped by source prefix


def keys_for(itf):
    if itf == "DIR":  # rgb_dataset_test: list dir stems, same stride as cache
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = sorted(p for p in RGBDS_DIR.iterdir() if p.suffix.lower() in exts)
        st = max(1, len(imgs) // TARGET)
        return [p.stem for p in imgs[::st][:TARGET]]
    pairs = list(itf())
    st = max(1, len(pairs) // TARGET)
    return [pr["key"] for pr in pairs[::st][:TARGET]]


def trust(clf, M):
    if isinstance(clf, dict) and "features" in clf:
        idx = [FEATURE_COLS.index(f) for f in clf["features"]]; m = clf["model"]
    else:
        idx = list(range(len(FEATURE_COLS))); m = clf
    return m.predict(M[:, idx])


def cell_alerts(rows):
    A = {k: np.array([r[k] for r in rows]) for k in ("rgb_any", "ir_any", "rs_mlp", "is_mlp", "rs_pch", "is_pch")}
    f_all = np.array([r["f_all"] for r in rows]); f_mlp = np.array([r["f_mlp"] for r in rows])
    out = {"bare": A["rgb_any"] | A["ir_any"], "filter[mlp]": A["rs_mlp"] | A["is_mlp"],
           "filter[patch]": A["rs_pch"] | A["is_pch"]}
    for cn, clf in (("sa32", SA32), ("robust6", ROB6)):
        ta = trust(clf, f_all); tf = trust(clf, f_mlp)
        trgb, tir = np.isin(ta, [1, 3]), np.isin(ta, [2, 3])
        out[f"clf[{cn}]"] = ta != 0
        out[f"clf->filter[{cn},mlp]"] = (trgb & A["rs_mlp"]) | (tir & A["is_mlp"])
        out[f"filter->clf[{cn},mlp]"] = tf != 0
    return out


def prf(alert, pos):
    tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum()); fn = int((~alert & pos).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    for name, itf in SURF:
        pk = CACHE / f"{name}.pkl"
        if not pk.exists():
            print(f"[skip {name}: no cache]"); continue
        rows = pickle.load(open(pk, "rb"))["rows"]
        keys = keys_for(itf)
        if len(keys) != len(rows):
            print(f"[{name}] LEN MISMATCH keys={len(keys)} rows={len(rows)} -> skip (alignment unsafe)"); continue
        seq = [re.sub(r'_f?\d+$', '', k) for k in keys]
        pos = np.array([r["n_gt"] for r in rows]) > 0
        cells = cell_alerts(rows)
        # group consecutive indices by sequence
        groups = {}
        for i, s in enumerate(seq):
            groups.setdefault(s, []).append(i)
        print(f"\n## {name}  (n={len(rows)}, {len(groups)} sequences)")
        print(f"{'cell':<24}{'frame P/R/F1':<26}{'window(2of3) P/R/F1':<26}{'dR':>6}")
        for cell, al in cells.items():
            fp_, fr_, ff = prf(al, pos)
            # window-level: per sequence sliding 3
            wa, wp = [], []
            for s, idxs in groups.items():
                if len(idxs) < 3:
                    continue
                a = al[idxs]; pp = pos[idxs]
                for j in range(len(idxs) - 2):
                    wa.append(a[j:j+3].sum() >= 2); wp.append(pp[j:j+3].sum() >= 2)
            if not wa:
                print(f"{cell:<24}{f'{fp_:.3f}/{fr_:.3f}/{ff:.3f}':<26}{'(no 3-windows)':<26}"); continue
            wa, wp = np.array(wa), np.array(wp)
            wpp, wrr, wff = prf(wa, wp)
            print(f"{cell:<24}{f'{fp_:.3f}/{fr_:.3f}/{ff:.3f}':<26}{f'{wpp:.3f}/{wrr:.3f}/{wff:.3f}':<26}{wrr-fr_:+.3f}")


if __name__ == "__main__":
    main()
