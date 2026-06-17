"""build_balanced_v2_surgical.py — FAST v2 (no full re-mine; ~10 min).
From the existing balanced cache (_v5_balanced_remine), surgically:
  (1) swap mixed->PURE selcom  (slice weight-1.8/1.5 rows out; mine ~1765 pure imgs),
  (2) restore the shipped drone:confuser ratio by subsampling the weight-1.0 drones
      (antiuav+rgb_dataset) to the shipped budget (4000+9500=13500),
then retrain. v2 corpus == shipped pure corpus EXCEPT rgb_dataset is balanced.
  py eval/build_balanced_v2_surgical.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
import distill_v5_swap_selcom as S                       # slice_out_selcom, mine_pure_selcom, train_v5_mlp
from ultralytics import YOLO                              # noqa: E402

BAL = REPO / "eval/results/_v5_balanced_remine/training_data.npz"
OUT = REPO / "eval/results/_v5_balanced_v2"; (OUT / "classifiers").mkdir(parents=True, exist_ok=True)
TARGET_W10_DRONES = 13500     # antiuav 4000 + rgb_dataset 9500 = shipped weight-1.0 drone budget


def main():
    z = np.load(BAL)
    X, y, w = z["X"].astype(np.float32), z["y"].astype(np.float32), z["w"].astype(np.float32)
    print(f"balanced cache: {len(X)} ({int((y==1).sum())}d / {int((y==0).sum())}c)")

    # 1) slice out the MIXED selcom (weight 1.8 drone / 1.5 confuser)
    X, y, w = S.slice_out_selcom(X, y, w)

    # 2) restore ratio: subsample weight-1.0 drones (antiuav + balanced rgb_dataset) to the shipped budget
    m = (np.abs(w - 1.0) < 0.01) & (y == 1); idx = np.where(m)[0]
    print(f"  weight-1.0 drones (antiuav+rgb_dataset): {len(idx)} -> target {TARGET_W10_DRONES}")
    if len(idx) > TARGET_W10_DRONES:
        rng = np.random.RandomState(S.SEED)
        drop = rng.choice(idx, len(idx) - TARGET_W10_DRONES, replace=False)
        keep = np.ones(len(X), bool); keep[drop] = False
        X, y, w = X[keep], y[keep], w[keep]
    print(f"  after rebalance: {len(X)} ({int((y==1).sum())}d / {int((y==0).sum())}c)")

    # 3) mine PURE selcom (blocklists 311 selcom_val; imgsz 1280; IoP; weights 1.8/1.5) and add
    yolo = YOLO(S.MODEL_PATHS["ft4_r3"]); hook = S.DetectInputHook(); h = hook.register(yolo)
    try:
        sx_tp, sy_tp, sw_tp, sx_fp, sy_fp, sw_fp = S.mine_pure_selcom(yolo, hook, 1.8, 1.5, imgsz=1280)
    finally:
        h.remove()
    X = np.concatenate([X, sx_tp, sx_fp]); y = np.concatenate([y, sy_tp, sy_fp]); w = np.concatenate([w, sw_tp, sw_fp])
    rng = np.random.RandomState(S.SEED); p = rng.permutation(len(X)); X, y, w = X[p], y[p], w[p]
    print(f"v2 corpus: {len(X)} ({int((y==1).sum())}d / {int((y==0).sum())}c)  "
          f"(expect ~19334d / ~13597c = shipped, with balanced rgb_dataset)")

    np.savez_compressed(OUT / "training_data.npz", X=X, y=y, w=w)
    import json
    (OUT / "training_meta.json").write_text(json.dumps({
        "variant": "balanced_v2_surgical", "from": str(BAL),
        "fix": "pure selcom (swap) + ratio restore (weight-1.0 drone subsample to 13500)",
        "n_total": int(len(X)), "n_drone": int((y == 1).sum()), "n_confuser": int((y == 0).sum()),
        "n_pure_selcom_drones": int(len(sx_tp)), "n_pure_selcom_confusers": int(len(sx_fp)),
    }, indent=2))
    S.train_v5_mlp(X, y, w, OUT / "classifiers" / "mlp_v5_balanced_v2.pt")
    print(f"\nDONE -> {OUT/'classifiers'/'mlp_v5_balanced_v2.pt'}\n"
          f"eval: py eval/filter_acceptance_eval.py --mode rgb --candidate {OUT/'classifiers'/'mlp_v5_balanced_v2.pt'}")


if __name__ == "__main__":
    main()
