"""test_failopen_verifier.py — would an OOD-abstain ('fail-open') rule recover the
mlp_v5 recall drop without releasing confusers? + generate the figures.

Idea: the MLP should only VETO when the detection is near the confuser distribution;
when it is OOD (far from known confusers) it should ABSTAIN -> KEEP. Test whether the
falsely-vetoed real drones (want to release) and the correctly-vetoed confusers (want to
keep vetoed) are separable by an OOD-from-confuser score (kNN distance to confusers).

Figures -> docs/analysis/images/:
  failopen_pca.png            kept/vetoed drones + confusers in PCA space
  failopen_ood_hist.png       OOD-from-confuser score: vetoed-drones vs vetoed-confusers
  failopen_tradeoff.png       recovered drone recall vs confuser leak as tau sweeps

  py eval/test_failopen_verifier.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from eval_v4_vs_patch import MLPv4Verifier
CACHE = REPO / "eval/results/_offline_pipeline/cache"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
mlp = MLPv4Verifier(REPO / "models/verifiers/rgb_v5/mlp_v5.pt", device="cpu")
THR = 0.25


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.


# collect kept/vetoed real drones (rgb_dataset_test) and vetoed confusers (rgb_confuser)
K, V = [], []
d = pickle.load(open(CACHE / "rgb_dataset_test.pkl", "rb"))
for fr in d["frames"]:
    if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
        continue
    p = mlp.predict_drone_probs(fr["feats"])
    for i, b in enumerate(fr["boxes"]):
        if max((iou(b, g) for g in fr["gt_boxes"]), default=0) >= 0.5:
            (K if p[i] >= THR else V).append(fr["feats"][i])
Cf_all, Cf_vetoed = [], []
cf = pickle.load(open(CACHE / "rgb_confuser.pkl", "rb"))
for fr in cf["frames"]:
    if len(fr["feats"]) == 0:
        continue
    p = mlp.predict_drone_probs(fr["feats"])
    for i in range(len(fr["feats"])):
        Cf_all.append(fr["feats"][i])
        if p[i] < THR:
            Cf_vetoed.append(fr["feats"][i])   # confusers the MLP correctly kills
K, V, Cf_all, Cf_vetoed = map(np.array, (K, V, Cf_all, Cf_vetoed))
print(f"kept drones={len(K)} vetoed drones={len(V)} confusers(all)={len(Cf_all)} vetoed confusers={len(Cf_vetoed)}")

# standardize on confuser distribution; OOD score = mean dist to k=5 nearest confusers
mu, sd = Cf_all.mean(0), Cf_all.std(0) + 1e-6
z = lambda x: (x - mu) / sd
nn = NearestNeighbors(n_neighbors=min(5, len(Cf_all))).fit(z(Cf_all))
ood = lambda x: nn.kneighbors(z(x))[0].mean(1)
ood_V = ood(V)          # vetoed drones — want HIGH (OOD -> release)
ood_Cfv = ood(Cf_vetoed)  # vetoed confusers — want LOW (stay vetoed)
print(f"OOD-from-confuser  vetoed-drones median={np.median(ood_V):.2f}  vetoed-confusers median={np.median(ood_Cfv):.2f}")

# fail-open sweep: keep a vetoed det if ood>tau. recovered drone recall vs confuser leak
taus = np.linspace(0, max(ood_V.max(), ood_Cfv.max()), 60)
rec = [(ood_V > t).mean() for t in taus]          # frac of vetoed drones recovered
leak = [(ood_Cfv > t).mean() for t in taus]        # frac of vetoed confusers released (bad)
# pick tau recovering >=80% drones, min leak
best = min((t for t in taus if (ood_V > t).mean() >= 0.8), default=taus[0])
print(f"@tau recovering 80% vetoed drones: confuser-leak = {(ood_Cfv>best).mean():.1%}")

# ---- figures ----
P = PCA(2).fit(z(np.vstack([K, V, Cf_all])))
plt.figure(figsize=(6.5, 5.5))
for A, c, l in [(Cf_all, "red", "confusers"), (K, "green", "kept drones"), (V, "blue", "vetoed drones")]:
    Z = P.transform(z(A)); plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, c=c, label=l)
plt.legend(); plt.title("mlp_v5: kept vs vetoed drones vs confusers (PCA)")
plt.tight_layout(); plt.savefig(IMG / "failopen_pca.png", dpi=160); plt.close()

plt.figure(figsize=(7, 4))
plt.hist(ood_V, bins=40, alpha=0.6, color="blue", density=True, label=f"vetoed DRONES (n={len(V)})")
plt.hist(ood_Cfv, bins=40, alpha=0.6, color="red", density=True, label=f"vetoed CONFUSERS (n={len(Cf_vetoed)})")
plt.axvline(best, color="k", ls="--", label=f"fail-open τ={best:.1f}")
plt.xlabel("OOD-from-confuser score (kNN dist)"); plt.ylabel("density")
plt.title("Fail-open separability: release drones (high), keep confusers vetoed (low)")
plt.legend(); plt.tight_layout(); plt.savefig(IMG / "failopen_ood_hist.png", dpi=160); plt.close()

plt.figure(figsize=(6.5, 5))
plt.plot([l*100 for l in leak], [r*100 for r in rec], "-o", ms=3)
plt.xlabel("confuser leak % (vetoed confusers released — bad)")
plt.ylabel("vetoed real drones recovered % (good)")
plt.title("Fail-open trade-off (rgb_dataset_test)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(IMG / "failopen_tradeoff.png", dpi=160); plt.close()
print(f"figures -> {IMG}/failopen_pca.png, failopen_ood_hist.png, failopen_tradeoff.png")
