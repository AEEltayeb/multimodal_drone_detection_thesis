import pickle, numpy as np
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"eval"))
from mri.stats import per_feature_auroc
from eval_v4_vs_patch import MLPv4Verifier
C = REPO/"eval/results/_offline_pipeline/cache"
mlp = MLPv4Verifier(REPO/"models/verifiers/rgb_v5/mlp_v5.pt", device="cpu")
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    i=max(0.,x2-x1)*max(0.,y2-y1);ua=(a[2]-a[0])*(a[3]-a[1]);ub=(b[2]-b[0])*(b[3]-b[1]);return i/(ua+ub-i) if ua+ub-i>0 else 0.
# vetoed + kept real drones from rgb_dataset
d=pickle.load(open(C/"rgb_dataset_test.pkl","rb")); V,K=[],[]
for fr in d["frames"]:
    if len(fr["feats"])==0 or len(fr["gt_boxes"])==0: continue
    p=mlp.predict_drone_probs(fr["feats"])
    for i,b in enumerate(fr["boxes"]):
        if max((iou(b,g) for g in fr["gt_boxes"]),default=0)>=0.5:
            (K if p[i]>=0.25 else V).append(fr["feats"][i])
# confuser dets
cf=pickle.load(open(C/"rgb_confuser.pkl","rb")); Cf=[f for fr in cf["frames"] for f in fr["feats"]] if cf["frames"] else []
V=np.array(V);K=np.array(K);Cf=np.array(Cf)
print(f"vetoed={len(V)} kept={len(K)} confusers={len(Cf)}")
# standardize on kept+confuser pool
alld=np.vstack([K,Cf]); mu,sd=alld.mean(0),alld.std(0)+1e-6
def z(x): return (x-mu)/sd
Kz,Vz,Cz=z(K),z(V),z(Cf)
kc,cc,vc=Kz.mean(0),Cz.mean(0),Vz.mean(0)
import numpy.linalg as la
print(f"centroid dist  kept->confuser = {la.norm(kc-cc):.2f}")
print(f"centroid dist  vetoed->confuser = {la.norm(vc-cc):.2f}")
print(f"centroid dist  vetoed->kept(drone) = {la.norm(vc-kc):.2f}")
# separability from confusers (mean top-20 AUROC)
def sep(A):
    y=np.r_[np.ones(len(A)),np.zeros(len(Cf))]; X=np.vstack([A,Cf])
    a=per_feature_auroc(X,y); return float(np.sort(a)[::-1][:20].mean())
print(f"kept-vs-confuser  mean top20 AUROC = {sep(K):.3f}")
print(f"vetoed-vs-confuser mean top20 AUROC = {sep(V):.3f}  (lower => vetoed look more like confusers)")
