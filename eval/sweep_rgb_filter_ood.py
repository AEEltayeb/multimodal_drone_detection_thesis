"""sweep_rgb_filter_ood.py - OLD CNN-patch vs NEW MLP, isolated on FT4, offline.

Reuses the parallel agent's _offline_pipeline cache (FT4 boxes + 517-D feats +
CNN-patch P(confuser) + GT, all on the SAME ft4 detector), so this is a clean
filter-only comparison with NO GPU:
  - confuser SUPPRESSION measured on rgb_confuser (OOD RGB confusers, no drone GT)
  - drone RECALL measured on rgb_dataset_test (OOD), selcom_val (in-domain), svanstrom
Each filter is swept over its own threshold to trace the suppression-vs-recall Pareto.
  MLP survives if P(drone) >= thr ; CNN survives if P(confuser) < thr.

  py eval/sweep_rgb_filter_ood.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from mlp_verifier import MLPVerifier          # noqa: E402
from metrics import iou_iop                    # noqa: E402

CACHE = REPO / "eval" / "results" / "_offline_pipeline" / "cache"
MLP_RGB = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
THRS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
DRONE_SURF = [("rgb_dataset_test", "iou"), ("selcom_val", "iop"), ("svanstrom", "iop")]


def load(name):
    p = CACHE / f"{name}.pkl"
    return pickle.load(open(p, "rb"))["frames"] if p.exists() else None


def add_pdrone(frames, mlp):
    """Attach MLP P(drone) per det (batch)."""
    feats = [np.asarray(f["feats"], np.float32) for f in frames if len(f["boxes"])]
    if feats:
        allf = np.concatenate(feats)
        pd = mlp.predict_drone_probs(allf)
    o = 0
    for f in frames:
        n = len(f["boxes"])
        f["pdrone"] = pd[o:o+n] if n else np.zeros(0, np.float32)
        o += n if n else 0


def suppression(frames, keep_fn):
    before = after = 0
    for f in frames:
        n = len(f["boxes"])
        if not n:
            continue
        before += n
        after += int(keep_fn(f).sum())
    return 1 - after/before if before else 0.0, before, after


def recall(frames, keep_fn, rule):
    tp = fn = 0
    for f in frames:
        gts = [tuple(b) for b in f["gt_boxes"]]
        if not gts:
            continue
        keep = keep_fn(f)
        dets = [tuple(f["boxes"][i]) for i in range(len(f["boxes"])) if keep[i]]
        matched = set()
        for db in dets:
            best, bi = 0., -1
            for gi, g in enumerate(gts):
                iu, ip = iou_iop(db, g); s = iu if rule == "iou" else ip
                if s > best:
                    best, bi = s, gi
            if best >= 0.5 and bi not in matched:
                matched.add(bi)
        tp += len(matched); fn += len(gts) - len(matched)
    return tp/(tp+fn) if tp+fn else 0.0


def main():
    mlp = MLPVerifier(MLP_RGB, device="cpu")
    conf = load("rgb_confuser")
    if conf is None:
        print("no rgb_confuser cache"); return
    add_pdrone(conf, mlp)
    drones = {}
    for name, rule in DRONE_SURF:
        fr = load(name)
        if fr:
            add_pdrone(fr, mlp); drones[name] = (fr, rule)
    print(f"rgb_confuser dets={sum(len(f['boxes']) for f in conf)}  "
          f"drone surfaces={ {k: sum(len(f['boxes']) for f in v[0]) for k,v in drones.items()} }")

    def report(title, keepers):
        print(f"\n=== {title} ===")
        hdr = f"  {'thr':>5} {'confSuppr':>10} " + " ".join(f"{'R['+n[:8]+']':>13}" for n,_ in DRONE_SURF)
        print(hdr)
        for thr in THRS:
            kf = keepers(thr)
            supp, _, _ = suppression(conf, kf)
            recs = []
            for name, rule in DRONE_SURF:
                recs.append(recall(drones[name][0], kf, rule) if name in drones else float("nan"))
            print(f"  {thr:>5.2f} {supp:>9.1%} " + " ".join(f"{r:>13.4f}" for r in recs))

    report("NEW MLP  (survive if P(drone) >= thr)",
           lambda thr: (lambda f: f["pdrone"] >= thr))
    report("OLD CNN patch v2  (survive if P(confuser) < thr)",
           lambda thr: (lambda f: np.asarray(f["patch"]) < thr))


if __name__ == "__main__":
    main()
