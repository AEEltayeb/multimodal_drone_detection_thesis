"""bench_speed.py - latency of the two classifiers + two filters.

Classifiers (per frame): robust6 (6 free feats) vs fusion_no_fn (40 feats incl. scene
reads). We separate FEATURE-EXTRACTION time (where they differ) from model-predict time
(both XGBoost, ~equal). Filters (per detection): MLP forward vs CNN-patch forward.

Measured on CPU for a contention-free apples-to-apples relative comparison; the GPU
verifier-stage numbers from the ledger (MLP 1.3-2.1 ms/det, patch 59-112 ms/det) are the
deployment figures. Run:  py eval/bench_speed.py
"""
from __future__ import annotations
import json, pickle, glob, sys, time
from pathlib import Path
import cv2, numpy as np, joblib

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ir_gui")); sys.path.insert(0, str(REPO / "classifier"))
from fusion.features import TARGET_NAMES, compute_global_features, compute_target_features  # noqa
from patch_verifier import PatchVerifier   # noqa
from mlp_verifier import MLPVerifier        # noqa

R6 = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
FNFN = REPO / "classifier/runs/reliability/fusion/fusion_no_fn_model_v1.1.joblib"
MLP = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
PATCH = REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt"
MANIFEST = json.load(open(REPO / "classifier/runs/svanstrom_detections.json"))
N, REP = 200, 20


def img_from_label(p):
    lbl = Path(p); d = lbl.parent.parent / "images"
    for e in (".jpg", ".jpeg", ".png", ".bmp"):
        q = d / f"{lbl.stem}{e}"
        if q.exists():
            return q
    return None


def build40(rgb_dets, ir_dets, rgb_gray, ir_gray):
    feats = {}
    for pre, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]; n = len(confs)
        feats.update({f"{pre}_n_dets": n, f"{pre}_max_conf": max(confs) if n else 0.,
                      f"{pre}_mean_conf": float(np.mean(confs)) if n else 0., f"{pre}_detected": int(n > 0)})
    g_r, g_i = compute_global_features(rgb_gray), compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in g_r.items()}); feats.update({f"ir_{k}": v for k, v in g_i.items()})
    rh, rw = rgb_gray.shape[:2]; ih, iw = ir_gray.shape[:2]
    for pre, dets, gray, gw, gh in [("rgb", rgb_dets, rgb_gray, rw, rh), ("ir", ir_dets, ir_gray, iw, ih)]:
        if dets:
            tf = compute_target_features(gray, max(dets, key=lambda d: d[1])[0], gw, gh)
            feats.update({f"{pre}_best_{k}": v for k, v in tf.items()})
        else:
            feats.update({f"{pre}_best_{k}": 0. for k in TARGET_NAMES})
    rd, idd = len(rgb_dets) > 0, len(ir_dets) > 0
    feats.update({"both_detect": int(rd and idd), "neither_detect": int(not rd and not idd),
                  "rgb_only_detect": int(rd and not idd), "ir_only_detect": int(not rd and idd)})
    return feats


def best6(dets):
    if not dets:
        return 0., 0., 0.
    bc = max(c for _, c in dets); bb = max(dets, key=lambda d: d[1])[0]
    pw = max(1., bb[2]-bb[0]); ph = max(1., bb[3]-bb[1])
    return bc, round(float(np.log(pw*ph+1)), 4), round(float(pw/ph), 4)


def main():
    r6 = joblib.load(R6); fn = joblib.load(FNFN)
    r6m, r6f = r6["model"], r6["features"]; fnm, fnf = fn["model"], fn["features"]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = MLPVerifier(str(MLP), device=dev); patch = PatchVerifier(str(PATCH))
    print(f"filters on {dev}; classifiers on CPU (sklearn/xgboost)")

    keys = sorted(MANIFEST.keys())[:N*3]
    S = []
    for fr in glob.glob(str(REPO / "eval/results/_email_recompute/cache/svanstrom_0000.pkl")):
        for f in pickle.load(open(fr, "rb"))["frames"]:
            e = MANIFEST.get(f["key"])
            if not e:
                continue
            rp, ip = img_from_label(e["rgb_lbl"]), img_from_label(e["ir_lbl"])
            if not rp or not ip:
                continue
            ri, ii = cv2.imread(str(rp)), cv2.imread(str(ip))
            if ri is None or ii is None:
                continue
            rd = [(tuple(f["rgb"]["boxes"][i]), float(f["rgb"]["confs"][i])) for i in range(len(f["rgb"]["boxes"]))]
            idd = [(tuple(f["ir"]["boxes"][i]), float(f["ir"]["confs"][i])) for i in range(len(f["ir"]["boxes"]))]
            S.append({"rg": cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY), "ig": cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY),
                      "rd": rd, "id": idd, "rfeat": np.asarray(f["rgb"]["feats"], np.float32),
                      "ri": ri, "rboxes": [b for b, _ in rd]})
            if len(S) >= N:
                break
        if len(S) >= N:
            break
    nd = sum(len(s["rd"]) for s in S)
    print(f"frames={len(S)}  rgb_dets={nd}")

    def timeit(fn_, reps=REP):
        t = time.perf_counter()
        for _ in range(reps):
            fn_()
        return (time.perf_counter() - t) / reps

    # robust6 feature build + predict
    def r6_feat():
        for s in S:
            rmc, rla, rar = best6(s["rd"]); imc, ila, iar = best6(s["id"])
            fm = {"rgb_max_conf": rmc, "ir_max_conf": imc, "rgb_best_log_bbox_area": rla,
                  "ir_best_log_bbox_area": ila, "rgb_best_aspect_ratio": rar, "ir_best_aspect_ratio": iar}
            _ = [fm[k] for k in r6f]
    t_r6f = timeit(r6_feat) / len(S) * 1000
    X6 = np.array([[ {"rgb_max_conf": best6(s["rd"])[0], "ir_max_conf": best6(s["id"])[0],
        "rgb_best_log_bbox_area": best6(s["rd"])[1], "ir_best_log_bbox_area": best6(s["id"])[1],
        "rgb_best_aspect_ratio": best6(s["rd"])[2], "ir_best_aspect_ratio": best6(s["id"])[2]}[k]
        for k in r6f] for s in S], np.float32)
    t_r6p = timeit(lambda: r6m.predict(X6)) / len(S) * 1000

    # fusion_no_fn feature build + predict
    def fn_feat():
        for s in S:
            d = build40(s["rd"], s["id"], s["rg"], s["ig"])
            _ = [d.get(k, 0) for k in fnf]
    t_fnf = timeit(fn_feat, reps=max(3, REP // 4)) / len(S) * 1000
    Xfn = np.array([[build40(s["rd"], s["id"], s["rg"], s["ig"]).get(k, 0) for k in fnf] for s in S], np.float32)
    t_fnp = timeit(lambda: fnm.predict(Xfn)) / len(S) * 1000

    # filters per detection
    allfeat = np.concatenate([s["rfeat"] for s in S if len(s["rfeat"])])
    t_mlp = timeit(lambda: mlp.predict_drone_probs(allfeat)) / max(len(allfeat), 1) * 1000
    def patch_all():
        for s in S:
            if s["rboxes"]:
                patch.predict_boxes(s["ri"], s["rboxes"])
    t_patch = timeit(patch_all, reps=3) / max(nd, 1) * 1000

    print("\n=== CLASSIFIER latency (per frame, CPU) ===")
    print(f"  {'classifier':<16}{'feat_ext_ms':>12}{'predict_ms':>12}{'total_ms':>10}")
    print(f"  {'robust6 (6f)':<16}{t_r6f:>12.3f}{t_r6p:>12.3f}{t_r6f+t_r6p:>10.3f}")
    print(f"  {'fusion_no_fn(40)':<16}{t_fnf:>12.3f}{t_fnp:>12.3f}{t_fnf+t_fnp:>10.3f}")
    print(f"  -> robust6 feature-extraction is {t_fnf/max(t_r6f,1e-6):.0f}x cheaper; "
          f"total {(t_fnf+t_fnp)/max(t_r6f+t_r6p,1e-6):.1f}x faster")
    print("\n=== FILTER latency (per detection, CPU) ===")
    print(f"  MLP forward (mlp_v5)        {t_mlp:.4f} ms/det")
    print(f"  CNN patch (confuser v2)     {t_patch:.4f} ms/det   -> MLP {t_patch/max(t_mlp,1e-9):.0f}x faster (CPU)")


if __name__ == "__main__":
    main()
