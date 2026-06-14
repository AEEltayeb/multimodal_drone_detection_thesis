"""overnight_ablation_full.py — COMPREHENSIVE automatic pipeline ablation.

Extends overnight_ablation.py with:
  - filters: {none, patch, mlp}  (RGB: patch_v2 vs mlp_v5 ; IR: ir_patch vs aligned)
  - classifiers: {none, sa32, robust6}
  - cascade order: clf->filter, filter->clf
  - modality paths: native thermal IR AND grayscale-RGB->IR (svanstrom run BOTH for a
    direct native-vs-grayscale comparison)
  - more datasets (eval_1000 surfaces): antiuav, svanstrom(+gray), selcom_val,
    rgb_dataset_test, rgb_confuser, ir_dset_final, cbam.

Metric: frame-level ALERT (pipeline fires iff >=1 drone detection survives) P/R/F1 + fire.
IR-only surfaces (no paired RGB) skip classifier cells (verifier-only). Defensive +
resumable (per-surface cache skip).  Composition faithful to dual-verifier-fusion-rule.

  py -u eval/overnight_ablation_full.py --target 5000
"""
from __future__ import annotations
import argparse, json, pickle, time, traceback
from pathlib import Path
import numpy as np, cv2, joblib
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
import sys
for p in ("classifier", "eval"):
    sys.path.insert(0, str(REPO / p))
from generate_retrained_v2_data import build_row, FEATURE_COLS          # noqa
from rebuild_yolo_cache import iter_antiuav_pairs, iter_svanstrom_pairs # noqa
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features  # noqa
from eval_v4_vs_patch import MLPv4Verifier                              # noqa
from patch_verifier import PatchVerifier                               # noqa

OUT = REPO / "eval" / "results" / "_overnight_ablation_full"
(OUT / "cache").mkdir(parents=True, exist_ok=True)
FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
MLP_V5 = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
ALIGNED_T = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
ALIGNED_G = str(REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt")
RGB_PATCH = str(REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt")
IR_PATCH = str(REPO / "models/patches/confuser_filter4_ir_v2_backup.pt")
SA32 = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
ROBUST6 = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
RGB_MLP_THR, IR_MLP_THR, PATCH_THR, CONF = 0.25, 0.05, 0.5, 0.25
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def gray3(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), g


def parse_gt(lbl, w, h, cls=None):
    out = []
    if not lbl:
        return out
    p = Path(lbl)
    if not p.exists():
        return out
    for line in p.read_text().strip().splitlines():
        a = line.split()
        if len(a) < 5:
            continue
        if cls is not None and int(a[0]) != cls:
            continue
        cx, cy, bw, bh = map(float, a[1:5])
        out.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return out


def run_det(yolo, hook, img, imgsz):
    hook.clear()
    r = yolo.predict(img, imgsz=imgsz, conf=CONF, verbose=False, device="cuda")[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return [], np.zeros((0, 0))
    dets = [tuple(b.xyxy[i].cpu().numpy().tolist()) + (float(b.conf[i]),) for i in range(len(b))]
    ih, iw = img.shape[:2]
    feats = np.stack([_extract_detection_features(hook, d[:4], (ih, iw), d[4]) for d in dets])
    return dets, feats


# ---- surface specs: (name, kind, args...) ----
def iter_rgb_dir(rgb_dir, lbl_layout, target, cls=0):
    d = Path(rgb_dir)
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []
    st = max(1, len(imgs) // target)
    for p in imgs[::st][:target]:
        if lbl_layout == "A":   # .../images/<f> + .../labels/<f>.txt
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        elif lbl_layout == "B": # .../images/<split>/<f> + .../labels/<split>/<f>.txt
            lbl = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        else:
            lbl = None
        yield {"rgb_img": p, "ir_img": None, "lbl": lbl, "cls": cls}


SURFACES = [
    # name, iter_fn, ir_mode(thermal/gray), has_rgb, has_drones, rgb_imgsz, ir_imgsz, rule, gt_cls
    ("antiuav",          lambda t: iter_antiuav_pairs(),  "thermal", True,  True,  640, 640, "iou", 0),
    ("svanstrom",        lambda t: iter_svanstrom_pairs(),"thermal", True,  True,  1280,640, "iop", 0),
    ("svanstrom_gray",   lambda t: iter_svanstrom_pairs(),"gray",    True,  True,  1280,640, "iop", 0),
    ("selcom_val",       lambda t: iter_rgb_dir("G:/drone/_finetune_selcom_mixed_ft2/images/val","B",t), "gray", True, True, 1280, 640, "iop", 0),
    ("rgb_dataset_test", lambda t: iter_rgb_dir("G:/drone/dataset/dataset/images/test","B",t), "gray", True, True, 640, 640, "iou", 0),
    ("rgb_confuser",     lambda t: iter_rgb_dir("G:/drone/rgb_confusers_merged/images/test","none",t), "gray", True, False, 640, 640, "iou", 0),
    ("ir_dset_final",    lambda t: iter_rgb_dir("G:/drone/IR_dset_final/test/images","A",t), "thermal", False, True, 640, 640, "iou", 0),
    ("cbam",             lambda t: iter_rgb_dir("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images","A",t), "thermal", False, True, 640, 640, "iou", 1),
]


def cache_surface(spec, target, M):
    name, itf, irmode, has_rgb, has_dr, rsz, isz, rule, cls = spec
    pairs = list(itf(target))
    st = max(1, len(pairs) // target)
    pairs = pairs[::st][:target]
    ir_mlp = M["aln_g"] if irmode == "gray" else M["aln_t"]
    ir_patch = M["rgb_patch"] if irmode == "gray" else M["ir_patch"]
    rows, t0 = [], time.time()
    for k, pr in enumerate(pairs):
        try:
            # ---- RGB branch ----
            if has_rgb and pr.get("rgb_img"):
                rgb = cv2.imread(str(pr["rgb_img"]))
                if rgb is None:
                    continue
                _, rgb_g = gray3(rgb)
                rgb_dets, rgb_f = run_det(M["ft4"], M["hk_r"], rgb, rsz)
                rgb_mlp = M["mlp_v5"].predict_drone_probs(rgb_f) if len(rgb_f) else np.zeros(0)
                rgb_pch = np.asarray(M["rgb_patch"].predict_boxes(rgb, [d[:4] for d in rgb_dets]), float) if rgb_dets else np.zeros(0)
                rw, rh = rgb.shape[1], rgb.shape[0]
            else:
                rgb_dets, rgb_mlp, rgb_pch, rgb_g, rw, rh = [], np.zeros(0), np.zeros(0), None, 1, 1
            # ---- IR branch ----
            if irmode == "gray":
                src = pr.get("rgb_img")
                base = cv2.imread(str(src)) if src else None
                if base is None:
                    continue
                ir_in, ir_g = gray3(base)
            else:
                ir_p = pr.get("ir_img") or pr.get("rgb_img")  # ir_dset/cbam: image is the IR itself
                ir_in = cv2.imread(str(ir_p))
                if ir_in is None:
                    continue
                _, ir_g = gray3(ir_in)
            ir_dets, ir_f = run_det(M["v3b"], M["hk_i"], ir_in, isz)
            ir_mlp_p = ir_mlp.predict_drone_probs(ir_f) if len(ir_f) else np.zeros(0)
            ir_pch = np.asarray(ir_patch.predict_boxes(ir_in, [d[:4] for d in ir_dets]), float) if ir_dets else np.zeros(0)
            iw, ih = ir_in.shape[1], ir_in.shape[0]
            if rgb_g is None:
                rgb_g = ir_g; rw, rh = iw, ih
            # ---- survivors per filter ----
            r_keep_mlp = [d for d, p in zip(rgb_dets, rgb_mlp) if p >= RGB_MLP_THR]
            i_keep_mlp = [d for d, p in zip(ir_dets, ir_mlp_p) if p >= IR_MLP_THR]
            r_keep_pch = [d for d, p in zip(rgb_dets, rgb_pch) if p < PATCH_THR]
            i_keep_pch = [d for d, p in zip(ir_dets, ir_pch) if p < PATCH_THR]
            def feats(rd, idd):
                row = build_row(rd, idd, rgb_g, ir_g, (rw, rh), (iw, ih), 0, k, name, CONF)
                return [float(row[f]) for f in FEATURE_COLS]
            lbl = pr.get("lbl") or pr.get("rgb_lbl")
            ngt = len(parse_gt(lbl, rw if has_rgb else iw, rh if has_rgb else ih, cls)) if has_dr else 0
            rows.append({
                "rgb_any": len(rgb_dets) > 0, "ir_any": len(ir_dets) > 0,
                "rs_mlp": len(r_keep_mlp) > 0, "is_mlp": len(i_keep_mlp) > 0,
                "rs_pch": len(r_keep_pch) > 0, "is_pch": len(i_keep_pch) > 0,
                "f_all": feats(rgb_dets, ir_dets),
                "f_mlp": feats(r_keep_mlp, i_keep_mlp),
                "f_pch": feats(r_keep_pch, i_keep_pch),
                "n_gt": ngt,
            })
        except Exception:
            print(f"    [frame-err {name} #{k}] {traceback.format_exc(limit=1)}")
    meta = {"name": name, "rule": rule, "has_drones": has_dr, "has_rgb": has_rgb,
            "ir_mode": irmode, "n": len(rows), "stride": st}
    pickle.dump({"meta": meta, "rows": rows}, open(OUT / "cache" / f"{name}.pkl", "wb"))
    print(f"  [{name}] {len(rows)} frames ({irmode}, rgb={has_rgb}), stride={st}, "
          f"{len(rows)/max(time.time()-t0,.01):.1f} fps")


def predict_trust(clf, frows, key):
    if isinstance(clf, dict) and "features" in clf:
        idx = [FEATURE_COLS.index(f) for f in clf["features"]]; model = clf["model"]
    else:
        idx = list(range(len(FEATURE_COLS))); model = clf
    X = np.array([[r[key][i] for i in idx] for r in frows], float)
    return model.predict(X)


def ablate(meta, rows, classifiers):
    hd, has_rgb = meta["has_drones"], meta["has_rgb"]
    pos = np.array([r["n_gt"] for r in rows]) > 0
    A = {k: np.array([r[k] for r in rows]) for k in
         ("rgb_any", "ir_any", "rs_mlp", "is_mlp", "rs_pch", "is_pch")}

    def M(alert):
        tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum())
        fn = int((~alert & pos).sum()); tn = int((~alert & ~pos).sum())
        o = {"tp": tp, "fp": fp, "fn": fn, "fire": round(fp / max(fp + tn, 1), 4)}
        if hd:
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            o.update({"precision": round(p, 4), "recall": round(r, 4), "f1": round(2*p*r/max(p+r, 1e-9), 4)})
        return o

    res = {"bare": M(A["rgb_any"] | A["ir_any"]),
           "filter_only[mlp]": M(A["rs_mlp"] | A["is_mlp"]),
           "filter_only[patch]": M(A["rs_pch"] | A["is_pch"])}
    if not has_rgb:   # IR-only: classifier (fusion) meaningless
        return res
    surv = {"mlp": (A["rs_mlp"], A["is_mlp"], "f_mlp"), "patch": (A["rs_pch"], A["is_pch"], "f_pch")}
    for cname, clf in classifiers.items():
        try:
            t_all = predict_trust(clf, rows, "f_all")
        except Exception as e:
            print(f"    [clf {cname} fail: {e}]"); continue
        trgb = np.isin(t_all, [1, 3]); tir = np.isin(t_all, [2, 3])
        res[f"clf_only[{cname}]"] = M(t_all != 0)
        for fn_, (rs, is_, fkey) in surv.items():
            res[f"clf->filter[{cname},{fn_}]"] = M((trgb & rs) | (tir & is_))
            t_f = predict_trust(clf, rows, fkey)
            res[f"filter->clf[{cname},{fn_}]"] = M(t_f != 0)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--overwrite", action="store_true"); args = ap.parse_args()
    print(f"FULL ablation: target {args.target}/surface, {len(SURFACES)} surfaces, {len(FEATURE_COLS)} feats")
    M = {}
    M["ft4"] = YOLO(FT4); M["hk_r"] = DetectInputHook(); M["hk_r"].register(M["ft4"])
    M["v3b"] = YOLO(V3B); M["hk_i"] = DetectInputHook(); M["hk_i"].register(M["v3b"])
    M["mlp_v5"] = MLPv4Verifier(Path(MLP_V5), device="cuda")
    M["aln_t"] = MLPv4Verifier(Path(ALIGNED_T), device="cuda")
    M["aln_g"] = MLPv4Verifier(Path(ALIGNED_G), device="cuda")
    M["rgb_patch"] = PatchVerifier(RGB_PATCH); M["ir_patch"] = PatchVerifier(IR_PATCH)

    for spec in SURFACES:
        name = spec[0]
        if (OUT / "cache" / f"{name}.pkl").exists() and not args.overwrite:
            print(f"  [skip:cached] {name}"); continue
        try:
            cache_surface(spec, args.target, M)
        except Exception:
            print(f"  [SURFACE-ERR {name}]\n{traceback.format_exc()}")

    classifiers = {}
    for cn, pth in [("sa32", SA32), ("robust6", ROBUST6)]:
        try:
            classifiers[cn] = joblib.load(pth); print(f"  loaded {cn}")
        except Exception as e:
            print(f"  [clf {cn} load fail: {e}]")

    allr, lines = {}, ["# Comprehensive Pipeline Ablation\n", f"{time.strftime('%Y-%m-%d %H:%M')}\n"]
    for pkl in sorted((OUT / "cache").glob("*.pkl")):
        d = pickle.load(open(pkl, "rb")); meta = d["meta"]
        res = ablate(meta, d["rows"], classifiers); allr[meta["name"]] = res
        lines.append(f"\n## {meta['name']} (n={meta['n']}, {meta['ir_mode']} IR, rgb={meta['has_rgb']}, rule={meta['rule']}, drones={meta['has_drones']})\n")
        if meta["has_drones"]:
            lines += ["| cell | TP | FP | FN | P | R | F1 | fire |", "|---|---|---|---|---|---|---|---|"]
        else:
            lines += ["| cell | FP | fire |", "|---|---|---|"]
        print(f"\n== {meta['name']} (n={meta['n']}, {meta['ir_mode']}) ==")
        for cell, m in res.items():
            if meta["has_drones"]:
                print(f"   {cell:<26} F1={m.get('f1'):.3f} P={m.get('precision'):.3f} R={m.get('recall'):.3f} fire={m['fire']}")
                lines.append(f"| {cell} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['fire']} |")
            else:
                print(f"   {cell:<26} FP={m['fp']} fire={m['fire']}")
                lines.append(f"| {cell} | {m['fp']} | {m['fire']} |")
    json.dump(allr, open(OUT / "ablation_full.json", "w"), indent=2)
    (OUT / "ablation_full.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE -> {OUT/'ablation_full.md'}")


if __name__ == "__main__":
    main()
