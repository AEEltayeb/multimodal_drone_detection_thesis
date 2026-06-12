"""overnight_ablation.py — full-pipeline ablation, built to run overnight.

Question: is the new robust6 trust classifier production vs sa32 / none, and in which
cascade order (classifier->filter vs filter->classifier)?

For each PAIRED dataset (2 drone + >=1 confuser), strided to ~N frames, runs ft4 (RGB) +
v3b (IR; thermal where paired, RGB->gray fallback for confuser), extracts per-detection
517-D features -> MLP verifier probs (RGB mlp_v5, IR aligned_thermal), and the 32 fusion
features (faithful sa32 set; robust6 = named 6-subset). Caches per frame, then runs the
frame-level alert ablation. Defensive + resumable (per-dataset cache skip).

Ablation cells (frame-level alert: pipeline fires iff >=1 drone detection survives):
  bare | filter_only | clf_only(C) | clf->filter(C) | filter->clf(C)   for C in {sa32, robust6}
Metric per dataset: alert P/R/F1 (drone frames vs not) + confuser fire-rate.

Composition faithful to ledger dual-verifier-fusion-rule / pyside_engine._mlp_trust_first:
  clf->filter : trust=C(all-det feats); trusted modality keeps alert iff a det survives its
                verifier (trusted modality fully vetoed -> loses trust). recall-first OR.
  filter->clf : verifier filters dets first; trust=C(surviving-det feats); alert iff trust!=0.

  py -u eval/overnight_ablation.py --target 5000
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
from generate_retrained_v2_data import build_row, FEATURE_COLS       # 32 fusion feats (sa32-faithful)  # noqa
from generate_lean19_data import parse_yolo_gt                         # noqa
from rebuild_yolo_cache import iter_antiuav_pairs, iter_svanstrom_pairs  # noqa
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features  # noqa
from eval_v4_vs_patch import MLPv4Verifier                             # noqa

OUT = REPO / "eval" / "results" / "_overnight_ablation"
(OUT / "cache").mkdir(parents=True, exist_ok=True)
FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
MLP_V5 = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
ALIGNED = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
SA32 = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
ROBUST6 = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
RGB_THR_MLP, IR_THR_MLP = 0.25, 0.05
CONF = 0.25


def gray3(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), g


def run_det(yolo, hook, img, imgsz):
    hook.clear()
    r = yolo.predict(img, imgsz=imgsz, conf=CONF, verbose=False, device="cuda")[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return [], np.zeros((0, 0))
    boxes = [tuple(b.xyxy[i].cpu().numpy().tolist()) + (float(b.conf[i]),) for i in range(len(b))]
    ih, iw = img.shape[:2]
    feats = np.stack([_extract_detection_features(hook, db[:4], (ih, iw), db[4]) for db in boxes])
    return boxes, feats


def iter_confuser(rgb_dir: Path, target):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in exts) if rgb_dir.exists() else []
    st = max(1, len(imgs) // target)
    for p in imgs[::st][:target]:
        yield {"key": p.stem, "rgb_img": p, "ir_img": None, "rgb_lbl": None}


def cache_dataset(name, pairs, rgb_imgsz, ir_imgsz, ir_mode, rule, target,
                  yolo_r, hook_r, yolo_i, hook_i, mlp_r, mlp_i):
    pairs = list(pairs)
    st = max(1, len(pairs) // target) if name != "rgb_confuser" else 1
    pairs = pairs[::st][:target]
    rows, t0 = [], time.time()
    for k, pr in enumerate(pairs):
        try:
            rgb = cv2.imread(str(pr["rgb_img"]))
            if rgb is None:
                continue
            rgb3, rgb_g = gray3(rgb)
            if ir_mode == "thermal" and pr.get("ir_img"):
                ir = cv2.imread(str(pr["ir_img"]))
                if ir is None:
                    continue
                _, ir_g = gray3(ir)
                ir_in = ir
            else:  # grayscale fallback: feed RGB->gray to v3b
                ir_in, ir_g = rgb3, rgb_g
            rgb_dets, rgb_feats = run_det(yolo_r, hook_r, rgb, rgb_imgsz)
            ir_dets, ir_feats = run_det(yolo_i, hook_i, ir_in, ir_imgsz)
            rgb_p = mlp_r.predict_drone_probs(rgb_feats) if len(rgb_feats) else np.zeros(0)
            ir_p = mlp_i.predict_drone_probs(ir_feats) if len(ir_feats) else np.zeros(0)
            # survivors under each modality verifier
            rgb_keep = [d for d, p in zip(rgb_dets, rgb_p) if p >= RGB_THR_MLP]
            ir_keep = [d for d, p in zip(ir_dets, ir_p) if p >= IR_THR_MLP]
            rh, rw = rgb.shape[:2]; ih, iw = ir_in.shape[:2]
            feats_all = build_row(rgb_dets, ir_dets, rgb_g, ir_g, (rw, rh), (iw, ih), 0, k, name, CONF)
            feats_flt = build_row(rgb_keep, ir_keep, rgb_g, ir_g, (rw, rh), (iw, ih), 0, k, name, CONF)
            ngt = len(parse_yolo_gt(pr["rgb_lbl"], rw, rh)) if pr.get("rgb_lbl") else 0
            rows.append({
                "rgb_any": len(rgb_dets) > 0, "ir_any": len(ir_dets) > 0,
                "rgb_surv": len(rgb_keep) > 0, "ir_surv": len(ir_keep) > 0,
                "f_all": [float(feats_all[f]) for f in FEATURE_COLS],
                "f_flt": [float(feats_flt[f]) for f in FEATURE_COLS],
                "n_gt": ngt,
            })
        except Exception:
            print(f"    [frame-err {name} #{k}] {traceback.format_exc(limit=1)}")
    meta = {"name": name, "rule": rule, "n": len(rows), "stride": st,
            "has_drones": name != "rgb_confuser"}
    pickle.dump({"meta": meta, "rows": rows, "feat_order": FEATURE_COLS},
                open(OUT / "cache" / f"{name}.pkl", "wb"))
    print(f"  [{name}] {len(rows)} frames, stride={st}, {len(rows)/max(time.time()-t0,.01):.1f} fps")
    return meta


def predict_trust(clf, feat_order, frows, key):
    """clf: dict{model,features} (robust6) or raw model (sa32, expects FEATURE_COLS)."""
    if isinstance(clf, dict) and "features" in clf:
        idx = [feat_order.index(f) for f in clf["features"]]; model = clf["model"]
    else:
        idx = list(range(len(feat_order))); model = clf
    X = np.array([[r[key][i] for i in idx] for r in frows], dtype=float)
    return model.predict(X)


def ablate(meta, rows, classifiers):
    hd = meta["has_drones"]
    ngt = np.array([r["n_gt"] for r in rows]); pos = ngt > 0
    rgb_any = np.array([r["rgb_any"] for r in rows]); ir_any = np.array([r["ir_any"] for r in rows])
    rgb_s = np.array([r["rgb_surv"] for r in rows]); ir_s = np.array([r["ir_surv"] for r in rows])
    fo = meta["feat_order"] if "feat_order" in meta else FEATURE_COLS

    def metrics(alert):
        tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum())
        fn = int((~alert & pos).sum()); tn = int((~alert & ~pos).sum())
        out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fire_rate": round(fp / max(fp + tn, 1), 4)}
        if hd:
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            out.update({"precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(2 * p * r / max(p + r, 1e-9), 4)})
        return out

    res = {"bare": metrics(rgb_any | ir_any), "filter_only": metrics(rgb_s | ir_s)}
    for cname, clf in classifiers.items():
        try:
            t_all = predict_trust(clf, fo, rows, "f_all")
            t_flt = predict_trust(clf, fo, rows, "f_flt")
        except Exception as e:
            print(f"    [clf {cname} predict fail: {e}]"); continue
        trgb_a = np.isin(t_all, [1, 3]); tir_a = np.isin(t_all, [2, 3])
        res[f"clf_only[{cname}]"] = metrics(t_all != 0)
        # clf->filter: trusted modality keeps alert iff a det survives its verifier
        res[f"clf->filter[{cname}]"] = metrics((trgb_a & rgb_s) | (tir_a & ir_s))
        # filter->clf: trust from surviving-det features
        res[f"filter->clf[{cname}]"] = metrics(t_flt != 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    print(f"Overnight ablation: target {args.target}/dataset\nFEATURE_COLS={len(FEATURE_COLS)} feats")

    yolo_r = YOLO(FT4); hook_r = DetectInputHook(); hook_r.register(yolo_r)
    yolo_i = YOLO(V3B); hook_i = DetectInputHook(); hook_i.register(yolo_i)
    mlp_r = MLPv4Verifier(Path(MLP_V5), device="cuda")
    mlp_i = MLPv4Verifier(Path(ALIGNED), device="cuda")

    DS = [
        ("antiuav", iter_antiuav_pairs, 640, 640, "thermal", "iou"),
        ("svanstrom", iter_svanstrom_pairs, 1280, 640, "thermal", "iop"),
        ("rgb_confuser", lambda: iter_confuser(Path("G:/drone/rgb_confusers_merged/images/test"), args.target), 640, 640, "gray", "iou"),
    ]
    for (name, it, rsz, isz, irm, rule) in DS:
        if (OUT / "cache" / f"{name}.pkl").exists() and not args.overwrite:
            print(f"  [skip:cached] {name}"); continue
        try:
            cache_dataset(name, it(), rsz, isz, irm, rule, args.target,
                          yolo_r, hook_r, yolo_i, hook_i, mlp_r, mlp_i)
        except Exception:
            print(f"  [DATASET-ERR {name}]\n{traceback.format_exc()}")

    # ---- ablation (load classifiers, replay) ----
    classifiers = {}
    for cname, path in [("sa32", SA32), ("robust6", ROBUST6)]:
        try:
            classifiers[cname] = joblib.load(path)
            print(f"  loaded clf {cname}")
        except Exception as e:
            print(f"  [clf {cname} load fail: {e}]")

    all_res, lines = {}, ["# Overnight Full-Pipeline Ablation\n", f"{time.strftime('%Y-%m-%d %H:%M')}\n"]
    for pkl in sorted((OUT / "cache").glob("*.pkl")):
        d = pickle.load(open(pkl, "rb")); meta = d["meta"]; meta["feat_order"] = d["feat_order"]
        res = ablate(meta, d["rows"], classifiers)
        all_res[meta["name"]] = res
        lines.append(f"\n## {meta['name']} (n={meta['n']}, rule={meta['rule']}, drones={meta['has_drones']})\n")
        if meta["has_drones"]:
            lines.append("| cell | TP | FP | FN | P | R | F1 | fire |"); lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| cell | FP | fire_rate |"); lines.append("|---|---|---|")
        print(f"\n== {meta['name']} (n={meta['n']}) ==")
        for cell, m in res.items():
            if meta["has_drones"]:
                print(f"   {cell:<22} P={m.get('precision'):.3f} R={m.get('recall'):.3f} F1={m.get('f1'):.3f} fire={m['fire_rate']}")
                lines.append(f"| {cell} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['fire_rate']} |")
            else:
                print(f"   {cell:<22} FP={m['fp']} fire={m['fire_rate']}")
                lines.append(f"| {cell} | {m['fp']} | {m['fire_rate']} |")
    json.dump(all_res, open(OUT / "ablation_results.json", "w"), indent=2)
    (OUT / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE -> {OUT/'ablation_results.md'}")


if __name__ == "__main__":
    main()
