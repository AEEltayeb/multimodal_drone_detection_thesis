"""compare_routing_pipeline.py — does the new routing classifier beat the old ones in the FULL pipeline?

ONE script, two jobs:
  STEP 1 (train+calibrate): train the proposed router = robust6 + rgb_mean_conf + is_grayscale on
          full56, then pick a per-class trust_rgb decision threshold tau on a held-out grayscale split
          (argmax suppresses the rare trust_rgb class; tau cashes the separability in). Saves model+tau.
  STEP 2 (full-pipeline eval): cache ft4(RGB)+v3b(IR) detections + MLP-verifier survivors + GT across
          PAIRED-thermal (antiuav, svanstrom), GRAYSCALE-drone (svanstrom_gray, video_drone) and
          GRAYSCALE-confuser (rgb_confuser, video_confuser) surfaces, then replay the frame-level alert
          ablation (bare/filter_only/clf_only/clf->filter/filter->clf) for THREE routers:
            sa32      (old 32-feat, argmax)   — its native 32 features
            robust6   (current, argmax)       — its native 6 features
            new       (robust6+rgb_mean_conf+is_grayscale, tau)  — its native 8 features
          Each router scored on ITS OWN feature definitions (sa32: generate_retrained_v2_data.build_row;
          robust6/new: generate_lean19_data.build_row = the source behind full56) so the comparison is fair.

Frame-level alert (pipeline fires iff >=1 drone detection survives). Metric: drone-surface P/R/F1 +
confuser fire-rate. Resumable per-surface cache. GPU for caching; ablation is zero-GPU.

  py -u eval/compare_routing_pipeline.py --target 4000                       # full run
  py -u eval/compare_routing_pipeline.py --target 30 --surfaces svanstrom_gray,rgb_confuser   # smoke
  py -u eval/compare_routing_pipeline.py --ablate-only                       # re-replay from cache
"""
from __future__ import annotations
import argparse, json, pickle, re, time, traceback
from pathlib import Path
import numpy as np, cv2, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
import sys
for p in ("classifier", "eval"):
    sys.path.insert(0, str(REPO / p))
from generate_retrained_v2_data import build_row as build_row32, FEATURE_COLS as FCOLS32   # 32-feat (sa32)  # noqa
from generate_lean19_data import build_row as build_row_lean, parse_yolo_gt                # lean defs (full56)  # noqa
from rebuild_yolo_cache import iter_antiuav_pairs, iter_svanstrom_pairs                     # noqa
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features               # noqa
from eval_v4_vs_patch import MLPv4Verifier                                                  # noqa

OUT = REPO / "eval" / "results" / "_routing_pipeline_cmp"
(OUT / "cache").mkdir(parents=True, exist_ok=True)
FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
MLP_V5 = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
ALIGNED_THR = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
ALIGNED_GRAY = str(REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt")
SA32 = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
ROBUST6_JBL = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
ROBUST8_JBL = REPO / "models/routers/robust8.joblib"   # SHIPPED production router (models.csv weights_path)
FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
VIDEO_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"

ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
F8 = ROBUST6 + ["rgb_mean_conf", "is_grayscale"]          # new router's native feature order
RGB_THR_MLP, IR_THR_MLP, CONF = 0.25, 0.05, 0.25
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


# ─────────────────────────── STEP 1: train + calibrate the new router ───────────────────────────
def train_new_router(tau_override=None):
    import pandas as pd
    df = pd.read_csv(FULL56)
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")
    df["is_grayscale"] = (df["regime"] == "grayscale").astype(int)
    df["_seq"] = [seq_id(s, c) for s, c in zip(df["stem"], df["source"])]
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(df, df["trust_label"], df["_seq"]))
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="multi:softprob", num_class=4,
                      eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=4)
    m.fit(df.iloc[tr][F8].values, df.iloc[tr]["trust_label"].values)
    # pick tau on held-out GRAYSCALE: maximize trust_rgb F1 under rule "trust_rgb if P1>=tau else argmax"
    gte = te[df.iloc[te]["regime"].values == "grayscale"]
    yg = df.iloc[gte]["trust_label"].values
    pg = m.predict_proba(df.iloc[gte][F8].values)
    rows = []
    for tau in np.round(np.arange(0.05, 0.51, 0.05), 2):
        pred = np.where(pg[:, 1] >= tau, 1, pg.argmax(1))
        p, r, f, _ = precision_recall_fscore_support(yg, pred, labels=[1], zero_division=0)
        rows.append((float(tau), float(r[0]), float(p[0]), float(f[0])))
    best = max(rows, key=lambda x: x[3])
    tau = tau_override if tau_override is not None else best[0]
    chosen = next(x for x in rows if abs(x[0] - tau) < 1e-9)
    print("STEP 1 — new router = robust6 + rgb_mean_conf + is_grayscale")
    print(f"  held-out grayscale trust_rgb tau sweep (tau, R, P, F1):")
    for t, r, p, f in rows:
        mark = "  <- chosen" if abs(t - tau) < 1e-9 else ("  (max-F1)" if t == best[0] else "")
        print(f"    tau={t:.2f}  R={r:.3f} P={p:.3f} F1={f:.3f}{mark}")
    print(f"  CHOSEN tau={tau:.2f}  (gray trust_rgb R={chosen[1]:.3f} P={chosen[2]:.3f} F1={chosen[3]:.3f})")
    clf = {"model": m, "features": F8, "tau": tau, "feat_key": "f8"}
    joblib.dump(clf, OUT / "new_router.joblib")
    return clf


# ─────────────────────────── STEP 2: detection + feature caching ───────────────────────────
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


def mean_conf(dets):
    cs = [d[4] for d in dets if d[4] >= CONF]
    return float(np.mean(cs)) if cs else 0.0


def f8_vec(rgb_dets, ir_dets, rgb_g, ir_g, rwh, iwh, is_gray, k, name):
    d = build_row_lean(rgb_dets, ir_dets, rgb_g, ir_g, rwh, iwh, 0, k, name, CONF)
    v = [float(d[f]) for f in ROBUST6]
    v.append(mean_conf(rgb_dets)); v.append(float(is_gray))
    return v


def iter_video(cats):
    for cat in cats:
        cd = VIDEO_ROOT / cat
        if not cd.exists():
            continue
        for clip in sorted(p for p in cd.iterdir() if p.is_dir()):
            img_d = clip / "images" / "test" if (clip / "images" / "test").exists() else clip / "images"
            lbl_d = clip / "labels" / "test" if (clip / "labels" / "test").exists() else clip / "labels"
            if not img_d.exists():
                continue
            for ip in sorted(p for p in img_d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
                yield {"key": f"{cat}_{clip.name}_{ip.stem}", "rgb_img": ip, "ir_img": None,
                       "rgb_lbl": lbl_d / f"{ip.stem}.txt"}


def iter_confuser_dir(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []
    for p in imgs:
        yield {"key": p.stem, "rgb_img": p, "ir_img": None, "rgb_lbl": None}


# (name, iterator-factory, rgb_imgsz, ir_imgsz, ir_mode, rule, has_drones)
def surface_defs():
    return {
        "antiuav":        (iter_antiuav_pairs, 640, 640, "thermal", "iou", True),
        "svanstrom":      (iter_svanstrom_pairs, 1280, 640, "thermal", "iop", True),
        "svanstrom_gray": (iter_svanstrom_pairs, 1280, 640, "gray", "iop", True),
        "video_drone":    (lambda: iter_video(["drone"]), 640, 640, "gray", "iop", True),
        "rgb_confuser":   (lambda: iter_confuser_dir(Path("G:/drone/rgb_confusers_merged/images/test")), 640, 640, "gray", "iou", False),
        "rgb_bird_confuser": (lambda: iter_confuser_dir(Path("G:/drone/bird.v1i.yolo26-birds-zekpr-bird-pn3pj/train/images")), 640, 640, "gray", "iou", False),
        "video_confuser": (lambda: iter_video(["birds", "airplanes", "helicopters"]), 640, 640, "gray", "iou", False),
    }


def cache_surface(name, pairs, rsz, isz, ir_mode, rule, has_drones, target,
                  yolo_r, hook_r, yolo_i, hook_i, mlp_r, mlp_i_thr, mlp_i_gray):
    pairs = list(pairs)
    st = max(1, len(pairs) // target)
    pairs = pairs[::st][:target]
    mlp_i = mlp_i_thr if ir_mode == "thermal" else mlp_i_gray
    is_gray = 0 if ir_mode == "thermal" else 1
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
                _, ir_g = gray3(ir); ir_in = ir
            else:
                ir_in, ir_g = rgb3, rgb_g
            rgb_dets, rgb_feats = run_det(yolo_r, hook_r, rgb, rsz)
            ir_dets, ir_feats = run_det(yolo_i, hook_i, ir_in, isz)
            rgb_p = mlp_r.predict_drone_probs(rgb_feats) if len(rgb_feats) else np.zeros(0)
            ir_p = mlp_i.predict_drone_probs(ir_feats) if len(ir_feats) else np.zeros(0)
            rgb_keep = [d for d, p in zip(rgb_dets, rgb_p) if p >= RGB_THR_MLP]
            ir_keep = [d for d, p in zip(ir_dets, ir_p) if p >= IR_THR_MLP]
            rh, rw = rgb.shape[:2]; ih, iw = ir_in.shape[:2]
            ngt = len(parse_yolo_gt(pr["rgb_lbl"], rw, rh)) if pr.get("rgb_lbl") else 0
            r32a = build_row32(rgb_dets, ir_dets, rgb_g, ir_g, (rw, rh), (iw, ih), 0, k, name, CONF)
            r32f = build_row32(rgb_keep, ir_keep, rgb_g, ir_g, (rw, rh), (iw, ih), 0, k, name, CONF)
            rows.append({
                "rgb_any": len(rgb_dets) > 0, "ir_any": len(ir_dets) > 0,
                "rgb_surv": len(rgb_keep) > 0, "ir_surv": len(ir_keep) > 0, "n_gt": ngt,
                "f8_all": f8_vec(rgb_dets, ir_dets, rgb_g, ir_g, (rw, rh), (iw, ih), is_gray, k, name),
                "f8_flt": f8_vec(rgb_keep, ir_keep, rgb_g, ir_g, (rw, rh), (iw, ih), is_gray, k, name),
                "f32_all": [float(r32a[f]) for f in FCOLS32],
                "f32_flt": [float(r32f[f]) for f in FCOLS32],
            })
        except Exception:
            print(f"    [frame-err {name} #{k}] {traceback.format_exc(limit=1)}")
    meta = {"name": name, "rule": rule, "n": len(rows), "stride": st, "has_drones": has_drones}
    pickle.dump({"meta": meta, "rows": rows, "F8": F8, "F32": FCOLS32},
                open(OUT / "cache" / f"{name}.pkl", "wb"))
    print(f"  [{name}] {len(rows)} frames, stride={st}, {len(rows)/max(time.time()-t0,.01):.1f} fps")
    return meta


# ─────────────────────────── ablation replay ───────────────────────────
def predict_trust(clf, F8_order, F32_order, rows, which):
    """which in {'all','flt'}. Returns 4-class labels. Applies tau rule if clf has tau."""
    order = F32_order if clf.get("feat_key") == "f32" else F8_order
    vec = f"f32_{which}" if clf.get("feat_key") == "f32" else f"f8_{which}"
    model, feats = clf["model"], clf.get("features")
    if feats is None:                                   # raw model expects the full vector in order
        X = np.array([r[vec] for r in rows], float)
    else:
        idx = [order.index(f) for f in feats]
        X = np.array([[r[vec][i] for i in idx] for r in rows], float)
    if clf.get("tau") is not None:
        proba = model.predict_proba(X)
        return np.where(proba[:, 1] >= clf["tau"], 1, proba.argmax(1))
    return model.predict(X)


def ablate(meta, rows, F8_order, F32_order, classifiers):
    hd = meta["has_drones"]
    ngt = np.array([r["n_gt"] for r in rows]); pos = ngt > 0
    rgb_any = np.array([r["rgb_any"] for r in rows]); ir_any = np.array([r["ir_any"] for r in rows])
    rgb_s = np.array([r["rgb_surv"] for r in rows]); ir_s = np.array([r["ir_surv"] for r in rows])

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
            t_all = predict_trust(clf, F8_order, F32_order, rows, "all")
            t_flt = predict_trust(clf, F8_order, F32_order, rows, "flt")
        except Exception as e:
            print(f"    [clf {cname} predict fail: {e}]"); continue
        trgb_a = np.isin(t_all, [1, 3]); tir_a = np.isin(t_all, [2, 3])
        res[f"clf_only[{cname}]"] = metrics(t_all != 0)
        res[f"clf->filter[{cname}]"] = metrics((trgb_a & rgb_s) | (tir_a & ir_s))
        res[f"filter->clf[{cname}]"] = metrics(t_flt != 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=4000, help="frames/surface")
    ap.add_argument("--surfaces", default="", help="comma list to subset (default: all)")
    ap.add_argument("--tau", type=float, default=None, help="override new-router trust_rgb threshold")
    ap.add_argument("--ablate-only", action="store_true", help="skip caching; replay from existing cache")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # EVAL-ONLY: LOAD the shipped production robust8 (models.csv weights_path); do NOT retrain.
    new_clf = joblib.load(ROBUST8_JBL)
    new_clf["feat_key"] = "f8"; new_clf["tau"] = 0.20           # ship setting: per-class trust_rgb τ=0.20
    tau = args.tau if args.tau is not None else 0.20
    new_clf["tau"] = tau
    print(f"STEP 1 — LOADED shipped robust8 ({ROBUST8_JBL.name}) @ τ={tau:.2f} (eval-only; NO training)")
    sa32_raw = joblib.load(SA32)
    sa32 = ({"model": sa32_raw["model"], "features": sa32_raw.get("features"), "feat_key": "f32"}
            if isinstance(sa32_raw, dict) else {"model": sa32_raw, "features": None, "feat_key": "f32"})
    robust6 = joblib.load(ROBUST6_JBL); robust6["feat_key"] = "f8"   # {model, features(6)}
    # all three classifiers are LOADED artifacts (no in-eval training); robust8 at its ship τ.
    classifiers = {"sa32": sa32, "robust6": robust6, f"robust8@{tau:.2f}": new_clf}

    defs = surface_defs()
    want = [s.strip() for s in args.surfaces.split(",") if s.strip()] or list(defs)

    if not args.ablate_only:
        from ultralytics import YOLO
        yolo_r = YOLO(FT4); hook_r = DetectInputHook(); hook_r.register(yolo_r)
        yolo_i = YOLO(V3B); hook_i = DetectInputHook(); hook_i.register(yolo_i)
        mlp_r = MLPv4Verifier(Path(MLP_V5), device="cuda")
        mlp_i_thr = MLPv4Verifier(Path(ALIGNED_THR), device="cuda")
        mlp_i_gray = mlp_i_thr   # use the SINGLE aligned IR verifier (mlp_v5_ir_aligned) for grayscale too
        for name in want:
            if (OUT / "cache" / f"{name}.pkl").exists() and not args.overwrite:
                print(f"  [skip:cached] {name}"); continue
            it, rsz, isz, irm, rule, hd = defs[name]
            try:
                cache_surface(name, it(), rsz, isz, irm, rule, hd, args.target,
                              yolo_r, hook_r, yolo_i, hook_i, mlp_r, mlp_i_thr, mlp_i_gray)
            except Exception:
                print(f"  [SURFACE-ERR {name}]\n{traceback.format_exc()}")

    # ---- replay + report ----
    all_res, lines = {}, ["# Routing-Classifier Full-Pipeline Comparison",
                          f"{time.strftime('%Y-%m-%d %H:%M')} | new tau={new_clf['tau']}\n"]
    scorecard = {}   # name -> {surface: {cell: metric}}
    for name in want:
        pkl = OUT / "cache" / f"{name}.pkl"
        if not pkl.exists():
            continue
        d = pickle.load(open(pkl, "rb")); meta = d["meta"]
        res = ablate(meta, d["rows"], d["F8"], d["F32"], classifiers)
        all_res[name] = res; scorecard[name] = (meta, res)
        lines.append(f"\n## {name} (n={meta['n']}, rule={meta['rule']}, drones={meta['has_drones']})\n")
        hdr = ("| cell | TP | FP | FN | P | R | F1 | fire |\n|---|---|---|---|---|---|---|---|"
               if meta["has_drones"] else "| cell | FP | fire_rate |\n|---|---|---|")
        lines.append(hdr)
        print(f"\n== {name} (n={meta['n']}) ==")
        for cell, m in res.items():
            if meta["has_drones"]:
                print(f"   {cell:<24} P={m.get('precision',0):.3f} R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f} fire={m['fire_rate']}")
                lines.append(f"| {cell} | {m['tp']} | {m['fp']} | {m['fn']} | {m.get('precision')} | {m.get('recall')} | {m.get('f1')} | {m['fire_rate']} |")
            else:
                print(f"   {cell:<24} FP={m['fp']} fire={m['fire_rate']}")
                lines.append(f"| {cell} | {m['fp']} | {m['fire_rate']} |")

    # ---- HEAD-TO-HEAD scorecard: the tradeoff that decides the winner ----
    lines.append("\n## SCORECARD — clf->filter cell (production cascade)\n")
    lines.append("| router | thermal drone F1 | GRAYSCALE drone recall | confuser fire-rate |")
    lines.append("|---|---|---|---|")
    print("\n" + "=" * 70 + "\nSCORECARD (clf->filter cell): who wins the recall-vs-false-alert tradeoff")
    def cell_metric(name, cellname, key):
        vals = []
        for sname, (meta, res) in scorecard.items():
            if name == "thermal" and sname in ("antiuav", "svanstrom") and cellname in res:
                vals.append(res[cellname].get(key))
            elif name == "graydrone" and sname in ("svanstrom_gray", "video_drone") and cellname in res:
                vals.append(res[cellname].get(key))
            elif name == "confuser" and sname in ("rgb_confuser", "video_confuser") and cellname in res:
                vals.append(res[cellname].get(key))
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")
    for cname in classifiers:
        cell = f"clf->filter[{cname}]"
        tf1 = cell_metric("thermal", cell, "f1")
        gr = cell_metric("graydrone", cell, "recall")
        cf = cell_metric("confuser", cell, "fire_rate")
        print(f"  {cname:<9} thermal_F1={tf1:.3f}  GRAY_drone_R={gr:.3f}  confuser_fire={cf:.3f}")
        lines.append(f"| {cname} | {tf1:.3f} | {gr:.3f} | {cf:.3f} |")

    json.dump(all_res, open(OUT / "comparison.json", "w"), indent=2)
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE -> {OUT/'comparison.md'}")


if __name__ == "__main__":
    main()
