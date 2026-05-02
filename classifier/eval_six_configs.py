"""
eval_six_configs.py — per-detection evaluation of 6 configurations on paired
RGB-IR datasets, reusing cached YOLO inference.

Configs (applied on the SAME frame set):
  1. ir_only              — raw IR YOLO dets
  2. rgb_only             — raw RGB YOLO dets
  3. classifier           — meta classifier decides trust; kept per trusted mod
  4. ir_filter            — IR dets surviving confuser filter (p < 0.70)
  5. rgb_filter           — RGB dets surviving confuser filter
  6. classifier_filter    — filter first, then classifier on survivors

Per-detection scoring:
  TP: det IoU>=0.5 with any drone GT
  FP: det matches no drone GT
  FN: drone GT with no matching kept det

Outputs:
  runs/eval_six_configs/{dataset}/metrics.csv
  runs/eval_six_configs/{dataset}/confusion.json
  runs/eval_six_configs/{dataset}/progress.jsonl   (resume checkpoint)
  runs/eval_six_configs/{dataset}/patch_probs.json (cached filter scores)

Datasets:
  antiuav   -> G:/drone/Anti-UAV-RGBT_yolo_converted/test
  svanstrom -> G:/drone/svanstrom_paired   (not truly paired; caveat noted)

Usage:
  python classifier/eval_six_configs.py --dataset antiuav
  python classifier/eval_six_configs.py --dataset svanstrom
  python classifier/eval_six_configs.py --dataset both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "ir_gui"))

from fusion.features import TARGET_NAMES, compute_global_features, compute_target_features

from patch_verifier import PatchVerifier

RAW_ANTIUAV_BASE = SCRIPT_DIR / "runs" / "raw_detections"
RAW_SVAN_BASE    = SCRIPT_DIR / "runs" / "svanstrom_detections"
CLF_PATH    = SCRIPT_DIR / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib"
# CLI-overridable; resolved in main()
RAW_ANTIUAV = RAW_ANTIUAV_BASE.with_suffix(".json")
RAW_SVAN    = RAW_SVAN_BASE.with_suffix(".json")
PATCH_RGB   = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_rgb.pt"
PATCH_IR    = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"
OUT_ROOT    = SCRIPT_DIR / "runs" / "eval_six_configs"

ANTIUAV_RGB_IMG = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_IR_IMG  = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images")
SVAN_RGB_IMG    = Path("G:/drone/svanstrom_paired/RGB/images")
SVAN_IR_IMG     = Path("G:/drone/svanstrom_paired/IR/images")

IOU_MATCH   = 0.5     # overridable via --iou
IOP_MATCH   = 0.5     # overridable via --iop (intersection / pred area)
PATCH_THR   = 0.70    # overridable via --patch-thr
RGB_CONF    = 0.25    # overridable via --rgb-conf
IR_CONF     = 0.40    # overridable via --ir-conf
CKPT_EVERY  = 500

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")


def svan_category(key: str) -> str:
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"

CONFIG_NAMES = [
    "ir_only", "rgb_only", "classifier",
    "ir_filter", "rgb_filter", "classifier_filter",
]


# ── IO helpers ────────────────────────────────────────────────────

def read_yolo_labels(path: Path, w: int, h: int):
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5 or p[0] != "0":
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ))
    return boxes


def iou_iop(a, b):
    """Return (IoU, IoP) — IoP = intersection / pred(a) area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    iou = inter / u if u > 0 else 0.0
    iop = inter / aa if aa > 0 else 0.0
    return iou, iop


def iou(a, b):
    return iou_iop(a, b)[0]


def score_dets(dets, gts, rule="iou", iou_thr=0.5, iop_thr=0.5):
    """Per-det TP/FP + FN (unmatched GT). rule: 'iou' | 'iop'."""
    tp = fp = 0
    matched_gt = set()
    for d_box, _ in dets:
        best_i, best_score = -1, 0.0
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(d_box, g)
            s = iu if rule == "iou" else ip
            if s > best_score:
                best_score, best_i = s, gi
        thr = iou_thr if rule == "iou" else iop_thr
        if best_score >= thr and best_i not in matched_gt:
            tp += 1
            matched_gt.add(best_i)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


# ── feature builder (same as eval_full_pipeline) ──────────────────

FEAT_COLS_CACHE = None


def build_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
    feats = {}
    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        n = len(confs)
        if n == 0:
            feats.update({f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.0,
                          f"{prefix}_mean_conf": 0.0, f"{prefix}_detected": 0})
        else:
            feats.update({f"{prefix}_n_dets": n,
                          f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
                          f"{prefix}_detected": 1})
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
    g_rgb = compute_global_features(rgb_gray)
    g_ir  = compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in g_rgb.items()})
    feats.update({f"ir_{k}": v for k, v in g_ir.items()})
    for prefix, dets, gray, gw, gh in [
        ("rgb", rgb_dets, rgb_gray, rw, rh),
        ("ir",  ir_dets,  ir_gray,  iw, ih),
    ]:
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    rd, id_ = len(rgb_dets) > 0, len(ir_dets) > 0
    feats["both_detect"]     = int(rd and id_)
    feats["neither_detect"]  = int(not rd and not id_)
    feats["rgb_only_detect"] = int(rd and not id_)
    feats["ir_only_detect"]  = int(not rd and id_)
    return feats


# ── image path resolution ─────────────────────────────────────────

def img_from_label(lbl_path: Path, img_dir: Path) -> Path | None:
    """Derive image path by mirroring GT label stem into images dir."""
    stem = Path(lbl_path).stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── config dispatch ───────────────────────────────────────────────

def apply_configs(rgb_dets, ir_dets, rgb_flt, ir_flt, clf_label_raw, clf_label_flt):
    """Return dict: config_name -> (kept_rgb, kept_ir)."""
    out = {
        "ir_only":   ([], ir_dets),
        "rgb_only":  (rgb_dets, []),
        "ir_filter": ([], ir_flt),
        "rgb_filter":(rgb_flt, []),
    }
    # classifier raw
    kr = rgb_dets if clf_label_raw in (1, 3) else []
    ki = ir_dets  if clf_label_raw in (2, 3) else []
    out["classifier"] = (kr, ki)
    # classifier + filter
    kr = rgb_flt if clf_label_flt in (1, 3) else []
    ki = ir_flt  if clf_label_flt in (2, 3) else []
    out["classifier_filter"] = (kr, ki)
    return out


# ── main eval loop ────────────────────────────────────────────────

def evaluate(ds_name: str, raw_json: Path, rgb_img_dir: Path, ir_img_dir: Path,
             clf_bundle, patch_rgb: PatchVerifier, patch_ir: PatchVerifier,
             limit: int | None = None, stride: int = 1):
    global FEAT_COLS_CACHE
    FEAT_COLS_CACHE = clf_bundle["features"]
    clf_model = clf_bundle["model"]

    out_dir = OUT_ROOT / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path   = out_dir / "progress.jsonl"
    ppath       = out_dir / "patch_probs.json"
    perdet_path = out_dir / "per_det.jsonl"  # fallback for offline sweeps

    print(f"[{ds_name}] loading cached detections from {raw_json.name}...")
    raw = json.loads(raw_json.read_text())
    keys = sorted(raw.keys())
    if stride > 1:
        keys = keys[::stride]
    if limit:
        keys = keys[:limit]
    print(f"[{ds_name}] {len(keys):,} frame pairs")

    # resume: load per-config counters + processed set (iou + iop rules)
    RULES = ("iou", "iop")
    counters = {rule: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CONFIG_NAMES}
                for rule in RULES}
    fp_by_cat = {rule: {c: {cat: 0 for cat in (*SVAN_CATS, "OTHER")}
                         for c in CONFIG_NAMES}
                 for rule in RULES}
    done: set = set()
    if prog_path.exists():
        for ln in prog_path.read_text().splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            done.add(rec["key"])
            cat = svan_category(rec["key"])
            for rule in RULES:
                key_name = f"inc_{rule}"
                if key_name not in rec:
                    continue
                for c, d in rec[key_name].items():
                    counters[rule][c]["tp"] += d[0]
                    counters[rule][c]["fp"] += d[1]
                    counters[rule][c]["fn"] += d[2]
                    fp_by_cat[rule][c][cat] += d[1]
        print(f"[{ds_name}] resumed: {len(done):,} frames already processed")

    patch_cache: dict = {}
    if ppath.exists():
        try:
            patch_cache = json.loads(ppath.read_text())
        except Exception:
            patch_cache = {}

    t0 = time.time()
    n_done_session = 0
    buffered = []           # progress lines to flush periodically
    perdet_buffered = []    # per-det records for offline sweeps

    for idx, key in enumerate(keys):
        if key in done:
            continue
        entry = raw[key]
        rgb_dets_all = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["rgb_dets"]]
        ir_dets_all  = [((d[0], d[1], d[2], d[3]), d[4]) for d in entry["ir_dets"]]
        rgb_dets_raw = [d for d in rgb_dets_all if d[1] >= RGB_CONF]
        ir_dets_raw  = [d for d in ir_dets_all  if d[1] >= IR_CONF]
        rw, rh = entry["rgb_w"], entry["rgb_h"]
        iw, ih = entry["ir_w"],  entry["ir_h"]
        rgb_gt = read_yolo_labels(Path(entry["rgb_lbl"]), rw, rh)
        ir_gt  = read_yolo_labels(Path(entry["ir_lbl"]),  iw, ih)

        # resolve image paths (need for filter + classifier scene feats)
        rgb_path = img_from_label(Path(entry["rgb_lbl"]), rgb_img_dir)
        ir_path  = img_from_label(Path(entry["ir_lbl"]),  ir_img_dir)
        if rgb_path is None or ir_path is None:
            # skip silently if no image on disk
            continue
        rgb_img = cv2.imread(str(rgb_path))
        ir_img  = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        # filter probs on ALL dets (cache) — lets offline sweeps drop conf later
        cached = patch_cache.get(key)
        if cached is None:
            rgb_probs_all = patch_rgb.predict_boxes(
                rgb_img, [d[0] for d in rgb_dets_all]
            ).tolist() if rgb_dets_all else []
            ir_probs_all = patch_ir.predict_boxes(
                ir_img, [d[0] for d in ir_dets_all]
            ).tolist() if ir_dets_all else []
            patch_cache[key] = {"rgb": rgb_probs_all, "ir": ir_probs_all}
        else:
            rgb_probs_all = cached["rgb"]
            ir_probs_all  = cached["ir"]

        # slice probs down to thresholded raw dets (preserve order)
        rgb_probs = [p for d, p in zip(rgb_dets_all, rgb_probs_all) if d[1] >= RGB_CONF]
        ir_probs  = [p for d, p in zip(ir_dets_all,  ir_probs_all)  if d[1] >= IR_CONF]
        rgb_flt = [d for d, p in zip(rgb_dets_raw, rgb_probs) if p < PATCH_THR]
        ir_flt  = [d for d, p in zip(ir_dets_raw,  ir_probs)  if p < PATCH_THR]

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray  = cv2.cvtColor(ir_img,  cv2.COLOR_BGR2GRAY)

        feats_raw = build_features(rgb_dets_raw, ir_dets_raw, rgb_gray, ir_gray)
        feats_flt = build_features(rgb_flt,      ir_flt,      rgb_gray, ir_gray)
        x_raw = np.array([[feats_raw.get(c, 0) for c in FEAT_COLS_CACHE]], dtype=np.float32)
        x_flt = np.array([[feats_flt.get(c, 0) for c in FEAT_COLS_CACHE]], dtype=np.float32)
        lbl_raw = int(clf_model.predict(x_raw)[0])
        lbl_flt = int(clf_model.predict(x_flt)[0])

        configs = apply_configs(rgb_dets_raw, ir_dets_raw, rgb_flt, ir_flt,
                                lbl_raw, lbl_flt)

        # Scope GT per config: single-modality configs score only against their
        # own modality's GT; classifier configs see both.
        GT_SCOPE = {
            "ir_only":            ("ir",),
            "rgb_only":           ("rgb",),
            "ir_filter":          ("ir",),
            "rgb_filter":         ("rgb",),
            "classifier":         ("rgb", "ir"),
            "classifier_filter":  ("rgb", "ir"),
        }
        inc_iou = {}; inc_iop = {}
        cat = svan_category(key)
        for c_name, (kr, ki) in configs.items():
            scope = GT_SCOPE[c_name]
            for rule, inc_dict in (("iou", inc_iou), ("iop", inc_iop)):
                tp = fp = fn = 0
                if "rgb" in scope:
                    t, f, n = score_dets(kr, rgb_gt, rule=rule,
                                         iou_thr=IOU_MATCH, iop_thr=IOP_MATCH)
                    tp += t; fp += f; fn += n
                else:
                    t, f, _ = score_dets(kr, [], rule=rule)
                    tp += t; fp += f
                if "ir" in scope:
                    t, f, n = score_dets(ki, ir_gt, rule=rule,
                                         iou_thr=IOU_MATCH, iop_thr=IOP_MATCH)
                    tp += t; fp += f; fn += n
                else:
                    t, f, _ = score_dets(ki, [], rule=rule)
                    tp += t; fp += f
                counters[rule][c_name]["tp"] += tp
                counters[rule][c_name]["fp"] += fp
                counters[rule][c_name]["fn"] += fn
                fp_by_cat[rule][c_name][cat] += fp
                inc_dict[c_name] = [tp, fp, fn]

        buffered.append(json.dumps({"key": key,
                                    "inc_iou": inc_iou,
                                    "inc_iop": inc_iop}))

        # per-det dump for offline threshold sweeps: conf, filter_prob,
        # matched_iou, matched_iop
        def _score_match(dets, gts):
            out_iou = []; out_iop = []
            used_u = set(); used_p = set()
            for (db, _c) in dets:
                best_iu = best_ip = 0.0; bi_u = bi_p = -1
                for gi, g in enumerate(gts):
                    iu, ip = iou_iop(db, g)
                    if iu > best_iu: best_iu, bi_u = iu, gi
                    if ip > best_ip: best_ip, bi_p = ip, gi
                m_u = int(best_iu >= IOU_MATCH and bi_u not in used_u)
                m_p = int(best_ip >= IOP_MATCH and bi_p not in used_p)
                if m_u: used_u.add(bi_u)
                if m_p: used_p.add(bi_p)
                out_iou.append(m_u); out_iop.append(m_p)
            return out_iou, out_iop
        rgb_m_iou, rgb_m_iop = _score_match(rgb_dets_all, rgb_gt)
        ir_m_iou,  ir_m_iop  = _score_match(ir_dets_all,  ir_gt)
        perdet_rec = {
            "key": key,
            "rgb": [[round(d[1], 4), round(p, 4), mu, mp]
                    for d, p, mu, mp in zip(rgb_dets_all, rgb_probs_all,
                                             rgb_m_iou, rgb_m_iop)],
            "ir":  [[round(d[1], 4), round(p, 4), mu, mp]
                    for d, p, mu, mp in zip(ir_dets_all, ir_probs_all,
                                             ir_m_iou, ir_m_iop)],
            "rgb_n_gt": len(rgb_gt),
            "ir_n_gt":  len(ir_gt),
            "clf_raw":  lbl_raw,
            "clf_flt":  lbl_flt,
        }
        perdet_buffered.append(json.dumps(perdet_rec))
        n_done_session += 1

        if n_done_session % CKPT_EVERY == 0:
            with prog_path.open("a") as fh:
                fh.write("\n".join(buffered) + "\n")
            with perdet_path.open("a") as fh:
                fh.write("\n".join(perdet_buffered) + "\n")
            buffered.clear()
            perdet_buffered.clear()
            ppath.write_text(json.dumps(patch_cache))
            elapsed = time.time() - t0
            fps = n_done_session / elapsed
            remaining = (len(keys) - len(done) - n_done_session) / max(fps, 1e-6)
            print(f"[{ds_name}] {len(done) + n_done_session:>6,}/{len(keys):,}  "
                  f"{fps:.1f} fps  ETA {remaining/60:.1f} min")

    # flush tail
    if buffered:
        with prog_path.open("a") as fh:
            fh.write("\n".join(buffered) + "\n")
    if perdet_buffered:
        with perdet_path.open("a") as fh:
            fh.write("\n".join(perdet_buffered) + "\n")
    ppath.write_text(json.dumps(patch_cache))

    # finalise metrics — per matching rule (IoU + IoP)
    rule_rows = {}
    cats = [*SVAN_CATS, "OTHER"]
    for rule in RULES:
        rows = []; confusion = {}
        for c_name in CONFIG_NAMES:
            tp = counters[rule][c_name]["tp"]
            fp = counters[rule][c_name]["fp"]
            fn = counters[rule][c_name]["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append({"config": c_name, "TP": tp, "FP": fp, "FN": fn,
                         "precision": round(p, 4), "recall": round(r, 4),
                         "f1": round(f1, 4)})
            confusion[c_name] = {"TP": tp, "FP": fp, "FN": fn}
        rule_rows[rule] = rows
        with (out_dir / f"metrics_{rule}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", "TP", "FP", "FN",
                                                "precision", "recall", "f1"])
            w.writeheader(); w.writerows(rows)
        (out_dir / f"confusion_{rule}.json").write_text(
            json.dumps(confusion, indent=2))
        with (out_dir / f"fp_by_category_{rule}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", *cats, "total"])
            w.writeheader()
            for c_name in CONFIG_NAMES:
                row = {"config": c_name}; total = 0
                for cat in cats:
                    row[cat] = fp_by_cat[rule][c_name][cat]; total += row[cat]
                row["total"] = total
                w.writerow(row)

        print(f"\n[{ds_name}] RESULTS ({rule.upper()} match)")
        print(f"  {'config':<20s} {'TP':>8s} {'FP':>8s} {'FN':>8s} "
              f"{'prec':>7s} {'rec':>7s} {'f1':>7s}")
        for row in rows:
            print(f"  {row['config']:<20s} {row['TP']:>8d} {row['FP']:>8d} "
                  f"{row['FN']:>8d} {row['precision']:>7.4f} "
                  f"{row['recall']:>7.4f} {row['f1']:>7.4f}")
        print(f"[{ds_name}] FP by category ({rule.upper()})")
        print(f"  {'config':<20s} " + " ".join(f"{c:>10s}" for c in cats))
        for c_name in CONFIG_NAMES:
            vals = " ".join(f"{fp_by_cat[rule][c_name][c]:>10d}" for c in cats)
            print(f"  {c_name:<20s} {vals}")

    # plots per rule
    for rule, rows in rule_rows.items():
        plot_metrics(rows, out_dir, f"{ds_name} [{rule.upper()}]",
                     suffix=f"_{rule}")
        plot_confusion(rows, out_dir, f"{ds_name} [{rule.upper()}]",
                       suffix=f"_{rule}")
    plot_pr_curves(perdet_path, out_dir, ds_name)


def plot_metrics(rows, out_dir, ds_name, suffix=""):
    names = [r["config"] for r in rows]
    prec  = [r["precision"] for r in rows]
    rec   = [r["recall"]    for r in rows]
    f1s   = [r["f1"]        for r in rows]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#3498db")
    ax.bar(x,     rec,  w, label="Recall",    color="#e74c3c")
    ax.bar(x + w, f1s,  w, label="F1",        color="#2ecc71")
    for i, vs in enumerate([prec, rec, f1s]):
        for j, v in enumerate(vs):
            ax.text(x[j] + (i - 1) * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"{ds_name} — per-detection metrics"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"metrics_bars{suffix}.png", dpi=140); plt.close(fig)


def plot_confusion(rows, out_dir, ds_name, suffix=""):
    n = len(rows); cols = 3; r_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(r_, cols, figsize=(4 * cols, 3.5 * r_))
    axes = axes.flatten()
    for i, row in enumerate(rows):
        ax = axes[i]
        tp, fp, fn = row["TP"], row["FP"], row["FN"]
        # 2x2 grid: [TP FN; FP  -]
        m = np.array([[tp, fn], [fp, 0]], dtype=float)
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred+", "pred-"], fontsize=8)
        ax.set_yticklabels(["GT+", "GT-"], fontsize=8)
        labels = [[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", "—"]]
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, labels[ii][jj], ha="center", va="center",
                        fontsize=9,
                        color="white" if m[ii, jj] > m.max() * 0.5 else "black")
        ax.set_title(f"{row['config']}\nP={row['precision']:.3f} "
                     f"R={row['recall']:.3f} F1={row['f1']:.3f}", fontsize=9)
    for j in range(len(rows), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{ds_name} — per-detection confusion (no TN)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_matrices{suffix}.png", dpi=140); plt.close(fig)


def plot_pr_curves(perdet_path, out_dir, ds_name):
    """PR curves by sweeping conf threshold on cached per-det records.
    Only meaningful for raw-conf configs (single-mod ± filter). Classifier
    configs output single points."""
    if not perdet_path.exists():
        return
    rgb_records = []  # (conf, filter_p, m_iou, m_iop)
    ir_records  = []
    n_rgb_gt = 0; n_ir_gt = 0
    for ln in perdet_path.read_text().splitlines():
        if not ln.strip(): continue
        r = json.loads(ln)
        n_rgb_gt += r["rgb_n_gt"]; n_ir_gt += r["ir_n_gt"]
        rgb_records.extend(r["rgb"]); ir_records.extend(r["ir"])

    def pr_sweep(records, total_gt, match_idx, filter_mask=None):
        recs = [t for t in records if (filter_mask is None or filter_mask(t[1]))]
        recs.sort(key=lambda t: -t[0])
        tp = 0; fp = 0
        precs = []; recs_ = []; threshs = []
        for t in recs:
            if t[match_idx]: tp += 1
            else: fp += 1
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / total_gt if total_gt > 0 else 0.0
            precs.append(pr); recs_.append(rc); threshs.append(t[0])
        return np.array(precs), np.array(recs_), np.array(threshs)

    for rule, m_idx in (("iou", 2), ("iop", 3)):
        fig, ax = plt.subplots(figsize=(8, 6))
        specs = [
            ("rgb_only",   rgb_records, n_rgb_gt, None,                      "#3498db"),
            ("ir_only",    ir_records,  n_ir_gt,  None,                      "#e67e22"),
            ("rgb_filter", rgb_records, n_rgb_gt, (lambda p: p < PATCH_THR), "#2980b9"),
            ("ir_filter",  ir_records,  n_ir_gt,  (lambda p: p < PATCH_THR), "#d35400"),
        ]
        for name, recs, gt, mask, colour in specs:
            if gt == 0 or not recs: continue
            pr, rc, th = pr_sweep(recs, gt, m_idx, mask)
            ax.plot(rc, pr, label=name, color=colour, linewidth=1.8)
            op = 0.25 if "rgb" in name else 0.40
            if len(th):
                idx = int(np.argmin(np.abs(th - op)))
                ax.scatter([rc[idx]], [pr[idx]], color=colour, s=40, zorder=5,
                           edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="lower left")
        ax.set_title(f"{ds_name} — PR curves ({rule.upper()} match, "
                     f"dots = op thresh)")
        plt.tight_layout()
        fig.savefig(out_dir / f"pr_curves_{rule}.png", dpi=140); plt.close(fig)


def main():
    global IOU_MATCH, IOP_MATCH, RGB_CONF, IR_CONF, PATCH_THR
    global RAW_ANTIUAV, RAW_SVAN, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["antiuav", "svanstrom", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth sorted key (covers all categories evenly)")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate plots from existing metrics.csv/per_det.jsonl")
    ap.add_argument("--iou",       type=float, default=IOU_MATCH)
    ap.add_argument("--iop",       type=float, default=IOP_MATCH)
    ap.add_argument("--rgb-conf",  type=float, default=RGB_CONF)
    ap.add_argument("--ir-conf",   type=float, default=IR_CONF)
    ap.add_argument("--patch-thr", type=float, default=PATCH_THR)
    ap.add_argument("--cache-tag", type=str, default="",
                    help="cache suffix, e.g. 'v3more' -> raw_detections_v3more.json")
    ap.add_argument("--clf-path",  type=str, default=str(CLF_PATH),
                    help="path to fusion classifier joblib")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix on OUT_ROOT, e.g. '_v3more' -> eval_six_configs_v3more/")
    args = ap.parse_args()

    sfx = f"_{args.cache_tag}" if args.cache_tag else ""
    RAW_ANTIUAV = RAW_ANTIUAV_BASE.with_name(RAW_ANTIUAV_BASE.name + sfx + ".json")
    RAW_SVAN    = RAW_SVAN_BASE.with_name(RAW_SVAN_BASE.name + sfx + ".json")
    if args.out_suffix:
        OUT_ROOT = OUT_ROOT.with_name(OUT_ROOT.name + args.out_suffix)

    if args.plot_only:
        for ds in (["antiuav", "svanstrom"] if args.dataset == "both"
                   else [args.dataset]):
            d = OUT_ROOT / ds
            for rule in ("iou", "iop"):
                csv_p = d / f"metrics_{rule}.csv"
                if not csv_p.exists():
                    continue
                rows = list(csv.DictReader(csv_p.open()))
                for r in rows:
                    for k in ("TP", "FP", "FN"): r[k] = int(r[k])
                    for k in ("precision", "recall", "f1"): r[k] = float(r[k])
                plot_metrics(rows, d, f"{ds} [{rule.upper()}]", suffix=f"_{rule}")
                plot_confusion(rows, d, f"{ds} [{rule.upper()}]", suffix=f"_{rule}")
            plot_pr_curves(d / "per_det.jsonl", d, ds)
            print(f"[{ds}] plots regenerated → {d}")
        return

    IOU_MATCH = args.iou
    IOP_MATCH = args.iop
    RGB_CONF  = args.rgb_conf
    IR_CONF   = args.ir_conf
    PATCH_THR = args.patch_thr
    print(f"[cfg] iou={IOU_MATCH} iop={IOP_MATCH} "
          f"rgb_conf={RGB_CONF} ir_conf={IR_CONF} patch_thr={PATCH_THR}")

    print(f"[paths] antiuav_cache={RAW_ANTIUAV.name}  svan_cache={RAW_SVAN.name}")
    print(f"[paths] clf={Path(args.clf_path).name}  out_root={OUT_ROOT.name}")
    print("loading classifier + filters...")
    clf = joblib.load(args.clf_path)
    patch_rgb = PatchVerifier(PATCH_RGB)
    patch_ir  = PatchVerifier(PATCH_IR)
    limit = args.limit or None

    if args.dataset in ("antiuav", "both"):
        evaluate("antiuav", RAW_ANTIUAV, ANTIUAV_RGB_IMG, ANTIUAV_IR_IMG,
                 clf, patch_rgb, patch_ir, limit, args.stride)
    if args.dataset in ("svanstrom", "both"):
        evaluate("svanstrom", RAW_SVAN, SVAN_RGB_IMG, SVAN_IR_IMG,
                 clf, patch_rgb, patch_ir, limit, args.stride)


if __name__ == "__main__":
    main()
