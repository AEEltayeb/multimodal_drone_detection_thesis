"""
thesis_eval/pipeline_eval_unified.py — ZERO-GPU replay of the unified detection cache -> A/B/C/D tables.

Reads thesis_eval/cache/<surface>.pkl (written by pipeline_cache_unified.py: detect ONCE with ft4+v3b,
store per-detection {xyxy,conf,517-D verifier feat,patch P(confuser)} + per-frame trust rows {f8_all,
f32_all} + per-modality GT) and replays the WHOLE thesis evaluation with no detector forward pass:

  PART A  per-model in-domain, BARE detectors, per modality vs its OWN GT (the production ft4/v3b rows;
          the baseline/retrained_v2/V2/V5 versions need a separate detector sweep — NOT in this cache).
  PART B  full-pipeline ablation on PAIRED surfaces (antiuav, svanstrom): bare -> +classifier ->
          +verifier -> clf->filter (production cascade) -> filter->clf. Scored with the SANCTIONED
          per-modality rule metrics.score_trust_aware (trust-RGB excludes IR GT; reject-both -> both FN;
          NEVER a union). Temporal/segment grain comes from the SEPARATE real-video run, not this replay.
  PART S4 verifier-only ablation on SOLO drone surfaces (rgb_dataset_test, ir_dset_final, selcom_val,
          svanstrom_gray): bare -> +mlp -> +patch vs own GT (single modality; no routing).
  PART C  confuser FP-reduction (rgb/ir/gray confusers, no GT): bare -> +filter(mlp) -> +filter(patch).
  PART D  grayscale FINDING (good-only): IR-on-grayscale + aligned-GRAY-scaler filter, CLASSIFIER
          BYPASSED, vs RGB bare. Backed by a bootstrap CI.

Per-modality verifier scalers (one network, two scalers): RGB -> mlp_v5 @0.25; thermal IR -> mlp_aligned
@0.05; GRAYSCALE -> mlp_aligned_gray @0.25 (using the thermal scaler on grayscale under-cuts ~2x).
Classifier + verifier predicts are BATCHED per surface (the per-row XGBoost loop was ~45 min; now ~min).
Bootstrap CIs (frame-level resample, 1000 iters) on every P/R/F1 cell and confuser fire rate.

TIER-1 STRIDED = FINAL thesis numbers (decision 2026-06-10): ~4k/surface even-strided, all frames where
smaller; n (and n_source) printed everywhere.

  py -u thesis_eval/pipeline_eval_unified.py                       # all landed surfaces
  py -u thesis_eval/pipeline_eval_unified.py --only svanstrom      # one surface
"""
from __future__ import annotations
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
for _sub in ("eval", "classifier"):
    sys.path.insert(0, str(REPO / _sub))

import joblib                                                                    # noqa: E402
from metrics import score_detections, compute_prf, score_trust_aware            # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                                       # noqa: E402
from compare_routing_pipeline import f8_vec, F8 as F8_ORDER, CONF                # noqa: E402

# ── artifacts (the SHIPPED production stack) ─────────────────────────────────────────────────────────
ROBUST8_JBL  = REPO / "models/routers/robust8.joblib"        # trust router (8 feat, tau=0.20)
ROBUST6_JBL  = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"  # 6-feat base (argmax)
SA32_JBL     = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"  # sad 32-feat router
NR_DROP_JBL  = REPO / "models/routers/robust8_noreject_drop/model.joblib"   # no-reject 3-class (reject rows dropped)
NR_BOTH_JBL  = REPO / "models/routers/robust8_noreject_both/model.joblib"   # no-reject 3-class (reject -> both)
# Weights/thresholds are env-overridable; defaults = the PRODUCTION stack the frozen thesis JSONs used
# (RGB mlp_v5_v4, IR thermal-only), so the committed numbers reproduce out of the box. Verified 2026-06-20:
# ir_confusers filt_mlp 0.0278 reproduces with mlp_aligned_thermalonly, not the old mlp_aligned (0.237).
# Filter A/B harness sets THESIS_* to repoint at candidate filters WITHOUT clobbering the committed stack.
MLP_V5       = Path(os.environ.get("THESIS_MLP_V5",       REPO / "models/verifiers/rgb_v5/mlp_v5_v4.pt"))   # RGB verifier (production v4)
ALIGNED      = Path(os.environ.get("THESIS_ALIGNED",      REPO / "models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt"))  # IR verifier (production thermal-only)
ALIGNED_GRAY = Path(os.environ.get("THESIS_ALIGNED_GRAY", REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt"))   # same net, GRAYSCALE scaler
CACHE_DIR    = REPO / "thesis_eval" / "cache"
OUT_DIR      = REPO / "thesis_eval" / "results"

RGB_THR_MLP  = float(os.environ.get("THESIS_RGB_THR_MLP",  0.25))
IR_THR_MLP   = float(os.environ.get("THESIS_IR_THR_MLP",   0.05))
GRAY_THR_MLP = float(os.environ.get("THESIS_GRAY_THR_MLP", 0.25))
TAU = 0.20                                                                       # shipped thresholds
REJECT_FLOOR = 0.80   # round-8 reject-probability-floor ablation: ONE global x* applied to every surface
DUMMY_G, DUMMY_WH = np.zeros((64, 64), np.uint8), (64, 64)                       # f8 recompute ignores pixels
BOOT_ITERS, BOOT_SEED = 1000, 0

# per-surface-kind verifier slot: (det slot, verifier key, threshold)
KIND_VERIFIER = {"rgb": ("rgb", "mlp_v5", RGB_THR_MLP),
                 "ir": ("ir", "aligned", IR_THR_MLP),
                 "gray": ("ir", "aligned_gray", GRAY_THR_MLP),   # gray dets live in the ir slot
                 "rawrgb": ("ir", "aligned", IR_THR_MLP)}        # diagnostic control (3-way middle leg)

GRAY_SWEEP_THRS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]           # aligned_gray operating-point sweep

# display names: "aligned"/"aligned_gray" are ONE network (mlp_v5_ir_aligned) with two per-modality
# input scalers — NOT separate models (the dedicated mlp_v5_gray is superseded and unused here)
VERIFIER_LABELS = {"mlp_v5": "mlp_v5",
                   "aligned": "mlp_v5_ir_aligned (thermal scaler)",
                   "aligned_gray": "mlp_v5_ir_aligned (grayscale scaler)"}


def gate(label, modality):
    """Trust-label gating of a single live modality: 0 reject, 1 trust_rgb, 2 trust_ir, 3 both."""
    return label == 3 or (label == 1 and modality == "rgb") or (label == 2 and modality == "ir")


# ── classifier + verifier loading (CPU, zero-GPU) ───────────────────────────────────────────────────
def load_classifiers():
    clfs = {}
    if ROBUST8_JBL.exists():
        r8 = joblib.load(ROBUST8_JBL); r8["feat_key"] = "f8"; r8["tau"] = TAU
        clfs["robust8"] = r8
    if ROBUST6_JBL.exists():
        try:
            raw = joblib.load(ROBUST6_JBL)
            clfs["robust6"] = ({"model": raw["model"], "features": raw.get("features"), "feat_key": "f8", "tau": None}
                               if isinstance(raw, dict) else {"model": raw, "features": None, "feat_key": "f8", "tau": None})
        except Exception as e:
            print(f"  [robust6 load failed: {e}]")
    if SA32_JBL.exists():
        raw = joblib.load(SA32_JBL)
        clfs["sa32"] = ({"model": raw["model"], "features": raw.get("features"), "feat_key": "f32", "tau": None}
                        if isinstance(raw, dict) else {"model": raw, "features": None, "feat_key": "f32", "tau": None})
    # no-reject 3-class routers (dicts already carry model/features=F8/feat_key=f8/tau=None/label_map)
    for nm, pth in (("robust8_nr_drop", NR_DROP_JBL), ("robust8_nr_both", NR_BOTH_JBL)):
        if pth.exists():
            clfs[nm] = joblib.load(pth)
    r6 = clfs.get("robust6")
    if r6 and (not r6.get("features") or not set(r6["features"]) <= set(F8_ORDER)):
        print("  [robust6 dropped: features missing or not a subset of F8]"); clfs.pop("robust6")
    print(f"  classifiers: {list(clfs)}")
    return clfs


def load_verifiers(device="cpu"):
    v = {}
    for key, path in (("mlp_v5", MLP_V5), ("aligned", ALIGNED), ("aligned_gray", ALIGNED_GRAY)):
        try:
            v[key] = MLPv4Verifier(Path(path), device=device)
        except Exception as e:
            print(f"  [verifier {key} load failed: {e}]")
    print(f"  verifiers: {list(v)}")
    return v


def batch_labels(clf, F8mat, F32mat, F8, F32):
    """4-class trust labels for ALL frames in one predict (the 1-row loop was the replay bottleneck)."""
    order, mat = (F32, F32mat) if clf["feat_key"] == "f32" else (F8, F8mat)
    feats = clf.get("features")
    X = mat if feats is None else mat[:, [order.index(f) for f in feats]]
    if clf.get("tau") is not None:
        p = clf["model"].predict_proba(X)
        out = np.where(p[:, 1] >= clf["tau"], 1, p.argmax(1)).astype(int)
    else:
        out = np.asarray(clf["model"].predict(X), int)
    lm = clf.get("label_map")          # 3-class no-reject routers store {0:1,1:2,2:3} -> harness trust labels
    if lm:
        out = np.array([lm.get(int(v), int(v)) for v in out], int)
    return out


def reject_floor_labels(clf, F8mat, F8, x):
    """robust8 with a reject-probability FLOOR (round-8 ablation): the production rule, except a
    `reject` (0) is honoured only when P(reject) >= x; otherwise the frame is routed to the
    more-confident single modality (trust_rgb vs trust_ir), leaving false-positive removal to the
    per-frame filter. One global x for every surface."""
    feats = clf.get("features")
    X = F8mat if feats is None else F8mat[:, [F8.index(f) for f in feats]]
    P = clf["model"].predict_proba(X)
    classes = list(clf["model"].classes_); col = {c: i for i, c in enumerate(classes)}
    tau = clf.get("tau")
    out = np.empty(len(P), int)
    for i, row in enumerate(P):
        l = 1 if (tau is not None and 1 in col and row[col[1]] >= tau) else classes[int(np.argmax(row))]
        if l == 0 and (0 not in col or row[col[0]] < x):
            p1 = row[col[1]] if 1 in col else -1.0
            p2 = row[col[2]] if 2 in col else -1.0
            l = 1 if p1 >= p2 else 2
        out[i] = int(l)
    return out


def batch_probs(frames, slot_key, verifier):
    """Verifier P(drone) for every detection of every frame in ONE forward pass; split per frame."""
    counts = [len(fr[slot_key]["confs"]) for fr in frames]
    feats = [fr[slot_key]["feats"] for fr in frames if len(fr[slot_key]["confs"])]
    if not feats:
        return [np.zeros(0, np.float32) for _ in frames]
    probs = verifier.predict_drone_probs(np.concatenate(feats).astype(np.float32))
    out, o = [], 0
    for c in counts:
        out.append(np.asarray(probs[o:o + c], np.float32)); o += c
    return out


# ── per-detection helpers ───────────────────────────────────────────────────────────────────────────
def dets2(slot, mask=None):
    b, c = slot["boxes"], slot["confs"]
    idx = range(len(c)) if mask is None else np.where(mask)[0]
    return [((float(b[i][0]), float(b[i][1]), float(b[i][2]), float(b[i][3])), float(c[i])) for i in idx]


def dets5(slot, mask=None):
    b, c = slot["boxes"], slot["confs"]
    idx = range(len(c)) if mask is None else np.where(mask)[0]
    return [(float(b[i][0]), float(b[i][1]), float(b[i][2]), float(b[i][3]), float(c[i])) for i in idx]


def gts(arr):
    return [(float(g[0]), float(g[1]), float(g[2]), float(g[3])) for g in arr]


def patch_mask(slot, patch_thr):
    """fail-open: KEEP unless confidently a confuser (P(confuser) >= patch_thr)."""
    p = np.asarray(slot["patch"], np.float32)
    return p < patch_thr if len(p) else np.zeros(0, bool)


def recompute_f8(rgb5, ir5, is_gray):
    return np.asarray(f8_vec(rgb5, ir5, DUMMY_G, DUMMY_G, DUMMY_WH, DUMMY_WH, is_gray, 0, "replay"), np.float32)


def _sum_ta(s):
    return (sum(b["tp"] for b in s.values()), sum(b["fp"] for b in s.values()), sum(b["fn"] for b in s.values()))


# ── bootstrap (frame-level resample) ─────────────────────────────────────────────────────────────────
def boot_f1_ci(per_frame, iters=BOOT_ITERS, seed=BOOT_SEED):
    """per_frame: (n,3) tp/fp/fn ints. Returns (f1_lo, f1_hi) 95% percentile CI."""
    a = np.asarray(per_frame, np.int64)
    if not len(a):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (iters, len(a)))
    tp, fp, fn = a[idx, 0].sum(1), a[idx, 1].sum(1), a[idx, 2].sum(1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def boot_rate_ci(flags, iters=BOOT_ITERS, seed=BOOT_SEED):
    """flags: (n,) 0/1 fired-per-frame. 95% CI of the fire rate."""
    a = np.asarray(flags, np.int64)
    if not len(a):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (iters, len(a)))
    r = a[idx].mean(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


class Cells:
    """Accumulates per-frame (tp,fp,fn) per cell; emits P/R/F1 + bootstrap CI."""
    def __init__(self):
        self.d = {}

    def add(self, cell, tp, fp, fn):
        self.d.setdefault(cell, []).append((tp, fp, fn))

    def report(self):
        out = {}
        for cell, rows in self.d.items():
            a = np.asarray(rows, np.int64)
            prf = compute_prf(int(a[:, 0].sum()), int(a[:, 1].sum()), int(a[:, 2].sum()))
            ci = boot_f1_ci(a)
            if ci:
                prf["f1_ci"] = [round(ci[0], 4), round(ci[1], 4)]
            out[cell] = prf
        return out


# ── PART A: bare detectors, per modality vs own GT ───────────────────────────────────────────────────
SLOTS = {"paired": [("rgb", "gt_rgb"), ("ir", "gt_ir")], "rgb": [("rgb", "gt_rgb")],
         "ir": [("ir", "gt_ir")], "gray": [("ir", "gt_ir")],   # gray/rawrgb dets live in the ir slot
         "rawrgb": [("ir", "gt_ir")],
         "grayrgb_paired": [("rgb", "gt_rgb"), ("ir", "gt_ir")]}  # ft4 + gray-fed v3b (video regime)


def part_a(meta, frames):
    if not meta["has_drones"]:
        return {}
    rule = meta["rule"]; out = {}
    for slot_key, gt_key in SLOTS[meta["kind"]]:
        cells, ngt = Cells(), 0
        for fr in frames:
            gt = gts(fr["rgb_gt"] if gt_key == "gt_rgb" else fr["ir_gt"])
            t, f, n = score_detections(dets2(fr[slot_key]), gt, rule=rule)
            cells.add("bare", t, f, n); ngt += len(gt)
        label = f"{'ft4' if slot_key == 'rgb' else 'v3b'}/{slot_key}"
        out[label] = {**cells.report()["bare"], "n_gt": ngt}
    return out


# ── PART B: paired full-pipeline ablation (per-modality scoring, NO union) ────────────────────────────
def part_b(meta, frames, clfs, verifs, patch_thr):
    rule, is_gray = meta["rule"], meta["is_grayscale"]
    F8, F32 = meta["F8"], meta["F32"]
    cells = Cells()

    F8mat = np.stack([fr["f8_all"] for fr in frames])
    F32mat = np.stack([fr["f32_all"] for fr in frames])
    labels = {c: batch_labels(clf, F8mat, F32mat, F8, F32) for c, clf in clfs.items()}
    rf_labels = reject_floor_labels(clfs["robust8"], F8mat, F8, REJECT_FLOOR) if "robust8" in clfs else None
    rf_cell = f"clf->filt[robust8,rej>={REJECT_FLOOR}]"
    # f8-feature routers (robust8, robust6) are cascade-capable: their features are recomputable
    # from the filtered detections, so they get the filter->classifier ordering. sa32 (f32 scene
    # features) is excluded — those aren't recomputed post-filter here.
    f8_clfs = [c for c, cl in clfs.items() if cl.get("feat_key") == "f8"]
    # the IR slot's verifier follows what the channel was FED: thermal -> aligned, gray -> aligned_gray
    ir_vkey, ir_thr = ("aligned_gray", GRAY_THR_MLP) if is_gray else ("aligned", IR_THR_MLP)
    rgb_probs = batch_probs(frames, "rgb", verifs["mlp_v5"]) if "mlp_v5" in verifs else None
    ir_probs = batch_probs(frames, "ir", verifs[ir_vkey]) if ir_vkey in verifs else None

    # pass 1: per-frame masks/survivors + accumulate every cell except filt->clf (needs batched re-predict)
    surv, parity_done = [], False
    for i, fr in enumerate(frames):
        rgb, ir = fr["rgb"], fr["ir"]
        rgb_g, ir_g = gts(fr["rgb_gt"]), gts(fr["ir_gt"])
        rm = (rgb_probs[i] >= RGB_THR_MLP) if rgb_probs is not None else np.zeros(len(rgb["confs"]), bool)
        im = (ir_probs[i] >= ir_thr) if ir_probs is not None else np.zeros(len(ir["confs"]), bool)
        rgb_all, ir_all = dets2(rgb), dets2(ir)
        rgb_flt, ir_flt = dets2(rgb, rm), dets2(ir, im)

        def TA(label, rd, idd):
            return score_trust_aware(label, rd, idd, rgb_g, ir_g, 1920, 1080, 1920, 1080,
                                     is_paired=True, rule=rule)
        cells.add("bare",       *_sum_ta(TA(3, rgb_all, ir_all)))
        cells.add("filt_mlp",   *_sum_ta(TA(3, rgb_flt, ir_flt)))
        cells.add("filt_mlp_rgb", *_sum_ta(TA(3, rgb_flt, ir_all)))   # RGB filtered, IR raw (isolate RGB filter)
        cells.add("filt_mlp_ir",  *_sum_ta(TA(3, rgb_all, ir_flt)))   # IR filtered, RGB raw (isolate IR filter)
        rp, ip = patch_mask(rgb, patch_thr), patch_mask(ir, patch_thr)
        cells.add("filt_patch", *_sum_ta(TA(3, dets2(rgb, rp), dets2(ir, ip))))
        for cname in clfs:
            L = int(labels[cname][i])
            cells.add(f"clf[{cname}]",       *_sum_ta(TA(L, rgb_all, ir_all)))
            cells.add(f"clf->filt[{cname}]", *_sum_ta(TA(L, rgb_flt, ir_flt)))
        if rf_labels is not None:
            cells.add(rf_cell, *_sum_ta(TA(int(rf_labels[i]), rgb_flt, ir_flt)))
        if f8_clfs:
            f8f = recompute_f8(dets5(rgb, rm), dets5(ir, im), is_gray)
            if not parity_done:
                f8a = recompute_f8(dets5(rgb), dets5(ir), is_gray)
                assert np.allclose(f8a, fr["f8_all"], atol=1e-3), f"f8 parity FAIL {f8a} vs {fr['f8_all']}"
                parity_done = True
            surv.append((f8f, rgb_flt, ir_flt, rgb_g, ir_g))

    # pass 2: filter->clf[<router>] for each f8 router (robust8, robust6) with ONE batched predict
    # per router over the recomputed f8 rows
    if surv:
        F8f = np.stack([s[0] for s in surv])
        for cname in f8_clfs:
            L2 = batch_labels(clfs[cname], F8f, F32mat, F8, F32)
            for (f8f, rgb_flt, ir_flt, rgb_g, ir_g), L in zip(surv, L2):
                s = score_trust_aware(int(L), rgb_flt, ir_flt, rgb_g, ir_g, 1920, 1080, 1920, 1080,
                                      is_paired=True, rule=meta["rule"])
                cells.add(f"filt->clf[{cname}]", *_sum_ta(s))
    return cells.report()


# ── PART S4: SOLO drone surfaces — verifier ablation + classifier-gated pipeline rows ────────────────
def part_s4(meta, frames, clfs, verifs, patch_thr):
    slot_key, vkey, thr = KIND_VERIFIER[meta["kind"]]
    if vkey not in verifs:
        return {}
    rule = meta["rule"]; cells = Cells()
    probs = batch_probs(frames, slot_key, verifs[vkey])
    gt_key = "rgb_gt" if slot_key == "rgb" else "ir_gt"
    # production routing applied to the single live modality (the dead modality's features are
    # genuinely dead — reported as measured). NO clf cells for: rawrgb (diagnostic control) and
    # gray (the cache runs v3b-on-gray only — the production gray regime pairs ft4-on-RGB with
    # v3b-on-gray, so router cells here would be the wrong regime; Part D is classifier-bypassed
    # by design and the robust8 gray story cites the routing run).
    labels = {}
    if clfs and meta["kind"] not in ("rawrgb", "gray"):
        F8, F32 = meta["F8"], meta["F32"]
        F8mat = np.stack([fr["f8_all"] for fr in frames])
        F32mat = np.stack([fr["f32_all"] for fr in frames])
        labels = {c: batch_labels(clf, F8mat, F32mat, F8, F32) for c, clf in clfs.items()}
    for i, fr in enumerate(frames):
        slot, gt = fr[slot_key], gts(fr[gt_key])
        m_mlp = (probs[i] >= thr)
        cells.add("bare",       *score_detections(dets2(slot), gt, rule=rule))
        cells.add("filt_mlp",   *score_detections(dets2(slot, m_mlp), gt, rule=rule))
        cells.add("filt_patch", *score_detections(dets2(slot, patch_mask(slot, patch_thr)), gt, rule=rule))
        for cname, lab in labels.items():
            keep = gate(int(lab[i]), slot_key)
            cells.add(f"clf[{cname}]",       *score_detections(dets2(slot) if keep else [], gt, rule=rule))
            cells.add(f"clf->filt[{cname}]", *score_detections(dets2(slot, m_mlp) if keep else [], gt, rule=rule))
    out = cells.report()
    for cell in out:
        out[cell]["verifier"] = vkey
    return out


# ── GRAY operating-point sweep: aligned_gray threshold grid from cached probs (zero GPU) ─────────────
def part_gray_sweep(meta, frames, verifs):
    if meta["kind"] != "gray" or "aligned_gray" not in verifs:
        return {}
    rule = meta["rule"]
    probs = batch_probs(frames, "ir", verifs["aligned_gray"])
    out = {}
    for thr in GRAY_SWEEP_THRS:
        if meta["has_drones"]:
            tp = fp = fn = 0
            for i, fr in enumerate(frames):
                t, f, n = score_detections(dets2(fr["ir"], probs[i] >= thr), gts(fr["ir_gt"]), rule=rule)
                tp += t; fp += f; fn += n
            out[str(thr)] = compute_prf(tp, fp, fn)
        else:
            k = sum(int((probs[i] >= thr).sum()) for i in range(len(frames)))
            fired = sum(int((probs[i] >= thr).any()) for i in range(len(frames)))
            out[str(thr)] = {"FP": k, "fire_rate": round(fired / max(meta["n"], 1), 4)}
    return out


# ── PART C: confuser FP-reduction (no GT -> every surviving det is a false positive) ──────────────────
def part_c(meta, frames, clfs, verifs, patch_thr):
    slot_key, vkey, thr = KIND_VERIFIER[meta["kind"]]
    probs = batch_probs(frames, slot_key, verifs[vkey]) if vkey in verifs else None
    labels = {}; rf_labels = None
    rf_cell = f"clf->filt[robust8,rej>={REJECT_FLOOR}]"
    if clfs and meta["kind"] != "gray":  # router-on-confusers (gray excluded: wrong regime, see part_s4)
        F8, F32 = meta["F8"], meta["F32"]
        F8mat = np.stack([fr["f8_all"] for fr in frames])
        F32mat = np.stack([fr["f32_all"] for fr in frames])
        labels = {c: batch_labels(clf, F8mat, F32mat, F8, F32) for c, clf in clfs.items()}
        if "robust8" in clfs:
            rf_labels = reject_floor_labels(clfs["robust8"], F8mat, F8, REJECT_FLOOR)
    is_gray = meta["is_grayscale"]
    f8_clfs = [c for c in labels if clfs[c].get("feat_key") == "f8"]
    names = (["bare", "filt_mlp", "filt_patch"] + [f"clf[{c}]" for c in labels]
             + [f"clf->filt[{c}]" for c in labels] + [f"filt->clf[{c}]" for c in f8_clfs]
             + ([rf_cell] if rf_labels is not None else []))
    rows = {k: {"fp": 0, "flags": []} for k in names}

    def tally(cell, k):
        rows[cell]["fp"] += k; rows[cell]["flags"].append(int(k > 0))

    k_mlp_list, f8f_list = [], []
    for i, fr in enumerate(frames):
        slot = fr[slot_key]; n = len(slot["confs"])
        k_mlp = int((probs[i] >= thr).sum()) if probs is not None else n
        k_mlp_list.append(k_mlp)
        tally("bare", n)
        tally("filt_mlp", k_mlp)
        tally("filt_patch", int(patch_mask(slot, patch_thr).sum()))
        for cname, lab in labels.items():
            keep = gate(int(lab[i]), slot_key)
            tally(f"clf[{cname}]", n if keep else 0)
            tally(f"clf->filt[{cname}]", k_mlp if keep else 0)
        if rf_labels is not None:
            tally(rf_cell, k_mlp if gate(int(rf_labels[i]), slot_key) else 0)
        if f8_clfs and probs is not None:
            m = probs[i] >= thr
            rgb5f = dets5(fr["rgb"], m) if slot_key == "rgb" else dets5(fr["rgb"])
            ir5f = dets5(fr["ir"], m) if slot_key == "ir" else dets5(fr["ir"])
            f8f_list.append(recompute_f8(rgb5f, ir5f, is_gray))

    # filt->clf for f8 routers on confusers: filter first, then route on the recomputed f8 (one
    # batched predict per router). For a no-reject router this is the order's confuser fire.
    if f8_clfs and f8f_list:
        F8f = np.stack(f8f_list)
        F32dummy = np.zeros((len(F8f), len(meta["F32"])), np.float32)
        for cname in f8_clfs:
            fl = batch_labels(clfs[cname], F8f, F32dummy, meta["F8"], meta["F32"])
            for i in range(len(frames)):
                tally(f"filt->clf[{cname}]", k_mlp_list[i] if gate(int(fl[i]), slot_key) else 0)
    n = max(meta["n"], 1)
    out = {}
    for c, v in rows.items():
        ci = boot_rate_ci(v["flags"])
        out[c] = {"FP": v["fp"], "fire_rate": round(sum(v["flags"]) / n, 4),
                  "fired": int(sum(v["flags"])), "verifier": vkey}
        if ci:
            out[c]["fire_ci"] = [round(ci[0], 4), round(ci[1], 4)]
    return out


# ── report ────────────────────────────────────────────────────────────────────────────────────────────
def _f1c(p):
    ci = p.get("f1_ci")
    return f"{p['f1']}" + (f" [{ci[0]}–{ci[1]}]" if ci else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--only", default="", help="comma list of surfaces")
    ap.add_argument("--patch-thr", type=float, default=0.5, help="patch veto: drop iff P(confuser)>=thr")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cdir, outdir = Path(args.cache_dir), Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"Replay <- {cdir}  (zero-GPU, batched)")
    clfs, verifs = load_classifiers(), load_verifiers(args.device)
    pkls = sorted(p for p in cdir.glob("*.pkl") if not only or p.stem in only)
    if not pkls:
        print("  no cache .pkl yet — run pipeline_cache_unified.py first."); return

    results = {}
    L = ["# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)",
         f"{time.strftime('%Y-%m-%d %H:%M')} | detectors ft4+v3b | patch_thr={args.patch_thr} | "
         f"robust8 tau={TAU} | mlp thr rgb={RGB_THR_MLP} / ir={IR_THR_MLP} / gray={GRAY_THR_MLP}",
         "Even-strided ~4k cap per surface (all frames where smaller); n and n_source printed. "
         "Per-frame grain; temporal/segment evidence = the separate real-video run. "
         "95% bootstrap CIs (frame resample, 1000 iters) in brackets.\n"]

    for pkl in pkls:
        d = pickle.load(open(pkl, "rb")); meta, frames = d["meta"], d["frames"]
        name = meta["name"]; t0 = time.time()
        res = {"meta": {k: meta[k] for k in ("name", "kind", "rule", "has_drones", "is_grayscale",
                                             "rgb_imgsz", "ir_imgsz", "n", "stride", "tier") if k in meta}}
        res["meta"]["n_source"] = meta.get("n_source", meta["n"])
        res["A_bare"] = part_a(meta, frames)
        paired_like = meta["kind"] in ("paired", "grayrgb_paired")
        if paired_like and meta["has_drones"]:
            res["B_pipeline"] = part_b(meta, frames, clfs, verifs, args.patch_thr)
        elif meta["has_drones"]:
            res["S4_verifier"] = part_s4(meta, frames, clfs, verifs, args.patch_thr)
        if not meta["has_drones"] and not paired_like:
            res["C_confuser"] = part_c(meta, frames, clfs, verifs, args.patch_thr)
        # paired-like confuser surfaces (video_confuser) are scored by the TEMPORAL replay
        if meta["kind"] == "gray":
            res["GRAY_SWEEP"] = part_gray_sweep(meta, frames, verifs)
        results[name] = res
        print(f"  [{name}] n={meta['n']} {time.time()-t0:.1f}s")

        L.append(f"\n## {name}  (n={meta['n']} of {res['meta']['n_source']}, kind={meta['kind']}, "
                 f"rule={meta['rule']}, imgsz rgb={meta['rgb_imgsz']}/ir={meta['ir_imgsz']}, "
                 f"drones={meta['has_drones']})\n")
        if res["A_bare"]:
            L.append("**A — bare detector (per modality vs own GT)**\n")
            L.append("| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |\n|---|---|---|---|---|---|---|---|")
            for m, p in res["A_bare"].items():
                L.append(f"| {m} | {p['TP']} | {p['FP']} | {p['FN']} | {p['precision']} | {p['recall']} | {_f1c(p)} | {p['n_gt']} |")
        if "B_pipeline" in res:
            L.append("\n**B — full-pipeline ablation (per-modality scoring, NO union)**\n")
            L.append("| cell | TP | FP | FN | P | R | F1 [95% CI] |\n|---|---|---|---|---|---|---|")
            for c, p in res["B_pipeline"].items():
                L.append(f"| {c} | {p['TP']} | {p['FP']} | {p['FN']} | {p['precision']} | {p['recall']} | {_f1c(p)} |")
        if "S4_verifier" in res:
            vk = VERIFIER_LABELS.get(next(iter(res["S4_verifier"].values()))["verifier"], "?")
            L.append(f"\n**S4 — verifier-only ablation (single modality, verifier={vk})**\n")
            L.append("| cell | TP | FP | FN | P | R | F1 [95% CI] |\n|---|---|---|---|---|---|---|")
            for c, p in res["S4_verifier"].items():
                L.append(f"| {c} | {p['TP']} | {p['FP']} | {p['FN']} | {p['precision']} | {p['recall']} | {_f1c(p)} |")
        if "C_confuser" in res:
            vk = VERIFIER_LABELS.get(res["C_confuser"]["bare"]["verifier"], "?")
            L.append(f"\n**C — confuser FP-reduction (no GT; every surviving det = FP; verifier={vk})**\n")
            L.append("| stage | FP | fire_rate [95% CI] |\n|---|---|---|")
            for c, p in res["C_confuser"].items():
                ci = p.get("fire_ci")
                L.append(f"| {c} | {p['FP']} | {p['fire_rate']}" + (f" [{ci[0]}–{ci[1]}]" if ci else "") + " |")
        if res.get("GRAY_SWEEP"):
            L.append("\n**GRAY operating-point sweep (aligned_gray threshold; cached probs)**\n")
            if meta["has_drones"]:
                L.append("| thr | P | R | F1 |\n|---|---|---|---|")
                for t, p in res["GRAY_SWEEP"].items():
                    L.append(f"| {t} | {p['precision']} | {p['recall']} | {p['f1']} |")
            else:
                L.append("| thr | FP | fire_rate |\n|---|---|---|")
                for t, p in res["GRAY_SWEEP"].items():
                    L.append(f"| {t} | {p['FP']} | {p['fire_rate']} |")

    # PART D: grayscale finding — IR-on-gray (bare + aligned_gray filter, classifier bypassed) vs RGB
    if "svanstrom_gray" in results and "svanstrom" in results:
        L.append("\n## D — GRAYSCALE FINDING (good-only config): IR-on-gray + aligned-gray filter vs RGB\n")
        L.append("| config | P | R | F1 [95% CI] |\n|---|---|---|---|")
        rb = results["svanstrom"]["A_bare"].get("ft4/rgb", {})
        if rb:
            L.append(f"| RGB (ft4) bare on Svanström | {rb.get('precision')} | {rb.get('recall')} | {_f1c(rb)} |")
        raw = results.get("svanstrom_rawrgb", {}).get("A_bare", {}).get("v3b/ir", {})
        if raw:
            L.append(f"| IR on RAW RGB (control) | {raw.get('precision')} | {raw.get('recall')} | {_f1c(raw)} |")
        s4 = results["svanstrom_gray"].get("S4_verifier", {})
        gb, gf = s4.get("bare", {}), s4.get("filt_mlp", {})
        if gb:
            L.append(f"| IR-on-gray (v3b) bare | {gb.get('precision')} | {gb.get('recall')} | {_f1c(gb)} |")
        if gf:
            L.append(f"| IR-on-gray + aligned_gray filter (clf bypassed) | {gf.get('precision')} | {gf.get('recall')} | {_f1c(gf)} |")

    L.append("\n## SPEED (reported separately; NOT from this replay)\n")
    L.append("| component | sad (ms) | happy (ms) | speedup | source |\n|---|---|---|---|---|")
    L.append("| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |")
    L.append("| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |")
    L.append("_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._")

    (outdir / "tier1_screening_results.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(outdir / "tier1_results.json", "w"), indent=2, default=float)
    print(f"\nDONE -> {outdir/'tier1_screening_results.md'}  +  tier1_results.json")
    print("  (Tier-1 even-strided = FINAL per 2026-06-10 decision; temporal = separate real-video run.)")


if __name__ == "__main__":
    main()
