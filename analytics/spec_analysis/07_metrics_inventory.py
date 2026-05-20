"""
07_metrics_inventory.py — Consolidate all evaluation CSVs into one long-format
metrics inventory + a gap matrix.

Walks every known eval/results/*/summary.csv (or equivalent) and emits:

  docs/analysis/2026-05-19_metrics_inventory.csv
      Long format: one row per (model, surface, stage, category, size_bucket, metric, value).

  docs/analysis/2026-05-19_metrics_inventory_gap.md
      What's missing — cells where we don't have P/R/F1/FPPI for a (model x surface x size x stage) combo.

Read-only. No inference. Run from repo root.
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "eval" / "results"
OUT_DIR = REPO / "docs" / "analysis"
OUT_CSV = OUT_DIR / "2026-05-19_metrics_inventory.csv"
OUT_GAP = OUT_DIR / "2026-05-19_metrics_inventory_gap.md"

# Canonical model name normalisation (CSVs use varying labels)
MODEL_ALIASES = {
    "baseline_trained": "baseline",
    "rgb_baseline": "baseline",
    "Yolo26n_trained": "baseline",
    "retrained_v2": "retrained_v2",
    "rgb_retrained_v2": "retrained_v2",
    "selcom_1280": "selcom_1280",
    "selcom_640": "selcom_640",
    "hardneg_v3more": "hardneg_v3more",
    "ir_model": "ir_model",
    "ir_final_gray": "ir_grayscale",
    "ir_final_rgb": "ir_on_rgb",
}

ROWS: list[dict] = []


def push(**kw):
    row = {
        "model": "", "surface": "", "modality": "", "category": "all",
        "stage": "rgb_raw", "classifier": "none", "size_bucket": "all",
        "frames": "", "tp": "", "fp": "", "fn": "",
        "precision": "", "recall": "", "f1": "", "fppi": "", "fp_frame_rate": "",
        "source": "",
    }
    row.update(kw)
    row["model"] = MODEL_ALIASES.get(row["model"], row["model"])
    ROWS.append(row)


def safe_float(s):
    try:
        return float(s) if s not in ("", None) else ""
    except (TypeError, ValueError):
        return ""


# ── 1. Svanstrom per-category (baseline, hardneg_v3more, retrained_v2) ──
svan_csv = RESULTS / "_failure_diagnosis" / "svanstrom_1280_by_category.csv"
if svan_csv.exists():
    with svan_csv.open() as f:
        for r in csv.DictReader(f):
            push(
                model=r["model"], surface="svanstrom_rgb", modality="rgb",
                category=r["category"], stage="rgb_raw",
                frames=r["frames"], tp=r["TP"], fp=r["FP"], fn=r["FN"],
                precision=r["precision"], recall=r["recall"],
                source=str(svan_csv.relative_to(REPO)),
            )

# ── 2. Roboflow OOD (RGB + IR, all splits, with size buckets) ──
rob_csv = RESULTS / "roboflow_ood" / "summary.csv"
if rob_csv.exists():
    with rob_csv.open() as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]  # e.g. rgb_bird/test, ir_drone_night/valid
            modality = "rgb" if ds.startswith("rgb_") else ("ir" if ds.startswith("ir_") else "?")
            dataset_clean = ds.split("/")[0]
            # Aggregate row (raw)
            push(
                model=r["model"], surface=f"roboflow_{dataset_clean}", modality=modality,
                category="DRONE" if "drone" in dataset_clean else dataset_clean.split("_", 1)[1].upper(),
                stage="rgb_raw" if modality == "rgb" else "ir_raw",
                frames="", tp=r["raw_TP"], fp=r["raw_FP"], fn=r["raw_FN"],
                precision=r["raw_precision"], recall=r["raw_recall"], f1=r["raw_f1"],
                source=str(rob_csv.relative_to(REPO)) + f"#{ds}",
            )
            # +patch stage
            push(
                model=r["model"], surface=f"roboflow_{dataset_clean}", modality=modality,
                category="DRONE" if "drone" in dataset_clean else dataset_clean.split("_", 1)[1].upper(),
                stage="+patch",
                tp=r["filt_TP"], fp=r["filt_FP"], fn=r["filt_FN"],
                precision=r["filt_precision"], recall=r["filt_recall"], f1=r["filt_f1"],
                source=str(rob_csv.relative_to(REPO)) + f"#{ds}",
            )
            # Per-size FP breakdown (and TP if present)
            for bucket, tp_col, fp_col in [("small", "tp_S", "fp_S"), ("medium", "tp_M", "fp_M"), ("large", "tp_L", "fp_L")]:
                push(
                    model=r["model"], surface=f"roboflow_{dataset_clean}", modality=modality,
                    category="DRONE" if "drone" in dataset_clean else dataset_clean.split("_", 1)[1].upper(),
                    stage="+patch", size_bucket=bucket,
                    tp=r.get(tp_col, ""), fp=r.get(fp_col, ""),
                    source=str(rob_csv.relative_to(REPO)) + f"#{ds}",
                )

# ── 3. Video tests comparison (all 6 models, drone + confuser clips) ──
vt_csv = RESULTS / "video_tests" / "video_tests_comparison.csv"
if vt_csv.exists():
    with vt_csv.open() as f:
        for r in csv.DictReader(f):
            push(
                model=r["model"], surface=f"real_video_{r['dataset']}", modality="rgb",
                category=r["category"].upper(), stage="rgb_raw",
                frames=r["total_frames"], tp=r["iop_tp"], fp=r["iop_fp"], fn=r["iop_fn"],
                precision=r["iop_prec"], recall=r["iop_rec"], f1=r["iop_f1"],
                fppi=r["fppi"], fp_frame_rate=r["fp_frame_rate"],
                source=str(vt_csv.relative_to(REPO)),
            )

# ── 4. Pipeline (cascade) comparison ──
pl_csv = RESULTS / "pipeline_video_tests" / "pipeline_comparison.csv"
if pl_csv.exists():
    with pl_csv.open() as f:
        for r in csv.DictReader(f):
            stages = [
                ("rgb_raw", "rgb"),
                ("ir_raw", "ir"),
                ("+classifier", "clf"),
                ("+temporal", "seg_temp"),
                ("+temporal+patch", "seg_final"),
            ]
            for stage, prefix in stages:
                push(
                    model=r["rgb_model"], surface=f"real_video_{r['dataset']}", modality="rgb",
                    category=r["category"].upper(), stage=stage, classifier="sa32",
                    frames=r["total_frames"],
                    tp=r.get(f"{prefix}_tp", ""), fp=r.get(f"{prefix}_fp", ""), fn=r.get(f"{prefix}_fn", ""),
                    precision=r.get(f"{prefix}_p", ""), recall=r.get(f"{prefix}_r", ""), f1=r.get(f"{prefix}_f1", ""),
                    source=str(pl_csv.relative_to(REPO)),
                )

# Variants for control40 and fusion_no_fn
for variant_dir, clf_name in [
    ("pipeline_video_tests_control40", "control40"),
    ("pipeline_video_tests_fusionnofn", "fnfn"),
]:
    pc = RESULTS / variant_dir / "pipeline_comparison.csv"
    if pc.exists():
        with pc.open() as f:
            for r in csv.DictReader(f):
                for stage, prefix in [("+classifier", "clf"), ("+temporal", "seg_temp"), ("+temporal+patch", "seg_final")]:
                    push(
                        model=r["rgb_model"], surface=f"real_video_{r['dataset']}", modality="rgb",
                        category=r["category"].upper(), stage=stage, classifier=clf_name,
                        frames=r["total_frames"],
                        tp=r.get(f"{prefix}_tp", ""), fp=r.get(f"{prefix}_fp", ""), fn=r.get(f"{prefix}_fn", ""),
                        precision=r.get(f"{prefix}_p", ""), recall=r.get(f"{prefix}_r", ""), f1=r.get(f"{prefix}_f1", ""),
                        source=str(pc.relative_to(REPO)),
                    )

# ── 5. Svanstrom per-size (Phase 2 D) ──
svan_ps = RESULTS / "svanstrom_persize" / "summary.csv"
if svan_ps.exists():
    with svan_ps.open() as f:
        for r in csv.DictReader(f):
            push(
                model=r["model"], surface="svanstrom_rgb", modality="rgb",
                category=r["category"], stage="rgb_raw", size_bucket=r["size_bucket"],
                tp=r["TP"], fp=r["FP"], fn=r["FN"],
                precision=r["precision"], recall=r["recall"], f1=r["f1"],
                source=str(svan_ps.relative_to(REPO)),
            )

# ── 6. Real-video per-size (Phase 2 A) ──
vid_ps = RESULTS / "video_persize" / "summary.csv"
if vid_ps.exists():
    with vid_ps.open() as f:
        for r in csv.DictReader(f):
            push(
                model=r["model"], surface=f"real_video_{r['clip']}", modality="rgb",
                category="DRONE" if r["has_drone_gt"] == "True" else r["category"].upper(),
                stage="rgb_raw", size_bucket=r["size_bucket"],
                frames=r["n_frames"], tp=r["TP"], fp=r["FP"], fn=r["FN"],
                precision=r["precision"], recall=r["recall"], f1=r["f1"],
                fppi=r["fppi_bucket"],
                source=str(vid_ps.relative_to(REPO)),
            )

# ── 7. Selcom val held-out (Phase 2 B) ──
sel_dir = RESULTS / "selcom_val_holdout"
if sel_dir.exists():
    for jpath in sel_dir.rglob("*_results.json"):
        try:
            data = json.loads(jpath.read_text())
        except Exception:
            continue
        mname = data.get("model", jpath.parent.name)
        # Aggregate IoP metrics (index 1 if present)
        dm = data.get("detection_metrics", [])
        m = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
        push(
            model=mname, surface="selcom_val_holdout", modality="rgb",
            category="DRONE", stage="rgb_raw",
            tp=m.get("TP", ""), fp=m.get("FP", ""), fn=m.get("FN", ""),
            precision=m.get("precision", ""), recall=m.get("recall", ""), f1=m.get("f1", ""),
            source=str(jpath.relative_to(REPO)),
        )
        # Per-size
        psm = data.get("per_size_metrics", {}).get("iop", {})
        for b, vals in psm.items():
            tp = vals.get("tp", 0); fp = vals.get("fp", 0); fn = vals.get("fn", 0)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            push(
                model=mname, surface="selcom_val_holdout", modality="rgb",
                category="DRONE", stage="rgb_raw", size_bucket=b,
                tp=tp, fp=fp, fn=fn,
                precision=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0,
                source=str(jpath.relative_to(REPO)),
            )

# ── 8. Anti-UAV per-model (Phase 2 E) ──
au_dir = RESULTS / "antiuav_per_model"
if au_dir.exists():
    for jpath in au_dir.rglob("*_results.json"):
        try:
            data = json.loads(jpath.read_text())
        except Exception:
            continue
        mname = data.get("model", jpath.parent.name)
        dm = data.get("detection_metrics", [])
        m = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
        push(
            model=mname, surface="antiuav_per_model", modality="rgb",
            category="DRONE", stage="rgb_raw",
            tp=m.get("TP", ""), fp=m.get("FP", ""), fn=m.get("FN", ""),
            precision=m.get("precision", ""), recall=m.get("recall", ""), f1=m.get("f1", ""),
            source=str(jpath.relative_to(REPO)),
        )
        psm = data.get("per_size_metrics", {}).get("iop", {})
        for b, vals in psm.items():
            tp = vals.get("tp", 0); fp = vals.get("fp", 0); fn = vals.get("fn", 0)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            push(
                model=mname, surface="antiuav_per_model", modality="rgb",
                category="DRONE", stage="rgb_raw", size_bucket=b,
                tp=tp, fp=fp, fn=fn,
                precision=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0,
                source=str(jpath.relative_to(REPO)),
            )

# ── 9. Anti-UAV legacy summary ──
au_metrics = RESULTS / "antiuav" / "metrics_iop.csv"
if au_metrics.exists():
    with au_metrics.open() as f:
        for r in csv.DictReader(f):
            push(
                model=r["config"], surface="antiuav", modality="rgb_ir" if r["config"] == "rgb_only" else "ir" if r["config"] == "ir_only" else "fused",
                category="DRONE", stage="rgb_raw" if r["config"] == "rgb_only" else "ir_raw",
                tp=r["TP"], fp=r["FP"], fn=r["FN"],
                precision=r["precision"], recall=r["recall"], f1=r["f1"],
                source=str(au_metrics.relative_to(REPO)),
            )

# ── Write outputs ──
OUT_DIR.mkdir(parents=True, exist_ok=True)
fields = list(ROWS[0].keys()) if ROWS else []
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(ROWS)

# Build coverage / gap matrix
have = defaultdict(set)  # (surface) -> set of (model, stage, classifier, size_bucket)
for r in ROWS:
    have[r["surface"]].add((r["model"], r["stage"], r["classifier"], r["size_bucket"]))

# Target combos
TARGET_MODELS_RGB = ["baseline", "retrained_v2", "selcom_640", "selcom_1280", "hardneg_v3more"]
TARGET_MODELS_IR = ["ir_model", "ir_grayscale", "ir_on_rgb"]
TARGET_STAGES = ["rgb_raw", "+classifier", "+temporal", "+temporal+patch"]
TARGET_CLFS = ["sa32", "control40", "fnfn"]
TARGET_SIZES = ["all", "small", "medium", "large"]

surfaces_seen = sorted(have.keys())

lines = [
    "# Metrics inventory — gap matrix",
    "",
    f"Generated by `analytics/spec_analysis/07_metrics_inventory.py`.",
    f"Source CSV: `{OUT_CSV.relative_to(REPO)}`.",
    "",
    f"Surfaces with any data ({len(surfaces_seen)}):",
    "",
]
for s in surfaces_seen:
    lines.append(f"- `{s}` — {len(have[s])} (model, stage, classifier, size) tuples")
lines.append("")
lines.append("## Gaps by surface")
lines.append("")

for s in surfaces_seen:
    lines.append(f"### `{s}`")
    lines.append("")
    rows_have = have[s]
    models_have = sorted({m for (m, _, _, _) in rows_have})
    stages_have = sorted({st for (_, st, _, _) in rows_have})
    sizes_have = sorted({sz for (_, _, _, sz) in rows_have})
    clfs_have = sorted({cl for (_, _, cl, _) in rows_have})
    lines.append(f"- models present: {', '.join(models_have)}")
    lines.append(f"- stages present: {', '.join(stages_have)}")
    lines.append(f"- classifiers present: {', '.join(clfs_have)}")
    lines.append(f"- size buckets present: {', '.join(sizes_have)}")
    # Missing models (RGB surfaces)
    is_ir = s.startswith("roboflow_ir") or "ir" in s.lower() and "rgb" not in s.lower()
    target_models = TARGET_MODELS_IR if is_ir else TARGET_MODELS_RGB
    missing_models = sorted(set(target_models) - set(models_have))
    if missing_models:
        lines.append(f"- **missing models**: {', '.join(missing_models)}")
    if "all" not in sizes_have and not sizes_have:
        lines.append(f"- **no size breakdown**")
    elif sizes_have == ["all"]:
        lines.append(f"- **no size breakdown** (only aggregate)")
    lines.append("")

with OUT_GAP.open("w") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUT_CSV} ({len(ROWS)} rows)")
print(f"Wrote {OUT_GAP}")
print(f"Surfaces: {len(surfaces_seen)}")
