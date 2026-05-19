"""
06_confuser_failure_profile.py — Per-dataset confuser failure profile from existing eval CSVs.

Mines two CSV families (no re-inference):

  (A) eval/results/roboflow_ood/{dataset}/{model}/{split}/{model}_frame_detections.csv
      Columns: stem,n_gt,n_raw,n_filt,tp,fp,fn,tp_f,fp_f,fn_f,n_small,n_medium,n_large,dets,sizes
      The `dets` column is a ";"-separated list of "x1,y1,x2,y2,conf" per detection.
      The `sizes` column is a ";"-separated list of "small"/"medium"/"large" per detection.

  (B) eval/results/_patch_catch_audit/baseline_v2/per_detection.csv
      Columns: frame,category,bucket,det_conf,best_iop,matched_iop,patch_prob,patch_label
      Per-detection records on Svanström @ imgsz=1280 for the baseline RGB model.

Output: docs/analysis/2026-05-17_failure_profile_by_dataset.md

Question this answers:
  - Why does the same model give wildly different confuser-fire rates across datasets?
  - Concretely: is it bbox-size-distribution, detection-confidence distribution, or both?

The script is read-only. No inference runs. No images opened.
"""

from __future__ import annotations
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBO = ROOT / "eval" / "results" / "roboflow_ood"
SVAN_PERDET = ROOT / "eval" / "results" / "_patch_catch_audit" / "baseline_v2" / "per_detection.csv"
OUT_MD = ROOT / "docs" / "analysis" / "2026-05-17_failure_profile_by_dataset.md"

# Datasets to include, with brief notes for the report.
ROBO_TARGETS = [
    ("rgb_bird",           "rgb_baseline",     "RGB confuser: bird"),
    ("rgb_bird",           "rgb_retrained_v2", "RGB confuser: bird"),
    ("rgb_airplane",       "rgb_baseline",     "RGB confuser: airplane"),
    ("rgb_airplane",       "rgb_retrained_v2", "RGB confuser: airplane"),
    ("rgb_helicopter",     "rgb_baseline",     "RGB confuser: helicopter"),
    ("rgb_helicopter",     "rgb_retrained_v2", "RGB confuser: helicopter"),
    ("ir_airplane_hors2",  "ir_model",         "IR confuser: airplane (horizon)"),
    ("ir_airplane_plane",  "ir_model",         "IR confuser: airplane (plane-only)"),
    ("ir_bird",            "ir_model",         "IR confuser: bird"),
    ("ir_mixed_cbam",      "ir_model",         "IR mixed valid (60 each: drone/bird/airplane)"),
    ("ir_drone_night",     "ir_model",         "IR drone OOD (augmented night)"),
    ("rgb_drone",          "rgb_baseline",     "RGB drone OOD"),
    ("rgb_drone",          "rgb_retrained_v2", "RGB drone OOD"),
]


def parse_dets_field(dets_field: str, sizes_field: str):
    """Parse semicolon-separated 'x1,y1,x2,y2,conf' detections.

    Each Roboflow run wrote ';' between detections and ',' within a detection.
    Sizes is ';'-separated 'small/medium/large' aligned 1:1 with dets.
    Returns list of (x1,y1,x2,y2,conf,size_bucket).
    """
    if not dets_field or not dets_field.strip():
        return []
    det_chunks = dets_field.split(";")
    size_chunks = sizes_field.split(";") if sizes_field else []
    out = []
    for i, chunk in enumerate(det_chunks):
        parts = chunk.split(",")
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2, conf = map(float, parts[:5])
        except ValueError:
            continue
        size = size_chunks[i] if i < len(size_chunks) else ""
        out.append((x1, y1, x2, y2, conf, size))
    return out


def load_roboflow(dataset: str, model: str):
    """Aggregate detections across all splits of (dataset, model). Returns dict of summary stats."""
    base = ROBO / dataset / model
    if not base.exists():
        return None
    n_frames = 0
    n_with_dets = 0
    all_dets = []  # (conf, size_bucket, is_fp, bbox_area_px)
    n_gt_total = 0
    n_tp_raw = 0
    n_fp_raw = 0
    n_fn_raw = 0
    n_tp_filt = 0
    n_fp_filt = 0
    for split_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        csv_path = split_dir / f"{model}_frame_detections.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_frames += 1
                try:
                    n_gt = int(row["n_gt"])
                    n_raw = int(row["n_raw"])
                    tp = int(row["tp"])
                    fp = int(row["fp"])
                    fn = int(row["fn"])
                    tp_f = int(row["tp_f"])
                    fp_f = int(row["fp_f"])
                except (KeyError, ValueError):
                    continue
                n_gt_total += n_gt
                n_tp_raw += tp
                n_fp_raw += fp
                n_fn_raw += fn
                n_tp_filt += tp_f
                n_fp_filt += fp_f
                if n_raw > 0:
                    n_with_dets += 1
                dets = parse_dets_field(row.get("dets", ""), row.get("sizes", ""))
                # FP determination: in negatives-only datasets (n_gt=0), all dets are FP.
                # In drone-positive datasets, we cannot tell per-detection TP/FP from this CSV
                # without re-matching; we therefore mark is_fp as None in that case.
                is_neg_only = (n_gt == 0)
                for (x1, y1, x2, y2, conf, size) in dets:
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    is_fp = True if is_neg_only else None
                    all_dets.append((conf, size, is_fp, area))
    if n_frames == 0:
        return None
    return {
        "dataset": dataset,
        "model": model,
        "n_frames": n_frames,
        "n_with_dets": n_with_dets,
        "fire_rate": n_with_dets / n_frames if n_frames else 0.0,
        "n_gt_total": n_gt_total,
        "tp_raw": n_tp_raw,
        "fp_raw": n_fp_raw,
        "fn_raw": n_fn_raw,
        "tp_filt": n_tp_filt,
        "fp_filt": n_fp_filt,
        "fppi_raw": n_fp_raw / n_frames if n_frames else 0.0,
        "n_dets": len(all_dets),
        "dets": all_dets,
    }


def summarise_confs(confs):
    if not confs:
        return {"n": 0}
    confs = sorted(confs)
    return {
        "n": len(confs),
        "median": statistics.median(confs),
        "p25": confs[len(confs) // 4],
        "p75": confs[(len(confs) * 3) // 4],
        "min": confs[0],
        "max": confs[-1],
    }


def size_dist(dets):
    """Return fraction of detections in small/medium/large buckets."""
    if not dets:
        return {"small": 0.0, "medium": 0.0, "large": 0.0, "total": 0}
    counts = defaultdict(int)
    for (_conf, size, _is_fp, _area) in dets:
        counts[size or "unknown"] += 1
    total = sum(counts.values())
    return {
        "small": counts["small"] / total if total else 0.0,
        "medium": counts["medium"] / total if total else 0.0,
        "large": counts["large"] / total if total else 0.0,
        "total": total,
    }


def load_svan_perdet():
    """Per-category confidence distribution from Svanström baseline_v2 per_detection.csv."""
    by_cat = defaultdict(list)
    if not SVAN_PERDET.exists():
        return None
    with SVAN_PERDET.open() as f:
        for row in csv.DictReader(f):
            cat = row.get("bucket") or row.get("category") or ""
            try:
                conf = float(row["det_conf"])
                patch = float(row.get("patch_prob") or 0.0)
            except (KeyError, ValueError):
                continue
            by_cat[cat].append((conf, patch))
    return by_cat


def fmt_pct(x):
    return f"{x*100:.1f}%"


def main():
    print(f"[INFO] Roboflow root: {ROBO}", flush=True)
    print(f"[INFO] Svanström per-det: {SVAN_PERDET}", flush=True)
    print(f"[INFO] Output: {OUT_MD}", flush=True)

    roboflow_rows = []
    for ds, model, note in ROBO_TARGETS:
        info = load_roboflow(ds, model)
        if info is None:
            print(f"[WARN] missing data: {ds}/{model}")
            continue
        info["note"] = note
        roboflow_rows.append(info)
        print(f"  loaded {ds}/{model}: frames={info['n_frames']}, dets={info['n_dets']}, fppi_raw={info['fppi_raw']:.3f}")

    svan_by_cat = load_svan_perdet()
    if svan_by_cat is None:
        print(f"[WARN] Svanström per_detection.csv missing at {SVAN_PERDET}")

    # ---------- write markdown ----------
    lines = []
    lines.append("# Per-dataset confuser failure profile (2026-05-17)")
    lines.append("")
    lines.append("Mined from existing eval CSVs; no re-inference. See `analytics/spec_analysis/06_confuser_failure_profile.py`.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("- Roboflow OOD: `eval/results/roboflow_ood/{dataset}/{model}/{split}/{model}_frame_detections.csv`")
    lines.append("- Svanström per-detection: `eval/results/_patch_catch_audit/baseline_v2/per_detection.csv` (baseline RGB, imgsz=1280, IoP@0.5, conf=0.25, stride=9)")
    lines.append("")

    # Section: Roboflow fire rate + FPPI per (model, dataset)
    lines.append("## 1. Roboflow OOD --- fire rate, FPPI, size distribution by (model, dataset)")
    lines.append("")
    lines.append("`fire rate` = fraction of frames with at least one raw detection. `FPPI raw` = FPs per image on negatives-only datasets (drone datasets included for reference). Size buckets are as recorded in `frame_detections.csv` by the Roboflow eval harness.")
    lines.append("")
    lines.append("| Dataset | Model | Frames | Dets | Fire rate | FPPI raw | % small | % medium | % large |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for info in roboflow_rows:
        sd = size_dist(info["dets"])
        lines.append(
            f"| `{info['dataset']}` | `{info['model']}` | {info['n_frames']} | {info['n_dets']} | "
            f"{fmt_pct(info['fire_rate'])} | {fmt_pct(info['fppi_raw'])} | "
            f"{fmt_pct(sd['small'])} | {fmt_pct(sd['medium'])} | {fmt_pct(sd['large'])} |"
        )
    lines.append("")

    # Section: Roboflow per-detection confidence distribution
    lines.append("## 2. Roboflow OOD --- detection confidence distribution by (model, dataset)")
    lines.append("")
    lines.append("Per-detection confidence is parsed from the `dets` column. For negative-only datasets every detection is a false positive; for drone datasets a detection may be TP or FP (the CSV does not separate them, so confidences here are mixed for drone datasets).")
    lines.append("")
    lines.append("| Dataset | Model | N dets | conf p25 | conf median | conf p75 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for info in roboflow_rows:
        confs = [d[0] for d in info["dets"]]
        s = summarise_confs(confs)
        if s["n"] == 0:
            lines.append(f"| `{info['dataset']}` | `{info['model']}` | 0 | --- | --- | --- |")
        else:
            lines.append(
                f"| `{info['dataset']}` | `{info['model']}` | {s['n']} | "
                f"{s['p25']:.3f} | {s['median']:.3f} | {s['p75']:.3f} |"
            )
    lines.append("")

    # Section: Svanström baseline per-category
    if svan_by_cat:
        lines.append("## 3. Svanström baseline --- per-category detection confidence (imgsz=1280, stride=9)")
        lines.append("")
        lines.append("This is the *only* per-detection dump we have on Svanström. It is for the baseline RGB model with patch verifier v2; we read only the `det_conf` column (YOLO confidence, no patch suppression applied here). DRONE_TP and DRONE_FP are split by IoP match to GT; BIRD/AIRPLANE/HELICOPTER are confuser frames where any detection is, by construction, an FP.")
        lines.append("")
        lines.append("| Category | N dets | conf p25 | conf median | conf p75 | patch p median |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cat in ["DRONE_TP", "DRONE_FP", "BIRD", "AIRPLANE", "HELICOPTER", "OTHER"]:
            entries = svan_by_cat.get(cat, [])
            if not entries:
                continue
            confs = [c for (c, _p) in entries]
            patches = [p for (_c, p) in entries]
            s = summarise_confs(confs)
            patch_med = statistics.median(patches) if patches else 0.0
            lines.append(
                f"| {cat} | {s['n']} | {s['p25']:.3f} | {s['median']:.3f} | {s['p75']:.3f} | {patch_med:.3f} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## 4. What the numbers say")
    lines.append("")
    lines.append("Read off the tables above. The script does not interpret; it only summarises. Cross-reference for the thesis:")
    lines.append("")
    lines.append("- **Size distribution as fire-rate explanator.** Compare `%small` between e.g. `ir_airplane_hors2` and `ir_airplane_plane`. If `hors2` is overwhelmingly small detections and `plane` is medium/large, the IR detector's airplane fire-rate gap (58% vs 16.5% FPPI per Ledger §8.4) is at least partly a small-object regime story.")
    lines.append("- **Confidence as a thresholding lever.** Compare conf-median between confuser datasets for the same model. If one dataset's FPs are concentrated at low confidence and another's are at high confidence, a confidence-only threshold can separate them; if both are at high confidence, the model is genuinely sure and the cascade is what we need.")
    lines.append("- **RGB baseline on Svanström by category (§3).** Compare median DRONE_TP confidence vs median BIRD confidence. If they are similar, the baseline cannot discriminate at the confidence level; if BIRD confidence is lower, a higher conf threshold would help (at a recall cost).")
    lines.append("")
    lines.append("## 5. Delivered")
    lines.append("")
    lines.append(f"- `{OUT_MD.relative_to(ROOT).as_posix()}` --- this report.")
    lines.append("- `analytics/spec_analysis/06_confuser_failure_profile.py` --- the script that produced it.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
