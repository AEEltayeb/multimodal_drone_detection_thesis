"""
10_full_pipeline_doc.py — Build the per-dataset section of the full-pipeline
ablations doc.

Reads:
  eval/results/full_pipeline_persize/<dataset>/<detector>/<classifier>/summary.csv
    (produced by eval/eval_full_pipeline_persize.py)

Writes:
  docs/analysis/full_pipeline_ablations/<dataset>.md       <- per-dataset deep dive
  docs/analysis/full_pipeline_ablations/csv/<dataset>.csv  <- long-form table

Run one dataset at a time:
  python analytics/spec_analysis/10_full_pipeline_doc.py --dataset antiuav

The doc layout matches docs/analysis/full_pipeline_ablations design:
  - headline
  - overall summary table (all-sizes; one row per (model x stage))
  - per-size breakdown (one matrix per model)
  - per-stage commentary
  - sanity flags
  - delivered paths
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO / "docs" / "analysis" / "full_pipeline_ablations"
RESULTS_ROOT = DOC_ROOT / "raw_results"
CSV_DIR = DOC_ROOT / "csv"


# ── Stage / modality mapping ─────────────────────────────────────────

# Display label per (modality, stage). The doc reorders to: per-modality rows
# (detector-only, +filter) -> classifier rows -> temporal rows.
RGB_DETECTORS = {"baseline", "hardneg_v3more", "retrained_v2",
                 "selcom_1280", "selcom_960", "selcom_640"}
IR_NATIVE = "ir_model"
IR_GRAY = "ir_grayscale"


def modality_of(detector: str) -> str:
    if detector in RGB_DETECTORS:
        return "rgb"
    if detector == IR_NATIVE:
        return "ir_native"
    if detector == IR_GRAY:
        return "ir_grayscale"
    return "unknown"


def detector_stage_label(detector: str, stage: str) -> Optional[str]:
    """Map (detector, raw stage) to the doc's stage label, or None to drop.

    Doc stages:
      rgb_only / ir_native / ir_grayscale  <- S0 of that detector
      +rgb_filter / +ir_filter             <- S3 (patch only, no classifier)
      classifier                           <- S1 (only meaningful on RGB rows;
                                              IR rows duplicate this, so we
                                              report under the RGB detector)
      classifier+filter                    <- S2
      temporal                             <- S4
      temporal+alert-gate                  <- S5
    """
    mod = modality_of(detector)
    if stage == "S0_detector":
        return mod  # "rgb" / "ir_native" / "ir_grayscale"
    if stage == "S3_+patch_only":
        if mod == "rgb" or mod == "ir_grayscale":
            return "+rgb_filter"
        if mod == "ir_native":
            return "+ir_filter"
    if stage == "S1_+classifier":
        return "classifier"
    if stage == "S2_+classifier+patch":
        return "classifier→filter"
    if stage == "S4_temporal_no_filter":
        return "temporal"
    if stage == "S5_alert_gate_filter":
        return "temporal+alert_gate"
    return None


STAGE_ORDER = [
    "rgb", "ir_native", "ir_grayscale",
    "+rgb_filter", "+ir_filter",
    "classifier", "classifier→filter",
    "temporal", "temporal+alert_gate",
]


# ── CSV ingestion ────────────────────────────────────────────────────

def load_combo(dataset: str, detector: str, classifier: str) -> list[dict]:
    p = RESULTS_ROOT / dataset / detector / classifier / "summary.csv"
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def discover_combos(dataset: str) -> list[tuple[str, str]]:
    """Return (detector, classifier) combos with a summary.csv on disk."""
    out = []
    ds_root = RESULTS_ROOT / dataset
    if not ds_root.exists():
        return out
    for det_dir in sorted(ds_root.iterdir()):
        if not det_dir.is_dir(): continue
        for clf_dir in sorted(det_dir.iterdir()):
            if not clf_dir.is_dir(): continue
            if (clf_dir / "summary.csv").exists():
                out.append((det_dir.name, clf_dir.name))
    return out


# ── Aggregation ──────────────────────────────────────────────────────

def _to_float(s, default=0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _to_int(s, default=0):
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def aggregate_all_sizes(rows_for_stage: list[dict]) -> dict:
    """Sum TP/FP/FN across size buckets and recompute P/R/F1."""
    tp = sum(_to_int(r["TP"]) for r in rows_for_stage)
    fp = sum(_to_int(r["FP"]) for r in rows_for_stage)
    fn = sum(_to_int(r["FN"]) for r in rows_for_stage)
    tn_raw = rows_for_stage[0].get("TN") if rows_for_stage else ""
    tn = _to_int(tn_raw, default=0) if tn_raw not in ("", None) else None
    n_gt = sum(_to_int(r["n_gt"]) for r in rows_for_stage)
    n_frames = _to_int(rows_for_stage[0]["n_frames"]) if rows_for_stage else 0
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    fr_pct = 100.0 * (tp + fp) / n_frames if n_frames > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "P": P, "R": R, "F1": F, "FR_pct": fr_pct,
        "n_gt": n_gt, "n_frames": n_frames,
    }


def collect(dataset: str) -> dict:
    """Walk every (detector, classifier) under the dataset and build a
    nested dict:
      data[detector][stage_label][size_bucket] = row_dict
    where size_bucket is one of {small, medium, large, all}.
    """
    data: dict = defaultdict(lambda: defaultdict(dict))
    for detector, classifier in discover_combos(dataset):
        rows = load_combo(dataset, detector, classifier)
        # Group rows by raw stage
        by_stage: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_stage[r["stage"]].append(r)
        for raw_stage, srows in by_stage.items():
            label = detector_stage_label(detector, raw_stage)
            if label is None:
                continue
            # S1/S2 (classifier rows) only meaningful from RGB detectors —
            # IR detectors run in classifier=sa32 mode produce duplicates
            # that we drop to keep one canonical row per classifier stage.
            if label in ("classifier", "classifier→filter") and \
               modality_of(detector) != "rgb":
                continue
            # Only pull classifier rows when classifier != none
            if label in ("classifier", "classifier→filter") and classifier == "no_classifier":
                continue
            # Non-classifier stages: prefer the no_classifier file so we don't
            # double-count when both sa32 and no_classifier files exist.
            if label not in ("classifier", "classifier→filter") and \
               classifier != "no_classifier" and \
               (detector, "no_classifier") in [(d, c) for d, c in discover_combos(dataset)]:
                continue

            # Per size bucket
            for r in srows:
                bucket = r["size_bucket"]
                data[detector][label][bucket] = r
            # And an "all"-size aggregate (skip if the only bucket is "all" already)
            buckets_present = {r["size_bucket"] for r in srows}
            if buckets_present != {"all"}:
                data[detector][label]["all"] = {
                    **aggregate_all_sizes(srows),
                    "stage": raw_stage,
                    "size_bucket": "all",
                    "detector": detector,
                    "classifier": classifier,
                    "n_gt": sum(_to_int(r["n_gt"]) for r in srows),
                    "n_frames": _to_int(srows[0]["n_frames"]),
                }
            else:
                # Already an all-row (S4/S5)
                r0 = srows[0]
                tp = _to_int(r0["TP"]); fp = _to_int(r0["FP"]); fn = _to_int(r0["FN"])
                tn = _to_int(r0.get("TN", 0) or 0)
                P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                n_frames = _to_int(r0["n_frames"])
                fr_pct = 100.0 * (tp + fp) / n_frames if n_frames else 0.0
                data[detector][label]["all"] = {
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "P": P, "R": R, "F1": F, "FR_pct": fr_pct,
                    "n_gt": _to_int(r0["n_gt"]), "n_frames": n_frames,
                }
    return data


# ── Output: long-form CSV ────────────────────────────────────────────

def write_csv(dataset: str, scoring: str, data: dict) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_DIR / f"{dataset}.csv"
    fieldnames = ["dataset", "category", "model", "modality", "stage", "scoring",
                  "size_bucket", "TP", "FP", "FN", "TN", "P", "R", "F1",
                  "FR_pct", "n_gt", "n_frames", "flag"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for detector in sorted(data.keys()):
            mod = modality_of(detector)
            for stage in STAGE_ORDER:
                if stage not in data[detector]:
                    continue
                for bucket, row in data[detector][stage].items():
                    flag = ""
                    if _to_int(row.get("n_gt", 0)) == 0 and bucket != "all":
                        flag = "n_gt_zero"
                    P = row.get("P", _to_float(row.get("precision", 0)))
                    R = row.get("R", _to_float(row.get("recall", 0)))
                    F = row.get("F1", _to_float(row.get("f1", 0)))
                    fr_pct = row.get("FR_pct", "")
                    if fr_pct == "":
                        n_frames = _to_int(row.get("n_frames", 0))
                        tp = _to_int(row.get("TP", 0)); fp = _to_int(row.get("FP", 0))
                        fr_pct = 100.0 * (tp + fp) / n_frames if n_frames else 0.0
                    w.writerow({
                        "dataset": dataset,
                        "category": "drone",  # adjust per dataset later
                        "model": detector,
                        "modality": mod,
                        "stage": stage,
                        "scoring": scoring,
                        "size_bucket": bucket,
                        "TP": _to_int(row.get("TP", 0)),
                        "FP": _to_int(row.get("FP", 0)),
                        "FN": _to_int(row.get("FN", 0)),
                        "TN": row.get("TN", "") if row.get("TN") is not None else "",
                        "P": round(P, 4),
                        "R": round(R, 4),
                        "F1": round(F, 4),
                        "FR_pct": round(_to_float(fr_pct), 2),
                        "n_gt": _to_int(row.get("n_gt", 0)),
                        "n_frames": _to_int(row.get("n_frames", 0)),
                        "flag": flag,
                    })
    return out


# ── Output: markdown deep dive ───────────────────────────────────────

def fmt_row_bbox(detector: str, stage: str, r: dict) -> str:
    """Bbox-level row: TP/FP/FN + P/R/F1 only."""
    tp = _to_int(r.get("TP", 0)); fp = _to_int(r.get("FP", 0))
    fn = _to_int(r.get("FN", 0))
    P = _to_float(r.get("P", r.get("precision", 0)))
    R = _to_float(r.get("R", r.get("recall", 0)))
    F = _to_float(r.get("F1", r.get("f1", 0)))
    return f"| {detector} | {stage} | {tp:,} | {fp:,} | {fn:,} | {P:.4f} | {R:.4f} | {F:.4f} |"


def fmt_row_segment(detector: str, stage: str, r: dict) -> str:
    """Segment/frame-level row: TP/FP/FN/TN + P/R/F1 + FR%."""
    tp = _to_int(r.get("TP", 0)); fp = _to_int(r.get("FP", 0))
    fn = _to_int(r.get("FN", 0))
    tn = _to_int(r.get("TN", 0) or 0)
    P = _to_float(r.get("P", r.get("precision", 0)))
    R = _to_float(r.get("R", r.get("recall", 0)))
    F = _to_float(r.get("F1", r.get("f1", 0)))
    n_seg = tp + fp + fn + tn
    fr = 100.0 * (tp + fp) / n_seg if n_seg else 0.0
    return (f"| {detector} | {stage} | {tp:,} | {fp:,} | {fn:,} | {tn:,} | "
            f"{P:.4f} | {R:.4f} | {F:.4f} | {fr:.2f}% |")


def fmt_row_persize(stage: str, bucket: str, r: dict) -> str:
    n_gt = _to_int(r.get("n_gt", 0))
    if n_gt == 0 and bucket != "all":
        return f"| {stage} | {bucket} | 0 | — | — | — | — | — | — |"
    tp = _to_int(r.get("TP", 0)); fp = _to_int(r.get("FP", 0)); fn = _to_int(r.get("FN", 0))
    # Per-size rows from raw CSV carry precision/recall/f1; "all" rows from
    # aggregate_all_sizes() carry P/R/F1. Accept either spelling.
    P = _to_float(r.get("P", r.get("precision", 0)))
    R = _to_float(r.get("R", r.get("recall", 0)))
    F = _to_float(r.get("F1", r.get("f1", 0)))
    return (f"| {stage} | {bucket} | {n_gt} | {tp:,} | {fp:,} | {fn:,} | "
            f"{P:.4f} | {R:.4f} | {F:.4f} |")


# Per-dataset prose. Add an entry as each dataset comes online.
DATASET_META = {
    "svanstrom": {
        "title": "Svanström",
        "category": "RGB+IR not-truly-paired, mixed confuser + drone",
        "scoring": "iop",
        "headline": (
            "The classifier's keystone dataset. RGB collapses on confusers "
            "(birds, airplanes, helicopters) and hallucinates aggressively; IR is "
            "physically immune to feathers/wings and stays clean. Under trust-aware "
            "scoring the classifier routes RGB-confuser frames to the IR stream and "
            "lifts F1 from ~0.43 (RGB alone) to ~0.89 — the modality-arbitration "
            "rescue. The patch verifier (rgb_filter) only helps when applied before "
            "the classifier choice; once the classifier has picked the trustworthy "
            "modality the filter is largely redundant on this dataset."
        ),
        "commentary": {
            "rgb": "Baseline RGB detector. Saturated by confusers — most FPs are birds. The R looks healthy (>0.9) but P is destroyed (<0.3).",
            "ir_native": "IR detector on the paired IR frame. The most useful single signal on this dataset (F1≈0.95).",
            "ir_grayscale": "IR weights on grayscale-RGB. Shown for symmetry with the RGB-only doc; not used in production on paired data.",
            "+rgb_filter": "Patch verifier on RGB dets only. Catches some bird FPs but doesn't reach IR's confuser robustness.",
            "+ir_filter": "Patch verifier on IR dets. Marginal effect since IR rarely hallucinates here.",
            "classifier": "sa32 trust-aware: for each frame, classifier picks which modality (or both) to credit. The headline number — this is where modality arbitration shows up.",
            "classifier→filter": "Classifier picks, then filter applied to the trusted side. So + filters out the residual after arbitration.",
            "temporal": "3-frame segments, 2-of-3 voting on the raw detector firing pattern. Caps the per-segment FR%.",
            "temporal+alert_gate": "Production rule. The patch verifier runs only on the 3rd frame, gate-keeping the alert. So + temporal voting + confuser-veto on the decisive frame.",
        },
    },
    "selcom_val": {
        "title": "Selcom CCTV val (drone-only RGB)",
        "category": "RGB-only, drone-only (CCTV crops)",
        "scoring": "iou",
        "headline": (
            "Held-out drone-only CCTV val split. RGB-only, so the IR side comes "
            "from ir_grayscale (cross-modal fallback). No confusers in this split, "
            "so the patch verifier and classifier have nothing to arbitrate — "
            "the interesting question here is whether the soft-veto / classifier "
            "harms RGB recall in the no-confuser case."
        ),
        "commentary": {
            "rgb": "Baseline RGB on the val split. Reference number.",
            "ir_grayscale": "IR weights on the grayscale-RGB input — cross-modal fallback path. Low recall is expected (IR weights weren't trained on RGB-derived grayscale).",
            "+rgb_filter": "Patch verifier on RGB dets. On a confuser-free dataset, this is a pure recall tax.",
            "classifier": "sa32 trust-aware in grayscale mode (IR side = ir_grayscale).",
            "classifier→filter": "Classifier-trusted dets passed through the patch verifier.",
            "temporal": "3-frame segments. Tight clip framing in CCTV makes temporal voting easy.",
            "temporal+alert_gate": "Production rule, same as elsewhere.",
        },
    },
    "antiuav": {
        "title": "Anti-UAV RGBT",
        "category": "RGB+IR paired, drone-only",
        "scoring": "iou",
        "headline": (
            "Clean paired benchmark — no confusers. RGB and IR both saturate near "
            "F1=0.99 and the classifier should not change much (both modalities are "
            "trustworthy). The filter has nothing to suppress, so it slightly trims "
            "recall without buying precision."
        ),
        "commentary": {
            "rgb": "Detector alone, single frame. Reference RGB row.",
            "ir_native": "IR detector on the paired IR frame. Reference IR row.",
            "ir_grayscale": "IR weights applied to a grayscale copy of the RGB frame. Useful for the cross-modal-on-RGB fallback path.",
            "+rgb_filter": "Patch verifier (rgb_filter v2) applied to every RGB det. On a confuser-free dataset this is a pure recall tax.",
            "+ir_filter": "Patch verifier (ir_filter v2) on every IR det. Same dynamic.",
            "classifier": "sa32 trust classifier picks RGB / IR / both. Scored against the GT of the trusted modality, so TP counts can exceed a single-modality detector.",
            "classifier→filter": "Classifier picks, then the filter of the trusted modality is applied. So + filters out the small residual of mismatched dets.",
            "temporal": "3-frame segments, 2-of-3 voting on the raw detector firing pattern. So + temporal averaging without any confuser logic.",
            "temporal+alert_gate": "Production rule — the filter is applied only on the 3rd frame, right before the alert would fire. So + same as temporal but each fired segment is veto-able by the patch verifier on its triggering frame.",
        },
    },
}


def write_markdown(dataset: str, data: dict) -> Path:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    meta = DATASET_META.get(dataset, {
        "title": dataset, "category": "?", "scoring": "iop",
        "headline": "", "commentary": {},
    })
    out = DOC_ROOT / f"{dataset}.md"
    L: list[str] = []
    L.append(f"# {meta['title']} — Full Pipeline Ablations")
    L.append("")
    n_frames_any = 0
    for det in data:
        for st in data[det]:
            if "all" in data[det][st]:
                n_frames_any = max(n_frames_any, _to_int(data[det][st]["all"].get("n_frames", 0)))
    L.append(f"- **Category:** {meta['category']}")
    L.append(f"- **Scoring:** {meta['scoring'].upper()} @ 0.5")
    L.append(f"- **Frames evaluated (per detector, post-stride):** ~{n_frames_any:,}")
    if meta["headline"]:
        L.append("")
        L.append(meta["headline"])
    L.append("")

    # Overall summary — bbox-level stages first (P/R/F1 only),
    # then segment-level (temporal) stages with TN + FR%.
    L.append("## Overall summary (all sizes) — bbox-level")
    L.append("")
    L.append("Detection-on-drone scoring. Any detection that does not match a "
             "drone GT box (IoU/IoP ≥ 0.5) is an FP — confuser hallucinations "
             "(birds, planes, helis sharing the frame) are already counted here.")
    L.append("")
    L.append("| Model | Stage | TP | FP | FN | P | R | F1 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for detector in sorted(data.keys(), key=lambda d: (modality_of(d) != "rgb", d)):
        for stage in STAGE_ORDER:
            if stage.startswith("temporal"):
                continue
            if stage not in data[detector]:
                continue
            row = data[detector][stage].get("all")
            if row is None:
                continue
            L.append(fmt_row_bbox(detector, stage, row))
    L.append("")

    # Temporal / segment-level sub-table
    has_seg = any(
        s in data[d] for d in data for s in ("temporal", "temporal+alert_gate")
    )
    if has_seg:
        L.append("## Temporal stages — segment-level (3-frame windows, 2-of-3)")
        L.append("")
        L.append("Each row is one 3-frame segment scored as a single binary "
                 "decision: fired ≥ 2 of 3 frames vs. any GT in the window.")
        L.append("")
        L.append("| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for detector in sorted(data.keys(), key=lambda d: (modality_of(d) != "rgb", d)):
            for stage in ("temporal", "temporal+alert_gate"):
                if stage not in data[detector]:
                    continue
                row = data[detector][stage].get("all")
                if row is None:
                    continue
                L.append(fmt_row_segment(detector, stage, row))
        L.append("")

    # Sanity flags
    flags = []
    for detector in data:
        if modality_of(detector) != "rgb":
            continue
        clf = data[detector].get("classifier", {}).get("all")
        rgb_only = data[detector].get("rgb", {}).get("all")
        # IR-side comparison: best R among ir_native / ir_grayscale
        ir_R = 0.0
        for ir_det, key in ((IR_NATIVE, "ir_native"), (IR_GRAY, "ir_grayscale")):
            r = data.get(ir_det, {}).get(key, {}).get("all")
            if r:
                ir_R = max(ir_R, r.get("R", 0))
        if clf and rgb_only:
            max_indiv = max(rgb_only.get("R", 0), ir_R)
            if clf.get("R", 0) + 1e-6 < max_indiv:
                flags.append(f"⚠️  `{detector}`: classifier R={clf['R']:.4f} below "
                             f"max(R_rgb={rgb_only.get('R', 0):.4f}, R_ir={ir_R:.4f})")
    if flags:
        L.append("## Sanity flags")
        L.append("")
        L += flags
        L.append("")

    # Per-size breakdown — one matrix per model
    L.append("## Per-size breakdown")
    L.append("")
    for detector in sorted(data.keys(), key=lambda d: (modality_of(d) != "rgb", d)):
        L.append(f"### {detector} ({modality_of(detector)})")
        L.append("")
        L.append("| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for stage in STAGE_ORDER:
            if stage not in data[detector]:
                continue
            for bucket in ("small", "medium", "large", "all"):
                if bucket not in data[detector][stage]:
                    continue
                if bucket == "all" and not any(b in data[detector][stage] for b in ("small","medium","large")):
                    # Only an all-row (temporal stage) -> already shown in overall
                    pass
                row = data[detector][stage][bucket]
                L.append(fmt_row_persize(stage, bucket, row))
        L.append("")

    # Per-stage commentary
    L.append("## Per-stage commentary")
    L.append("")
    for stage in STAGE_ORDER:
        if stage in meta.get("commentary", {}):
            L.append(f"- **{stage}** — {meta['commentary'][stage]}")
    L.append("")

    # Delivered
    csv_rel = (CSV_DIR / f"{dataset}.csv").relative_to(REPO).as_posix()
    md_rel = out.relative_to(REPO).as_posix()
    L.append("## Delivered")
    L.append("")
    L.append(f"- `{md_rel}`")
    L.append(f"- `{csv_rel}`")
    L.append("")

    out.write_text("\n".join(L), encoding="utf-8")
    return out


# ── Entry point ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scoring", default=None,
                    help="iou | iop. If unset, inferred from DATASET_META.")
    args = ap.parse_args()

    meta = DATASET_META.get(args.dataset, {})
    scoring = args.scoring or meta.get("scoring", "iop")

    combos = discover_combos(args.dataset)
    if not combos:
        print(f"No summary.csv found under {RESULTS_ROOT / args.dataset}")
        print("Run:")
        print(f"  python eval/eval_full_pipeline_persize.py --datasets {args.dataset} --classifiers sa32 no_classifier")
        sys.exit(2)

    print(f"Found {len(combos)} (detector, classifier) combos for {args.dataset}")
    data = collect(args.dataset)
    csv_path = write_csv(args.dataset, scoring, data)
    md_path = write_markdown(args.dataset, data)
    print(f"  wrote {md_path}")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
