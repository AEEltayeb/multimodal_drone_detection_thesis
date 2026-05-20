"""
11_video_tests_doc.py — Aggregate the 19 drone-detection video-test clips
into a single per-family ablations doc.

Reads (per clip, produced by eval_full_pipeline_singlepass.py):
  docs/analysis/full_pipeline_ablations/raw_results/
      video_<cat>_<clipname>/<detector>/<classifier>/summary.csv

Groups by category prefix in the clip key:
  - video_drone_*           -> "drone" (positive frames, bbox scoring)
  - video_birds_*           -> "birds"        (confuser, frame-level)
  - video_airplanes_*       -> "airplanes"    (confuser, frame-level)
  - video_helicopters_*     -> "helicopters"  (confuser, frame-level)

Writes:
  docs/analysis/full_pipeline_ablations/drone_video_tests.md
  docs/analysis/full_pipeline_ablations/csv/drone_video_tests.csv

Doc layout (summary first, details below):
  1. Header (total clips, frames, scoring)
  2. Drone clips
       a. Aggregate summary across all drone clips: model x stage x per-size
       b. Temporal aggregate
       c. Per-clip drill-down (collapsible block per clip)
  3. Confuser clips (one section per category: birds, airplanes, helicopters)
       a. Aggregate FR%/TN% per (model, stage)
       b. Per-clip drill-down
  4. Sanity flags

Run:
  python analytics/spec_analysis/11_video_tests_doc.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO / "docs" / "analysis" / "full_pipeline_ablations"
RESULTS_ROOT = DOC_ROOT / "raw_results"
CSV_DIR = DOC_ROOT / "csv"

# Reuse the modality / stage mapping from doc 10.
sys.path.insert(0, str(REPO / "analytics" / "spec_analysis"))
# 10_full_pipeline_doc starts with a digit so import via importlib
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "doc10",
    REPO / "analytics" / "spec_analysis" / "10_full_pipeline_doc.py",
)
doc10 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(doc10)
modality_of = doc10.modality_of
detector_stage_label = doc10.detector_stage_label
STAGE_ORDER = doc10.STAGE_ORDER

RGB_DETECTORS = doc10.RGB_DETECTORS


# ── Discovery ───────────────────────────────────────────────────────

def discover_clips() -> dict[str, list[str]]:
    """Return {category: [clip_key, ...]} for video_* datasets present."""
    out: dict[str, list[str]] = defaultdict(list)
    if not RESULTS_ROOT.exists():
        return out
    for ds_dir in sorted(RESULTS_ROOT.iterdir()):
        if not ds_dir.is_dir(): continue
        key = ds_dir.name
        if not key.startswith("video_"):
            continue
        # video_<cat>_<rest>
        parts = key.split("_", 2)
        if len(parts) < 3:
            continue
        cat = parts[1]
        out[cat].append(key)
    return out


def discover_combos_for_clip(clip: str) -> list[tuple[str, str]]:
    out = []
    cd = RESULTS_ROOT / clip
    if not cd.exists(): return out
    for det_dir in sorted(cd.iterdir()):
        if not det_dir.is_dir(): continue
        for clf_dir in sorted(det_dir.iterdir()):
            if not clf_dir.is_dir(): continue
            if (clf_dir / "summary.csv").exists():
                out.append((det_dir.name, clf_dir.name))
    return out


# ── Per-combo CSV ingest ─────────────────────────────────────────────

def _ti(v, d=0):
    try: return int(v)
    except (ValueError, TypeError): return d


def _tf(v, d=0.0):
    try: return float(v)
    except (ValueError, TypeError): return d


def load_combo_rows(clip: str, detector: str, classifier: str) -> list[dict]:
    p = RESULTS_ROOT / clip / detector / classifier / "summary.csv"
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


# ── Aggregation across clips ─────────────────────────────────────────

def aggregate_drone_clips(drone_clips: list[str]) -> dict:
    """Sum TP/FP/FN across clips per (detector, stage_label, size_bucket).

    Returns nested dict:
        data[detector][stage_label][bucket] = {TP, FP, FN, n_gt, n_frames}
    Temporal stages stored at bucket='all' with TN.
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "n_gt": 0, "n_frames": 0})))
    for clip in drone_clips:
        for detector, classifier in discover_combos_for_clip(clip):
            rows = load_combo_rows(clip, detector, classifier)
            # Per-frame count for this clip: max n_frames over rows
            n_frames = max((_ti(r["n_frames"]) for r in rows), default=0)
            for r in rows:
                label = detector_stage_label(detector, r["stage"])
                if label is None: continue
                if label in ("classifier", "classifier→filter") and \
                        modality_of(detector) != "rgb":
                    continue
                if label in ("classifier", "classifier→filter") and classifier == "no_classifier":
                    continue
                if label not in ("classifier", "classifier→filter") and classifier != "no_classifier":
                    # prefer no_classifier file for non-classifier stages
                    continue
                b = r["size_bucket"]
                cell = data[detector][label][b]
                cell["TP"] += _ti(r["TP"])
                cell["FP"] += _ti(r["FP"])
                cell["FN"] += _ti(r["FN"])
                cell["TN"] += _ti(r.get("TN") or 0)
                cell["n_gt"] += _ti(r["n_gt"])
                cell["n_frames"] += n_frames if b == "all" else 0

    # Synthesize "all" bucket for bbox stages (small + medium + large).
    # Temporal stages (S4/S5) already write directly to bucket="all".
    for det, stages in data.items():
        for label, by_b in stages.items():
            if "all" in by_b and label.startswith("temporal"):
                continue
            tp = fp = fn = n_gt = 0
            for b in ("small", "medium", "large"):
                if b not in by_b: continue
                tp += by_b[b]["TP"]; fp += by_b[b]["FP"]; fn += by_b[b]["FN"]
                n_gt += by_b[b]["n_gt"]
            if tp + fp + fn + n_gt == 0 and "all" not in by_b:
                continue
            by_b["all"] = {"TP": tp, "FP": fp, "FN": fn, "TN": 0,
                           "n_gt": n_gt, "n_frames": 0}
    return data


def aggregate_confuser_clips(clips: list[str]) -> dict:
    """For confuser-only clips, derive frame-level FR%/TN% per (detector, stage).

    The S0/S3 rows have all-zero TP/FN (no GT). The total FP across size
    buckets bounds "frames where the model fired" only loosely (multiple
    boxes per frame). For frame-level we read the temporal S4/S5 rows
    (3-frame segments) which DO carry frame-level binary TP/FP/FN/TN, then
    upscale to per-frame:
        FR_frame = (segments_fired) / (segments_total), interpreted as the
                   fraction of *segments* with at least one fire. For a
                   confuser dataset (no positive segments), segments_fired
                   is the FP segment count.
    We additionally synthesize a single-frame approximation by collapsing
    S0_detector counts: a frame is considered to have "fired" if it
    contributed at least one box. We don't have per-frame booleans in the
    CSV; instead we report what the CSV does carry:
        - S0/S3: total FP boxes (call it "FP_boxes"), n_frames
        - S4/S5: segments_fired, segments_total, segment_FR%

    Returns:
        data[detector][stage_label] = {
           "FP_boxes": int,        # only meaningful for non-temporal stages
           "n_frames": int,
           "segments_fired": int,  # only for temporal stages
           "segments_total": int,
           "n_clips": int,
        }
    """
    data: dict = defaultdict(lambda: defaultdict(
        lambda: {"FP_boxes": 0, "n_frames": 0,
                 "segments_fired": 0, "segments_total": 0, "n_clips": 0}))
    for clip in clips:
        for detector, classifier in discover_combos_for_clip(clip):
            rows = load_combo_rows(clip, detector, classifier)
            n_frames = max((_ti(r["n_frames"]) for r in rows
                            if r["size_bucket"] != "all"), default=0)
            seen_labels: set[str] = set()
            for r in rows:
                label = detector_stage_label(detector, r["stage"])
                if label is None: continue
                if label in ("classifier", "classifier→filter") and \
                        modality_of(detector) != "rgb":
                    continue
                if label in ("classifier", "classifier→filter") and classifier == "no_classifier":
                    continue
                if label not in ("classifier", "classifier→filter") and classifier != "no_classifier":
                    continue
                b = r["size_bucket"]
                cell = data[detector][label]
                if b != "all":
                    cell["FP_boxes"] += _ti(r["FP"])
                else:
                    # Temporal/segment row
                    cell["segments_fired"] += _ti(r["FP"])  # confuser: FP segs = fired segs
                    cell["segments_total"] += _ti(r["n_frames"])  # n_frames here is n_segments
                seen_labels.add(label)
            for lab in seen_labels:
                data[detector][lab]["n_frames"] += n_frames
                data[detector][lab]["n_clips"] += 1
    return data


# ── Output: markdown ─────────────────────────────────────────────────

def fmt_drone_bbox_row(detector: str, stage: str, cell: dict) -> str:
    tp = cell["TP"]; fp = cell["FP"]; fn = cell["FN"]
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return f"| {detector} | {stage} | {tp:,} | {fp:,} | {fn:,} | {P:.4f} | {R:.4f} | {F:.4f} |"


def fmt_drone_segment_row(detector: str, stage: str, cell: dict) -> str:
    tp = cell["TP"]; fp = cell["FP"]; fn = cell["FN"]; tn = cell["TN"]
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    n = tp + fp + fn + tn
    fr = 100.0 * (tp + fp) / n if n else 0.0
    return (f"| {detector} | {stage} | {tp:,} | {fp:,} | {fn:,} | {tn:,} | "
            f"{P:.4f} | {R:.4f} | {F:.4f} | {fr:.2f}% |")


def fmt_confuser_bbox_row(detector: str, stage: str, cell: dict) -> str:
    """Confuser single-frame stage: FP-box count + boxes/frame proxy."""
    fp = cell["FP_boxes"]
    n_frames = cell["n_frames"]
    bpf = fp / n_frames if n_frames else 0.0
    return (f"| {detector} | {stage} | {cell['n_clips']} | {n_frames:,} | "
            f"{fp:,} | {bpf:.3f} |")


def fmt_confuser_segment_row(detector: str, stage: str, cell: dict) -> str:
    """Confuser temporal stage: segment-level FR% / TN%."""
    n_seg = cell["segments_total"]
    fired = cell["segments_fired"]
    fr = 100.0 * fired / n_seg if n_seg else 0.0
    tn_seg = n_seg - fired
    tn_pct = 100.0 * tn_seg / n_seg if n_seg else 100.0
    return (f"| {detector} | {stage} | {cell['n_clips']} | {n_seg} | "
            f"{fired} | {tn_seg} | {fr:.2f}% | {tn_pct:.2f}% |")


def emit_drone_section(L: list[str], drone_data: dict):
    L.append("## Drone clips (bbox-level scoring)")
    L.append("")
    L.append("Aggregated across all drone clips (TP/FP/FN summed, P/R/F1 recomputed).")
    L.append("")
    L.append("| Model | Stage | TP | FP | FN | P | R | F1 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    detectors = sorted(drone_data.keys(), key=lambda d: (modality_of(d) != "rgb", d))
    for det in detectors:
        for stage in STAGE_ORDER:
            if stage not in drone_data[det]: continue
            if stage.startswith("temporal"): continue
            cell = drone_data[det][stage].get("all")
            if not cell: continue
            L.append(fmt_drone_bbox_row(det, stage, cell))
    L.append("")
    L.append("### Temporal (3-frame segments, 2-of-3)")
    L.append("")
    L.append("| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for det in detectors:
        for stage in ("temporal", "temporal+alert_gate"):
            if stage not in drone_data[det]: continue
            cell = drone_data[det][stage].get("all")
            if not cell: continue
            L.append(fmt_drone_segment_row(det, stage, cell))
    L.append("")


def emit_confuser_section(L: list[str], cat: str, conf_data: dict):
    L.append(f"## {cat.title()} clips (no drone GT — every detection is FP)")
    L.append("")
    L.append("Two views: single-frame stages report total FP box count and "
             "boxes-per-frame (a precision proxy when no positive frames exist); "
             "temporal stages report segment-level fire rate / true-negative rate.")
    L.append("")
    L.append("### Single-frame stages")
    L.append("")
    L.append("| Model | Stage | clips | frames | FP boxes | boxes/frame |")
    L.append("|---|---|---:|---:|---:|---:|")
    detectors = sorted(conf_data.keys(), key=lambda d: (modality_of(d) != "rgb", d))
    for det in detectors:
        for stage in STAGE_ORDER:
            if stage.startswith("temporal"): continue
            if stage not in conf_data[det]: continue
            cell = conf_data[det][stage]
            L.append(fmt_confuser_bbox_row(det, stage, cell))
    L.append("")
    L.append("### Temporal stages (3-frame segments, 2-of-3)")
    L.append("")
    L.append("| Model | Stage | clips | segments | seg fired (FP) | seg quiet (TN) | FR% | TN% |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for det in detectors:
        for stage in ("temporal", "temporal+alert_gate"):
            if stage not in conf_data[det]: continue
            cell = conf_data[det][stage]
            L.append(fmt_confuser_segment_row(det, stage, cell))
    L.append("")


# ── Output: long-form CSV ────────────────────────────────────────────

def write_csv(drone_data: dict, conf_per_cat: dict[str, dict]) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_DIR / "drone_video_tests.csv"
    fieldnames = ["dataset", "category", "model", "modality", "stage",
                  "size_bucket", "TP", "FP", "FN", "TN", "P", "R", "F1",
                  "FR_pct", "TN_pct", "n_gt", "n_frames", "n_clips",
                  "FP_boxes", "segments_fired", "segments_total", "flag"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # Drone bbox-level rows
        for det in sorted(drone_data.keys()):
            for stage in STAGE_ORDER:
                if stage not in drone_data[det]: continue
                for bucket, cell in drone_data[det][stage].items():
                    tp, fp, fn, tn = cell["TP"], cell["FP"], cell["FN"], cell.get("TN", 0)
                    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                    flag = "n_gt_zero" if cell["n_gt"] == 0 and bucket != "all" else ""
                    n_seg = tp + fp + fn + tn
                    fr = 100.0 * (tp + fp) / n_seg if (n_seg and stage.startswith("temporal")) else ""
                    w.writerow({
                        "dataset": "drone_video_tests", "category": "drone",
                        "model": det, "modality": modality_of(det),
                        "stage": stage, "size_bucket": bucket,
                        "TP": tp, "FP": fp, "FN": fn,
                        "TN": tn if stage.startswith("temporal") else "",
                        "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                        "FR_pct": fr, "TN_pct": "",
                        "n_gt": cell["n_gt"], "n_frames": cell.get("n_frames", 0),
                        "n_clips": "", "FP_boxes": "", "segments_fired": "",
                        "segments_total": "", "flag": flag,
                    })

        # Confuser frame-level rows
        for cat, conf_data in conf_per_cat.items():
            for det in sorted(conf_data.keys()):
                for stage in STAGE_ORDER:
                    if stage not in conf_data[det]: continue
                    cell = conf_data[det][stage]
                    n_seg = cell["segments_total"]
                    fired = cell["segments_fired"]
                    fr = 100.0 * fired / n_seg if n_seg else 0.0
                    tn_pct = 100.0 * (n_seg - fired) / n_seg if n_seg else 100.0
                    w.writerow({
                        "dataset": "drone_video_tests", "category": cat,
                        "model": det, "modality": modality_of(det),
                        "stage": stage, "size_bucket": "all",
                        "TP": "", "FP": "", "FN": "", "TN": "",
                        "P": "", "R": "", "F1": "",
                        "FR_pct": round(fr, 2), "TN_pct": round(tn_pct, 2),
                        "n_gt": "", "n_frames": cell["n_frames"],
                        "n_clips": cell["n_clips"],
                        "FP_boxes": cell["FP_boxes"],
                        "segments_fired": fired, "segments_total": n_seg,
                        "flag": "",
                    })
    return out


# ── Entry ────────────────────────────────────────────────────────────

def main():
    cats = discover_clips()
    if not cats:
        print(f"No video_* datasets under {RESULTS_ROOT}.")
        print("Run:  python run_video_tests.py")
        sys.exit(2)

    drone_clips = cats.get("drone", [])
    confuser_cats = {c: cats[c] for c in ("birds", "airplanes", "helicopters") if c in cats}
    n_total = sum(len(v) for v in cats.values())
    print(f"Discovered {n_total} clips: {len(drone_clips)} drone, "
          f"{sum(len(v) for v in confuser_cats.values())} confuser")

    drone_data = aggregate_drone_clips(drone_clips) if drone_clips else {}
    conf_per_cat: dict[str, dict] = {}
    for cat, clips in confuser_cats.items():
        conf_per_cat[cat] = aggregate_confuser_clips(clips)

    # Markdown
    L: list[str] = []
    L.append("# Drone-detection video tests — Full Pipeline Ablations")
    L.append("")
    L.append("- **Category:** RGB videos, mixed drone + confuser scenes")
    L.append("- **Scoring:** IoP @ 0.5 (drone clips). Confuser clips use frame-level FR%/TN% (no GT).")
    L.append(f"- **Drone clips:** {len(drone_clips)}")
    L.append(f"- **Confuser clips:** {sum(len(v) for v in confuser_cats.values())} "
             f"({', '.join(f'{c}={len(v)}' for c, v in confuser_cats.items())})")
    L.append("")
    L.append("**Read order:** drone-clip aggregate first (the headline numbers), "
             "then per-confuser-category aggregates, then per-clip drill-downs at the end.")
    L.append("")

    if drone_data:
        emit_drone_section(L, drone_data)
    for cat, conf_data in conf_per_cat.items():
        emit_confuser_section(L, cat, conf_data)

    # Per-clip drill-down (compact)
    L.append("## Per-clip drill-down")
    L.append("")
    L.append("Each clip's summary table is in `raw_results/<clip_key>/<detector>/<classifier>/summary.csv`. "
             "Compact preview below — `frames` is the clip length; for drone clips `n_gt` is the total "
             "drone GT box count.")
    L.append("")
    for cat, clips in cats.items():
        L.append(f"### {cat}")
        L.append("")
        L.append("| Clip | Frames | n_gt | best F1 (drone) or lowest FR% (confuser) |")
        L.append("|---|---:|---:|---|")
        for clip in clips:
            combos = discover_combos_for_clip(clip)
            best = "—"
            best_n_frames = 0
            best_n_gt = 0
            best_score: float | None = None
            for det, clf in combos:
                rows = load_combo_rows(clip, det, clf)
                if not rows: continue
                # all-row for S0_detector
                all_rows = [r for r in rows
                            if r["stage"] == "S0_detector" and r["size_bucket"] == "all"]
                if not all_rows:
                    # synthesize from per-size sum
                    s = [r for r in rows if r["stage"] == "S0_detector"]
                    if not s: continue
                    tp = sum(_ti(r["TP"]) for r in s)
                    fp = sum(_ti(r["FP"]) for r in s)
                    fn = sum(_ti(r["FN"]) for r in s)
                    n_gt = sum(_ti(r["n_gt"]) for r in s)
                    n_frames = _ti(s[0]["n_frames"])
                else:
                    r = all_rows[0]
                    tp, fp, fn = _ti(r["TP"]), _ti(r["FP"]), _ti(r["FN"])
                    n_gt = _ti(r["n_gt"])
                    n_frames = _ti(r["n_frames"])
                best_n_frames = max(best_n_frames, n_frames)
                best_n_gt = max(best_n_gt, n_gt)
                if cat == "drone":
                    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                    if best_score is None or F > best_score:
                        best_score = F; best = f"{det}/{clf}: F1={F:.4f}"
                else:
                    # confuser: lower fire rate is better
                    fr = (tp + fp) / n_frames if n_frames else 0.0  # boxes/frame proxy
                    if best_score is None or fr < best_score:
                        best_score = fr; best = f"{det}/{clf}: {tp + fp} FP boxes ({fr:.2f}/frame)"
            L.append(f"| {clip} | {best_n_frames} | {best_n_gt} | {best} |")
        L.append("")

    md_path = DOC_ROOT / "drone_video_tests.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    csv_path = write_csv(drone_data, conf_per_cat)
    print(f"  wrote {md_path}")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
