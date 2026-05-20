"""
09_fill_ablations_doc.py — Build full_pipeline_ablations.md (compact format).

Layout: horizontal stages per detector (A), per-clip detail moved to companion
CSV (C), 1-paragraph prose summary leading each section (E).

Pulls from canonical sources:
  - eval/results/pipeline_video_tests*/pipeline_comparison.csv
  - eval/results/antiuav/metrics_iop.csv, svanstrom/metrics_iop.csv
  - eval/results/antiuav_per_model/<m>/<m>_results.json
  - eval/results/selcom_val_holdout/<m>/<m>_results.json
  - eval/results/svanstrom_persize/summary.csv
  - eval/results/video_persize/summary.csv
  - eval/results/roboflow_ood/summary.csv
  - eval/results/full_pipeline_persize/<ds>/<det>/<clf>/summary.csv (when present)

All P/R/F1 are IoP@0.5. `ir_grayscale` rows tagged with †; legacy IoU
disclosure in the methodology block. Re-run after any new canonical CSV.
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "eval" / "results"
DOC_DIR = REPO / "docs" / "analysis" / "full_pipeline_ablations"
DOC = DOC_DIR / "full_pipeline_ablations.md"
PER_CLIP_CSV = DOC_DIR / "per_clip_detail.csv"
PER_SIZE_CSV = DOC_DIR / "per_size_detail.csv"


# ── Math ───────────────────────────────────────────────────────────

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f


def fmt_f(x):
    if x is None or x == "":
        return "—"
    return f"{x:.3f}"


# ── Loaders ────────────────────────────────────────────────────────

def load_pipeline_video_tests(variant: str = "") -> dict:
    dir_name = "pipeline_video_tests" + (f"_{variant}" if variant else "")
    p = RES / dir_name / "pipeline_comparison.csv"
    out: dict = {}
    if not p.exists(): return out
    with p.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["dataset"], {})[r["rgb_model"]] = r
    return out


def load_phase2_results(path: Path) -> dict | None:
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except: return None


def p2_iop(d: dict):
    dm = d.get("detection_metrics", [])
    m = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
    return int(m.get("TP", 0)), int(m.get("FP", 0)), int(m.get("FN", 0))


def p2_per_size(d: dict):
    return d.get("per_size_metrics", {}).get("iop", {})


def load_full_pipeline_persize_combo(ds: str, det: str, clf: str = "no_classifier") -> dict | None:
    """Load full_pipeline_persize summary.csv for one combo if present.
    Returns {stage: {size: {tp,fp,fn}}}. Else None."""
    p = RES / "full_pipeline_persize" / ds / det / clf / "summary.csv"
    if not p.exists(): return None
    out: dict = defaultdict(lambda: defaultdict(lambda: {"tp":0,"fp":0,"fn":0}))
    with p.open() as f:
        for r in csv.DictReader(f):
            out[r["stage"]][r["size_bucket"]] = {
                "tp": int(r.get("TP", 0) or 0),
                "fp": int(r.get("FP", 0) or 0),
                "fn": int(r.get("FN", 0) or 0),
            }
    return dict(out)


def cascade_f1(combo_data: dict | None, stage_key: str) -> tuple[float, int, int, int] | None:
    """Aggregate (sum over sizes) for given stage from combo data."""
    if combo_data is None: return None
    st = combo_data.get(stage_key)
    if not st: return None
    tp = sum(v["tp"] for k, v in st.items() if k in ("small","medium","large"))
    fp = sum(v["fp"] for k, v in st.items() if k in ("small","medium","large"))
    fn = sum(v["fn"] for k, v in st.items() if k in ("small","medium","large"))
    if tp + fp + fn == 0:
        # Try "all" bucket (for segment stages)
        a = st.get("all")
        if a:
            tp, fp, fn = a["tp"], a["fp"], a["fn"]
        else:
            return None
    p, r, f = prf(tp, fp, fn)
    return f, tp, fp, fn


# ── Horizontal-cascade row builder ─────────────────────────────────

CASCADE_STAGES = [
    ("rgb_only",          "S0_detector"),
    ("+classifier",       "S1_+classifier"),
    ("+filter",           "S3_+patch_only"),
    ("+temporal",         "S4_temporal_no_filter"),
    ("+alert_gate",       "S5_alert_gate_filter"),
]


def cascade_row(detector_label: str, combo: dict | None, baseline_for_delta: float | None = None) -> tuple[str, float | None]:
    """Returns markdown row + final-stage F1 for delta computation."""
    cells = [detector_label]
    f1_per_stage = []
    for col_label, stage_key in CASCADE_STAGES:
        r = cascade_f1(combo, stage_key)
        if r is None:
            cells.append("—")
            f1_per_stage.append(None)
        else:
            f1_val = r[0]
            cells.append(f"{f1_val:.3f}")
            f1_per_stage.append(f1_val)
    # Delta column: final_stage F1 - rgb_only F1 (or baseline if provided)
    base = f1_per_stage[0] if f1_per_stage[0] is not None else baseline_for_delta
    final = f1_per_stage[-1] if f1_per_stage[-1] is not None else f1_per_stage[-2]
    if base is not None and final is not None:
        delta = final - base
        cells.append(f"{delta:+.3f}")
    else:
        cells.append("—")
    return "| " + " | ".join(cells) + " |", f1_per_stage[-1]


# ── Sections ───────────────────────────────────────────────────────

def hdr_cascade() -> str:
    cols = ["Detector"] + [c for c, _ in CASCADE_STAGES] + ["Δ vs rgb_only"]
    sep = "|" + "|".join(["---"] + [":---:"] * (len(cols)-2) + [":---:"]) + "|"
    return "| " + " | ".join(cols) + " |\n" + sep


def section_header() -> str:
    out = ["# Full pipeline ablations\n"]
    out.append("Compact cascade ablation across surfaces. **All metrics are IoP@0.5.** "
               "Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv); "
               "per-size detail in [`per_size_detail.csv`](per_size_detail.csv).\n")
    out.append("## Reading the tables\n")
    out.append("Each row is a detector. Each column is a pipeline step applied on top of the previous.\n"
               "- `rgb_only` — RGB YOLO alone\n"
               "- `+classifier` — scene-aware classifier (default sa32) gates dets via a trust label; any trust ≠ 0 passes both RGB and IR-grayscale dets where coords align\n"
               "- `+filter` — patch verifier applied to detector output (no classifier)\n"
               "- `+temporal` — 2-of-3 vote over 3-frame segments on detector dets\n"
               "- `+alert_gate` — temporal vote with patch filter applied at alert (production cascade endpoint)\n"
               "- `Δ vs rgb_only` — F1 gain from the full cascade vs detector alone\n")
    out.append("## Scoring rule and `ir_grayscale` †\n")
    out.append("IoP@0.5 throughout. RGB models move ≤1 pp F1 under IoU; only `ir_grayscale` moves meaningfully "
               "(legacy IoU aggregate on 9 real-video drone clips: P=0.588, R=0.441, F1=0.504). "
               "`ir_grayscale` rows marked †. See `docs/EVIDENCE_LEDGER.md` §12 for canonical numbers.\n")
    return "\n".join(out)


def section_antiuav() -> str:
    out = ["\n## 1. Anti-UAV RGBT (paired drone, clean benchmark)\n"]
    # Prose summary
    out.append("**Summary.** Saturated benchmark (rgb-only F1 ≈ 0.99 for baseline/retrained_v2/selcom_640). "
               "Cascade has nothing to do here — no confusers, no clutter. `selcom_1280` bleeds 849 FPs and "
               "drops to F1=0.90, the only model not at the ceiling. Cascade columns sourced from "
               "`full_pipeline_persize` (currently rgb_only + classifier + filter only; temporal pending data).\n")
    out.append(hdr_cascade())
    models = ["baseline", "retrained_v2", "selcom_640", "selcom_960", "selcom_1280"]
    for m in models:
        combo = load_full_pipeline_persize_combo("antiuav", m, "sa32") or \
                load_full_pipeline_persize_combo("antiuav", m, "no_classifier")
        if combo is None:
            # Fall back to phase 2 results.json for rgb_only
            d = load_phase2_results(RES / "antiuav_per_model" / m / f"{m}_results.json")
            if d:
                tp, fp, fn = p2_iop(d)
                _, _, f1 = prf(tp, fp, fn)
                out.append(f"| {m} | {f1:.3f} | — | — | — | — | — |")
            continue
        row, _ = cascade_row(m, combo)
        out.append(row)
    out.append("\n*Legacy 2-config aggregate (no per-model breakdown):*  rgb_only F1=0.992, ir_only F1=0.965 (from `eval/results/antiuav/metrics_iop.csv`).")
    return "\n".join(out)


def section_svanstrom() -> str:
    out = ["\n## 2. Svanström (paired drone + confusers)\n"]
    out.append("**Summary.** RGB-only collapses under confusers (F1=0.54); IR alone is stable (F1=0.96). "
               "Classifier+filter combination is where the cascade earns its keep on this surface — when the "
               "in-progress run lands, expect cascade F1 to recover toward IR-only levels.\n")
    out.append(hdr_cascade())
    models = ["baseline", "hardneg_v3more", "retrained_v2", "selcom_640", "selcom_960", "selcom_1280"]
    for m in models:
        combo = load_full_pipeline_persize_combo("svanstrom", m, "sa32") or \
                load_full_pipeline_persize_combo("svanstrom", m, "no_classifier")
        if combo:
            row, _ = cascade_row(m, combo)
            out.append(row)
            continue
        # Fallback: svanstrom_persize for rgb_only only
        sp = RES / "svanstrom_persize" / f"{m}_persize.csv"
        if sp.exists():
            tp = fp = fn = 0
            with sp.open() as f:
                for r in csv.DictReader(f):
                    if r.get("category") != "DRONE": continue
                    tp += int(r.get("TP", 0) or 0)
                    fp += int(r.get("FP", 0) or 0)
                    fn += int(r.get("FN", 0) or 0)
            if tp + fp + fn > 0:
                _, _, f1 = prf(tp, fp, fn)
                out.append(f"| {m} | {f1:.3f} | — | — | — | — | — |")
    out.append("\n*Legacy 2-config:*  rgb_only F1=0.544 (collapses under confusers), ir_only F1=0.959.")
    return "\n".join(out)


def section_selcom() -> str:
    out = ["\n## 3. Selcom held-out val (RGB only, 311 imgs)\n"]
    out.append("**Summary.** Detector-only winner: `selcom_960` (F1=0.585) edging `selcom_1280` (F1=0.580). "
               "Classifier S1 drops recall on this surface — the scene-aware classifier was trained on "
               "Svanström-like data and doesn't recognize CCTV signal, so it conservatively rejects. "
               "Cascade not the right tool here; rgb_only is the correct reporting baseline.\n")
    out.append(hdr_cascade())
    models = ["baseline", "hardneg_v3more", "retrained_v2", "selcom_640", "selcom_960", "selcom_1280"]
    for m in models:
        combo = load_full_pipeline_persize_combo("selcom_val", m, "sa32") or \
                load_full_pipeline_persize_combo("selcom_val", m, "no_classifier")
        if combo:
            row, _ = cascade_row(m, combo)
            out.append(row)
            continue
        d = load_phase2_results(RES / "selcom_val_holdout" / m / f"{m}_results.json")
        if d:
            tp, fp, fn = p2_iop(d)
            _, _, f1 = prf(tp, fp, fn)
            out.append(f"| {m} | {f1:.3f} | — | — | — | — | — |")
    return "\n".join(out)


def section_roboflow() -> str:
    out = ["\n## 4. Roboflow OOD drone (RGB only)\n"]
    out.append("**Summary.** `selcom_960` Pareto-best (rgb+filter F1=0.84). Cascade stages beyond +filter not "
               "evaluated on this surface — gap noted.\n")
    out.append(hdr_cascade())
    models = ["baseline", "hardneg_v3more", "retrained_v2", "selcom_640", "selcom_960", "selcom_1280"]
    for m in models:
        combo = load_full_pipeline_persize_combo("roboflow_rgb_drone_test", m, "sa32") or \
                load_full_pipeline_persize_combo("roboflow_rgb_drone_test", m, "no_classifier")
        if combo:
            row, _ = cascade_row(m, combo)
            out.append(row)
            continue
        # Fallback: sum across splits from roboflow_ood/summary.csv
        p = RES / "roboflow_ood" / "summary.csv"
        if p.exists():
            tp_r = fp_r = fn_r = tp_f = fp_f = fn_f = 0
            model_label = f"rgb_{m}"
            with p.open() as f:
                for r in csv.DictReader(f):
                    if r["model"] != model_label: continue
                    if not r["dataset"].startswith("rgb_drone"): continue
                    tp_r += int(r.get("raw_TP", 0) or 0); fp_r += int(r.get("raw_FP", 0) or 0); fn_r += int(r.get("raw_FN", 0) or 0)
                    tp_f += int(r.get("filt_TP", 0) or 0); fp_f += int(r.get("filt_FP", 0) or 0); fn_f += int(r.get("filt_FN", 0) or 0)
            if tp_r + fp_r + fn_r > 0:
                _, _, f1_r = prf(tp_r, fp_r, fn_r)
                _, _, f1_f = prf(tp_f, fp_f, fn_f) if tp_f+fp_f+fn_f else (0,0,0)
                delta = f1_f - f1_r if tp_f+fp_f+fn_f else 0
                out.append(f"| {m} | {f1_r:.3f} | — | {f1_f:.3f} | — | — | {delta:+.3f} |")
    return "\n".join(out)


def section_real_video_drone() -> str:
    out = ["\n## 5. Real-video drone clips (RGB only, 9 clips)\n"]
    out.append("**Summary.** This is where the cascade story is clearest: baseline goes from rgb_only F1=0.76 "
               "to +temporal F1=0.83 (+7 pp) — segment-level voting recovers single-frame recall noise. "
               "Patch filter at alert gate keeps the recall while cutting FPs.\n")
    out.append("\n### 5a. Cascade per detector (sa32 classifier, aggregated across drone clips)\n")

    sa32 = load_pipeline_video_tests("")
    drone_clips = sorted({c for c in sa32 if any(sa32[c][m].get("category") == "drone" for m in sa32[c])})
    models_in_pvt = ["baseline_trained", "retrained_v2", "selcom_1280", "selcom_640"]

    out.append(hdr_cascade())
    stage_keys = [("rgb", "rgb_only"), ("clf", "+classifier"), ("rgb", "+filter (n/a, no rgb+filter in pvt)"),
                  ("seg_temp", "+temporal"), ("seg_final", "+alert_gate")]
    # Override: pipeline_video_tests has no "+filter only" column; show "—" for it
    for m in models_in_pvt:
        cells = [m]
        f1s = []
        for prefix, _ in [("rgb","rgb_only"), ("clf","+cls"), (None,"+filter"), ("seg_temp","+temp"), ("seg_final","+gate")]:
            if prefix is None:
                cells.append("—"); f1s.append(None); continue
            tp = fp = fn = 0
            for clip in drone_clips:
                row = sa32.get(clip, {}).get(m)
                if not row: continue
                tp += int(row.get(f"{prefix}_tp", 0) or 0)
                fp += int(row.get(f"{prefix}_fp", 0) or 0)
                fn += int(row.get(f"{prefix}_fn", 0) or 0)
            if tp + fp + fn == 0:
                cells.append("—"); f1s.append(None)
            else:
                _, _, f = prf(tp, fp, fn)
                cells.append(f"{f:.3f}"); f1s.append(f)
        # Delta = final - rgb_only
        base = f1s[0]; final = f1s[-1] or f1s[-2]
        cells.append(f"{(final - base):+.3f}" if (base is not None and final is not None) else "—")
        out.append("| " + " | ".join(cells) + " |")

    out.append("\n### 5b. Three-classifier endpoint F1 (`+alert_gate` stage)\n")
    out.append("| Detector | sa32 | control40 | fnfn |\n|---|:---:|:---:|:---:|")
    c40 = load_pipeline_video_tests("control40")
    fnfn = load_pipeline_video_tests("fusionnofn")
    for m in models_in_pvt:
        f1_row = [m]
        for table in (sa32, c40, fnfn):
            tp = fp = fn = 0
            for clip in drone_clips:
                row = table.get(clip, {}).get(m)
                if not row: continue
                tp += int(row.get("seg_final_tp", 0) or 0)
                fp += int(row.get("seg_final_fp", 0) or 0)
                fn += int(row.get("seg_final_fn", 0) or 0)
            if tp+fp+fn == 0:
                f1_row.append("—")
            else:
                _, _, f = prf(tp, fp, fn)
                f1_row.append(f"{f:.3f}")
        out.append("| " + " | ".join(f1_row) + " |")
    out.append("\n**Read:** sa32 is the production pick. control40 trades 18+ pp F1 for halved FPs; "
               "fnfn rejects 85% of correct TPs — only viable when false-alarm fatigue dominates.\n")

    out.append("\n*Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv).*")
    return "\n".join(out)


def section_confuser() -> str:
    out = ["\n## 6. Confuser-only clips (no drone GT)\n"]
    out.append("**Summary.** Cascade FPPI by detector × stage, aggregated per confuser category. "
               "Lower is better. Watch the cascade columns ↓ left-to-right — that's the FP reduction story.\n")

    sa32 = load_pipeline_video_tests("")
    cats = {"airplanes", "birds", "helicopters"}
    confuser_clips_by_cat: dict[str, list[str]] = {c: [] for c in cats}
    for clip, models in sa32.items():
        for m, row in models.items():
            if row.get("category") in cats:
                confuser_clips_by_cat[row["category"]].append(clip)
                break
    for c in cats:
        confuser_clips_by_cat[c] = sorted(set(confuser_clips_by_cat[c]))

    out.append("| Category | Detector | rgb_only FPPI | +classifier | +temporal | +alert_gate | Δ |\n|---|---|:---:|:---:|:---:|:---:|:---:|")
    stages_pvt = [("rgb", "total_frames"), ("clf", "total_frames"), ("seg_temp", "seg_temp_segs"), ("seg_final", "seg_final_segs")]
    models_in_pvt = ["baseline_trained", "retrained_v2", "selcom_1280", "selcom_640"]
    for cat in ("birds", "airplanes", "helicopters"):
        for m in models_in_pvt:
            cells = [cat, m]
            fppis = []
            for prefix, nframes_col in stages_pvt:
                fp_sum = 0; nf_sum = 0
                for clip in confuser_clips_by_cat[cat]:
                    row = sa32.get(clip, {}).get(m)
                    if not row: continue
                    fp_sum += int(row.get(f"{prefix}_fp", 0) or 0)
                    nf_sum += int(row.get(nframes_col, 0) or 0)
                if nf_sum == 0:
                    cells.append("—"); fppis.append(None)
                else:
                    fppi = fp_sum / nf_sum
                    cells.append(f"{fppi:.4f}"); fppis.append(fppi)
            if fppis[0] is not None and fppis[-1] is not None:
                delta = fppis[-1] - fppis[0]
                cells.append(f"{delta:+.4f}")
            else:
                cells.append("—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("|  |  |  |  |  |  |  |")
    out = out[:-1]  # trim trailing separator
    out.append("\n*Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv).*\n")
    return "\n".join(out)


# ── Companion CSVs ─────────────────────────────────────────────────

def write_per_clip_csv():
    sa32 = load_pipeline_video_tests("")
    c40 = load_pipeline_video_tests("control40")
    fnfn = load_pipeline_video_tests("fusionnofn")
    rows = []
    for tag, table in (("sa32", sa32), ("control40", c40), ("fnfn", fnfn)):
        for clip, models in table.items():
            for m, row in models.items():
                rows.append({"classifier": tag, "clip": clip, "category": row.get("category",""),
                             "detector": m, **{k: row.get(k, "") for k in row if k not in ("dataset","rgb_model","category")}})
    if not rows: return
    fields = list(rows[0].keys())
    with PER_CLIP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_per_size_csv():
    rows = []
    # Real-video per-size
    p = RES / "video_persize" / "summary.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                if r["size_bucket"] == "all": continue
                rows.append({"source": "video_persize", "dataset": f"video_{r['category']}_{r['clip']}",
                             "detector": r["model"], "stage": "S0_detector",
                             "size_bucket": r["size_bucket"], **{k: r.get(k, "") for k in ("TP","FP","FN","n_gt","n_frames","precision","recall","f1")}})
    # Svanstrom per-size
    p = RES / "svanstrom_persize" / "summary.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("category") != "DRONE": continue
                rows.append({"source": "svanstrom_persize", "dataset": "svanstrom",
                             "detector": r["model"], "stage": "S0_detector",
                             "size_bucket": r["size_bucket"], **{k: r.get(k, "") for k in ("TP","FP","FN","n_gt","precision","recall","f1")}})
    # Phase 2 per-model per-size
    for ds_dir, ds_key in [("antiuav_per_model","antiuav"), ("selcom_val_holdout","selcom_val")]:
        for m_dir in (RES / ds_dir).glob("*/"):
            m = m_dir.name
            d = load_phase2_results(m_dir / f"{m}_results.json")
            if not d: continue
            for size, vals in p2_per_size(d).items():
                tp = vals.get("tp", 0); fp = vals.get("fp", 0); fn = vals.get("fn", 0)
                if tp + fp + fn == 0: continue
                p_, r_, f1 = prf(tp, fp, fn)
                rows.append({"source": ds_dir, "dataset": ds_key, "detector": m, "stage": "S0_detector",
                             "size_bucket": size, "TP": tp, "FP": fp, "FN": fn,
                             "precision": f"{p_:.4f}", "recall": f"{r_:.4f}", "f1": f"{f1:.4f}"})
    # full_pipeline_persize per-size cascade (when available)
    root = RES / "full_pipeline_persize"
    for sf in root.rglob("summary.csv"):
        parts = sf.relative_to(root).parts
        if len(parts) != 4: continue
        ds, det, clf, _ = parts
        with sf.open() as f:
            for r in csv.DictReader(f):
                if r["size_bucket"] == "all" and r["stage"] not in ("S4_temporal_no_filter","S5_alert_gate_filter"):
                    continue
                rows.append({"source": "full_pipeline_persize", "dataset": ds, "detector": det,
                             "stage": r["stage"], "classifier": clf,
                             "size_bucket": r["size_bucket"], **{k: r.get(k, "") for k in ("TP","FP","FN","precision","recall","f1")}})
    if not rows: return
    # Union of fieldnames
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); all_keys.append(k)
    with PER_SIZE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader(); w.writerows(rows)


def main():
    parts = []
    parts.append(section_header())
    parts.append(section_antiuav())
    parts.append(section_svanstrom())
    parts.append(section_selcom())
    parts.append(section_roboflow())
    parts.append(section_real_video_drone())
    parts.append(section_confuser())

    parts.append("\n## Reproduction\n")
    parts.append("```\n# Doc:\npython analytics/spec_analysis/09_fill_ablations_doc.py\n"
                 "# Per-size cascade gap-fill (in progress):\n"
                 "python eval/eval_full_pipeline_persize.py --classifiers sa32 no_classifier\n```\n")
    parts.append("Canonical sources used:\n"
                 "- `eval/results/antiuav/metrics_iop.csv`, `eval/results/svanstrom/metrics_iop.csv` (legacy aggregate)\n"
                 "- `eval/results/{antiuav_per_model, selcom_val_holdout}/<m>/<m>_results.json` (per-detector + per-size)\n"
                 "- `eval/results/svanstrom_persize/summary.csv` (per-size DRONE on Svanström)\n"
                 "- `eval/results/video_persize/summary.csv` (per-size on real-video)\n"
                 "- `eval/results/roboflow_ood/summary.csv` (rgb_drone + confuser size buckets)\n"
                 "- `eval/results/pipeline_video_tests*/pipeline_comparison.csv` (full cascade per clip, 3 classifiers)\n"
                 "- `eval/results/full_pipeline_persize/<ds>/<det>/<clf>/summary.csv` (per-size cascade, as it lands)\n")

    DOC.write_text("\n".join(parts), encoding="utf-8")
    write_per_clip_csv()
    write_per_size_csv()
    print(f"Wrote {DOC} ({sum(len(p) for p in parts)} chars)")
    print(f"Wrote {PER_CLIP_CSV}")
    print(f"Wrote {PER_SIZE_CSV}")


if __name__ == "__main__":
    main()
