"""
05_aggregate.py — One-stop synthesis JSON for the writeup.

Walks all known CSV/JSON outputs (existing + this session's new ones) and
produces analytics/spec_analysis/results/summary.json keyed by
(model, dataset, imgsz) so the writeup can cite from a single file.

CPU only. No new numbers — pure unification.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analytics" / "spec_analysis" / "results"

SOURCES = {
    "imgsz_sweep":      RESULTS / "imgsz_sweep.csv",
    "failures_summary": RESULTS / "failures.csv",
    "geometry_summary": RESULTS / "_geometry_summary.json",
    "selcom_val_finetune_eval": ROOT / "runs" / "rgb_finetune_eval" / "Yolo26n_selcom_mixed_ft2_1280" / "comparison.json",
    "preprocess_sweep": ROOT / "runs" / "preprocess_sweep" / "selcom_val.csv",
    "svanstrom_by_category": ROOT / "eval" / "results" / "_failure_diagnosis" / "svanstrom_1280_by_category.csv",
    "confuser_hallucination": ROOT / "eval" / "results" / "_failure_diagnosis" / "confuser_test_hallucination.csv",
}


def load_csv(p: Path):
    if not p.exists(): return None
    with p.open() as f:
        return list(csv.DictReader(f))


def load_json(p: Path):
    if not p.exists(): return None
    return json.loads(p.read_text())


def summarize_failures(rows):
    """Aggregate failures.csv into per (model,dataset) summary."""
    if not rows: return {}
    by_pair = {}
    for row in rows:
        k = (row["model"], row["dataset"])
        d = by_pair.setdefault(k, {"TP": 0, "FP": 0, "FN": 0,
                                    "tp_sqrt_px": [], "fn_sqrt_px": [],
                                    "tp_clutter": [], "fn_clutter": [],
                                    "fp_conf": []})
        s = row["status"]
        d[s] = d.get(s, 0) + 1
        try:
            sqrt_px = float(row["sqrt_area_px"])
            clutter = float(row["clutter"])
            conf = float(row["conf"])
            if s == "TP":
                d["tp_sqrt_px"].append(sqrt_px); d["tp_clutter"].append(clutter)
            elif s == "FN":
                d["fn_sqrt_px"].append(sqrt_px); d["fn_clutter"].append(clutter)
            elif s == "FP":
                d["fp_conf"].append(conf)
        except Exception:
            pass

    import statistics as st
    out = {}
    for (m, ds), d in by_pair.items():
        def med(lst): return round(st.median(lst), 2) if lst else None
        tp, fp, fn = d.get("TP", 0), d.get("FP", 0), d.get("FN", 0)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        out[f"{m}|{ds}"] = dict(
            tp=tp, fp=fp, fn=fn,
            precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
            tp_sqrt_px_median=med(d["tp_sqrt_px"]),
            fn_sqrt_px_median=med(d["fn_sqrt_px"]),
            tp_clutter_median=med(d["tp_clutter"]),
            fn_clutter_median=med(d["fn_clutter"]),
            fp_conf_median=med(d["fp_conf"]),
        )
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    summary = {}

    # Imgsz sweep — keyed by (model,dataset,imgsz)
    iss = load_csv(SOURCES["imgsz_sweep"])
    if iss:
        summary["imgsz_sweep"] = {
            f"{r['model']}|{r['dataset']}|{r['imgsz']}": {
                k: (float(v) if k in ("precision", "recall", "f1", "mean_ms_per_frame", "total_seconds") else int(v))
                for k, v in r.items() if k not in ("model", "dataset", "imgsz")
            } for r in iss
        }

    # Failures — per pair aggregate
    fl = load_csv(SOURCES["failures_summary"])
    if fl:
        summary["failures"] = summarize_failures(fl)

    # Geometry
    g = load_json(SOURCES["geometry_summary"])
    if g:
        summary["dataset_geometry"] = g

    # Existing eval outputs (raw)
    se = load_json(SOURCES["selcom_val_finetune_eval"])
    if se:
        summary["ledger_selcom_ft2_1280"] = se

    sv = load_csv(SOURCES["svanstrom_by_category"])
    if sv: summary["svanstrom_by_category"] = sv

    ch = load_csv(SOURCES["confuser_hallucination"])
    if ch: summary["confuser_hallucination"] = ch

    pp = load_csv(SOURCES["preprocess_sweep"])
    if pp:
        # Just keep conf=0.25 rows for compactness
        summary["preprocess_sweep_conf25"] = [r for r in pp if r.get("conf_thr") == "0.25"]

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Aggregated summary: {out}")
    print(f"  imgsz_sweep cells: {len(summary.get('imgsz_sweep', {}))}")
    print(f"  failures pairs:    {len(summary.get('failures', {}))}")
    print(f"  geometry datasets: {len(summary.get('dataset_geometry', {}))}")


if __name__ == "__main__":
    main()
