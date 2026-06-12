"""
compare_pipelines_from_perdet.py — Reconstruct eval_full_pipeline-style
4-layer metrics for OLD / NEW40 / NEW32 from cached eval_six_configs
per_det.jsonl files. No YOLO inference, no patch CNN re-runs.

Each per_det record holds:
  rgb: [[conf, patch_prob, matched_iou, matched_iop], ...]
  ir : [[conf, patch_prob, matched_iou, matched_iop], ...]
  rgb_n_gt, ir_n_gt  (drone GT counts)
  clf_raw, clf_flt   (classifier label: 0=reject 1=trust_rgb 2=trust_ir 3=trust_both)

That's enough to reproduce eval_full_pipeline.py's 4-layer table per dataset.
The only thing missing is size buckets (no box coords in per_det).

Usage:
  python classifier/compare_pipelines_from_perdet.py
"""
from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS = SCRIPT_DIR / "runs"

VARIANTS = {
    "OLD":   "eval_six_configs",
    "NEW40": "eval_six_configs_v3more_40feat",
    "NEW32": "eval_six_configs_v3more_32feat",
}

DATASETS = ["antiuav", "svanstrom"]

RGB_CONF = 0.25
IR_CONF  = 0.40
PATCH_THR = 0.70


def evaluate(jsonl_path: Path, rule="iou"):
    """rule in {iou, iop}: which match flag to use for matched_<rule>."""
    midx = 2 if rule == "iou" else 3  # index in det record

    # Per-frame YOLO frame confusion (for each modality)
    rgb_yolo = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    ir_yolo  = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    # Frame-level classifier confusion (binary: drone present?)
    clf_raw = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    clf_flt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    # Per-detection patch verifier confusion (per modality)
    rgb_patch = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    ir_patch  = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    n_frames = 0

    with jsonl_path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            rgb_dets = [d for d in r.get("rgb", []) if d[0] >= RGB_CONF]
            ir_dets  = [d for d in r.get("ir",  []) if d[0] >= IR_CONF]
            rgb_has = r["rgb_n_gt"] > 0
            ir_has  = r["ir_n_gt"]  > 0
            frame_has_drone = rgb_has or ir_has

            # ── frame-level YOLO ──
            for stats, dets, has in [(rgb_yolo, rgb_dets, rgb_has),
                                      (ir_yolo,  ir_dets,  ir_has)]:
                any_det = len(dets) > 0
                matched = any(d[midx] == 1 for d in dets)
                if has:
                    stats["TP" if matched else "FN"] += 1
                else:
                    stats["FP" if any_det else "TN"] += 1

            # ── frame-level classifier (binary) ──
            for stats, lbl in [(clf_raw, r["clf_raw"]), (clf_flt, r["clf_flt"])]:
                pred = 1 if lbl != 0 else 0  # any trust = drone detected
                gt = 1 if frame_has_drone else 0
                if pred == 1 and gt == 1:   stats["TP"] += 1
                elif pred == 0 and gt == 0: stats["TN"] += 1
                elif pred == 1 and gt == 0: stats["FP"] += 1
                else:                        stats["FN"] += 1

            # ── per-detection patch verifier ──
            # pred=1 (accept as drone) iff patch_prob < PATCH_THR
            # label=1 iff matched a drone GT (matched_iou or matched_iop = 1)
            for stats, dets in [(rgb_patch, rgb_dets), (ir_patch, ir_dets)]:
                for d in dets:
                    p = d[1]
                    pred = 1 if p < PATCH_THR else 0
                    label = 1 if d[midx] == 1 else 0
                    if pred == 1 and label == 1: stats["TP"] += 1
                    elif pred == 0 and label == 0: stats["TN"] += 1
                    elif pred == 1 and label == 0: stats["FP"] += 1
                    else: stats["FN"] += 1

            n_frames += 1

    return {
        "n_frames": n_frames,
        "rgb_yolo": rgb_yolo, "ir_yolo": ir_yolo,
        "classifier_raw": clf_raw, "classifier_filter": clf_flt,
        "rgb_patch": rgb_patch, "ir_patch": ir_patch,
    }


def fmt_row(name, m):
    tp, tn, fp, fn = m["TP"], m["TN"], m["FP"], m["FN"]
    n = tp + tn + fp + fn
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return (f"{name:<22s}  TP={tp:>7,}  TN={tn:>6,}  FP={fp:>6,}  FN={fn:>6,}  "
            f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}  n={n:,}")


def main():
    all_results = {}
    for ds in DATASETS:
        all_results[ds] = {}
        for rule in ("iou", "iop"):
            print(f"\n{'=' * 92}")
            print(f"  {ds.upper()}  —  rule={rule.upper()}")
            print(f"{'=' * 92}")
            for vname, dirname in VARIANTS.items():
                jpath = RUNS / dirname / ds / "per_det.jsonl"
                if not jpath.exists():
                    print(f"\n  [skip] {vname}: {jpath} missing")
                    continue
                res = evaluate(jpath, rule=rule)
                all_results[ds].setdefault(rule, {})[vname] = res
                print(f"\n  --- {vname} ({res['n_frames']:,} frames) ---")
                print("    " + fmt_row("rgb_yolo",          res["rgb_yolo"]))
                print("    " + fmt_row("ir_yolo",           res["ir_yolo"]))
                print("    " + fmt_row("classifier_raw",    res["classifier_raw"]))
                print("    " + fmt_row("classifier_filter", res["classifier_filter"]))
                print("    " + fmt_row("rgb_patch (per-det)", res["rgb_patch"]))
                print("    " + fmt_row("ir_patch  (per-det)", res["ir_patch"]))

    # Summary delta tables for the four "new" components
    print(f"\n{'=' * 92}")
    print("  HEADLINE: NEW vs OLD per-layer F1 deltas")
    print(f"{'=' * 92}")
    for ds in DATASETS:
        for rule in ("iou", "iop"):
            print(f"\n  [{ds.upper()}  rule={rule.upper()}]  layer  /  OLD F1  /  NEW40 F1  /  NEW32 F1")
            for layer in ("rgb_yolo", "ir_yolo", "classifier_raw", "classifier_filter",
                          "rgb_patch", "ir_patch"):
                f1s = []
                for v in ("OLD", "NEW40", "NEW32"):
                    m = all_results[ds].get(rule, {}).get(v, {}).get(layer)
                    if not m:
                        f1s.append("---")
                        continue
                    tp, fp, fn = m["TP"], m["FP"], m["FN"]
                    p = tp / max(1, tp + fp)
                    r = tp / max(1, tp + fn)
                    f1 = 2 * p * r / max(1e-9, p + r)
                    f1s.append(f"{f1:.4f}")
                print(f"    {layer:<22s}  {f1s[0]}     {f1s[1]}     {f1s[2]}")

    out = SCRIPT_DIR / "runs" / "full_pipeline_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
