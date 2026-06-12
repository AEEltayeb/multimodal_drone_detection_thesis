"""
compare_thesis_format.py — Reproduce the check.txt table format for OLD vs
NEW40 vs NEW32 from cached eval_six_configs per_det.jsonl files.

Output format (per dataset, per IoU/IoP rule):
  Config              TP      FP     FN     TN   Precision   Recall    F1
  ir_only            ...
  rgb_only           ...
  classifier         ...        (raw classifier on unfiltered dets)
  ir_filter          ...        (IR dets surviving patch filter)
  rgb_filter         ...        (RGB dets surviving patch filter)
  filter->classifier ...        (classifier on patch-filtered dets)
  classifier->filter ...        (apply patch filter ONLY to modality the classifier trusted)

Scoring (matches eval_six_configs methodology):
  TP/FP/FN  = detection-level, summed across BOTH modalities for classifier configs
  TN        = frame-level "no GT in either modality AND no surviving det in either"
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


def conf_idx(rule):
    return 2 if rule == "iou" else 3  # which match flag to use


def evaluate(jsonl_path: Path, rule="iou"):
    midx = conf_idx(rule)
    cfgs = ["ir_only", "rgb_only", "classifier",
            "ir_filter", "rgb_filter", "filter->classifier", "classifier->filter"]

    counters = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in cfgs}
    n_frames = 0

    with jsonl_path.open() as fh:
        for ln in fh:
            r = json.loads(ln)
            rgb_all = r.get("rgb", [])
            ir_all  = r.get("ir",  [])
            rgb_n_gt = r["rgb_n_gt"]
            ir_n_gt  = r["ir_n_gt"]
            clf_raw = r["clf_raw"]   # classifier label on raw dets
            clf_flt = r["clf_flt"]   # classifier label on filter-survived dets

            # Threshold by conf
            rgb_raw = [d for d in rgb_all if d[0] >= RGB_CONF]
            ir_raw  = [d for d in ir_all  if d[0] >= IR_CONF]
            # Filter-survived (patch_prob < threshold)
            rgb_flt = [d for d in rgb_raw if d[1] < PATCH_THR]
            ir_flt  = [d for d in ir_raw  if d[1] < PATCH_THR]

            # ── Config-specific surviving dets per modality ──
            # ir_only / rgb_only: just YOLO dets in that modality
            # classifier: keep dets from trusted modality(ies); on raw dets
            # ir_filter / rgb_filter: filtered dets (single modality)
            # filter->classifier: classifier(clf_flt) applied to filtered dets
            # classifier->filter: classifier(clf_raw) chooses modality;
            #                     filter applied only to that modality(ies)
            def trusted_split(label, rgb_set, ir_set):
                """Return (rgb_kept, ir_kept) given trust label."""
                if label == 0:        # reject
                    return [], []
                elif label == 1:      # trust_rgb
                    return rgb_set, []
                elif label == 2:      # trust_ir
                    return [], ir_set
                else:                 # trust_both
                    return rgb_set, ir_set

            # filter->classifier: classifier on filtered dets
            r_fc, i_fc = trusted_split(clf_flt, rgb_flt, ir_flt)
            # classifier->filter: classifier on raw dets, then filter only that modality
            r_cf_raw, i_cf_raw = trusted_split(clf_raw, rgb_raw, ir_raw)
            # apply patch filter to the chosen modality(ies)
            r_cf = [d for d in r_cf_raw if d[1] < PATCH_THR]
            i_cf = [d for d in i_cf_raw if d[1] < PATCH_THR]
            # classifier (raw — no patch filter at all)
            r_clf, i_clf = trusted_split(clf_raw, rgb_raw, ir_raw)

            # Surviving det sets per config
            kept = {
                "ir_only":            ([], ir_raw),
                "rgb_only":           (rgb_raw, []),
                "classifier":         (r_clf, i_clf),
                "ir_filter":          ([], ir_flt),
                "rgb_filter":         (rgb_flt, []),
                "filter->classifier": (r_fc, i_fc),
                "classifier->filter": (r_cf, i_cf),
            }

            # GT scope per config: which modality's GT counts as missable?
            gt_scope = {
                "ir_only":            ("ir",),
                "rgb_only":           ("rgb",),
                "classifier":         ("rgb", "ir"),
                "ir_filter":          ("ir",),
                "rgb_filter":         ("rgb",),
                "filter->classifier": ("rgb", "ir"),
                "classifier->filter": ("rgb", "ir"),
            }

            # TRUST-SCOPE methodology (matches check.txt format):
            #   classifier configs: scope follows trust label
            #     reject (0)     -> scope = ("none",) — no dets, FN = max(n_gt) per frame
            #     trust_rgb (1)  -> scope = ("rgb",)
            #     trust_ir (2)   -> scope = ("ir",)
            #     trust_both (3) -> scope = ("rgb", "ir")
            #   non-classifier configs: fixed scope as before
            classifier_label = {
                "classifier":         clf_raw,
                "filter->classifier": clf_flt,
                "classifier->filter": clf_raw,
            }
            fixed_scope = {
                "ir_only":    ("ir",),
                "rgb_only":   ("rgb",),
                "ir_filter":  ("ir",),
                "rgb_filter": ("rgb",),
            }
            for cfg, (rs, ir_s) in kept.items():
                tp = fp = fn = 0
                if cfg in fixed_scope:
                    scope = fixed_scope[cfg]
                else:
                    lbl = classifier_label[cfg]
                    if   lbl == 1: scope = ("rgb",)
                    elif lbl == 2: scope = ("ir",)
                    elif lbl == 3: scope = ("rgb", "ir")
                    else:          scope = ()  # reject

                if "rgb" in scope:
                    matched = sum(1 for d in rs if d[midx] == 1)
                    tp += matched
                    fp += len(rs) - matched
                    fn += max(0, rgb_n_gt - matched)
                if "ir" in scope:
                    matched = sum(1 for d in ir_s if d[midx] == 1)
                    tp += matched
                    fp += len(ir_s) - matched
                    fn += max(0, ir_n_gt - matched)
                if not scope:  # reject frame
                    # Drone in either modality counts as ONE missed event (paired-aware)
                    if rgb_n_gt > 0 or ir_n_gt > 0:
                        fn += max(rgb_n_gt, ir_n_gt)

                # TN: frame-level "no GT in either modality AND config produced no det"
                # We use modality-agnostic check: drone present anywhere?
                drone_present = (rgb_n_gt > 0) or (ir_n_gt > 0)
                any_det_in_scope = (("rgb" in scope and len(rs) > 0)
                                    or ("ir" in scope and len(ir_s) > 0))
                if not drone_present and not any_det_in_scope:
                    counters[cfg]["TN"] += 1

                counters[cfg]["TP"] += tp
                counters[cfg]["FP"] += fp
                counters[cfg]["FN"] += fn

            n_frames += 1

    return counters, n_frames


def fmt_counters(c):
    tp = c["TP"]; fp = c["FP"]; fn = c["FN"]; tn = c["TN"]
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return tp, fp, fn, tn, p, r, f1


def print_table(label, results, rule):
    """results: dict variant -> {cfg: counters}"""
    print(f"\n{label} — rule={rule.upper()} @ 0.5")
    cfgs = ["ir_only", "rgb_only", "classifier",
            "ir_filter", "rgb_filter", "filter->classifier", "classifier->filter"]
    for variant, ctrs in results.items():
        print(f"\n  [{variant}]")
        print(f"    {'Config':<22s} {'TP':>8} {'FP':>7} {'FN':>7} {'TN':>7}  "
              f"{'P':>8} {'R':>8} {'F1':>8}")
        for cfg in cfgs:
            tp, fp, fn, tn, p, r, f1 = fmt_counters(ctrs[cfg])
            print(f"    {cfg:<22s} {tp:>8,} {fp:>7,} {fn:>7,} {tn:>7,}  "
                  f"{p:>8.4f} {r:>8.4f} {f1:>8.4f}")


def main():
    all_results = {}
    for ds in DATASETS:
        all_results[ds] = {}
        for rule in ("iou", "iop"):
            ds_rule_results = {}
            for vname, dirname in VARIANTS.items():
                jpath = RUNS / dirname / ds / "per_det.jsonl"
                if not jpath.exists():
                    print(f"[skip] {vname} {ds}: {jpath} missing"); continue
                ctrs, n = evaluate(jpath, rule=rule)
                ds_rule_results[vname] = ctrs
            all_results[ds][rule] = ds_rule_results
            print_table(f"{ds.upper()} ({n:,} frames)", ds_rule_results, rule)

    out = SCRIPT_DIR / "runs" / "thesis_format_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {r: {v: cs for v, cs in vs.items()}
                                    for r, vs in dd.items()}
                                for k, dd in all_results.items()}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
