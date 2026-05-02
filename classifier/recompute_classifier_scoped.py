"""
recompute_classifier_scoped.py — Recompute classifier configs with per-frame GT scoping.

For classifier configs, GT scope follows the routing decision:
  - label=0 (trust neither): no GT → FN=0
  - label=1 (trust RGB): score against RGB GT only
  - label=2 (trust IR):  score against IR GT only
  - label=3 (trust both): score against both GTs

This is the methodologically correct approach: the classifier should only
be penalized for misses in the modality it chose to trust.

Reads per_det.jsonl — no model inference needed.
"""
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "runs" / "eval_six_configs"
PATCH_THR = 0.70
RGB_CONF = 0.25
IR_CONF  = 0.40

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
def svan_category(key):
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


def process_dataset(ds_name):
    d = OUT_ROOT / ds_name
    perdet = d / "per_det.jsonl"
    if not perdet.exists():
        print(f"  [{ds_name}] no per_det.jsonl — skip")
        return

    print(f"  [{ds_name}] processing per_det.jsonl...")

    configs = ["classifier", "classifier_filter", "classifier_then_filter"]
    counters = {rule: {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                       for c in configs} for rule in ("iou", "iop")}
    fp_by_cat = {rule: {c: defaultdict(int) for c in configs} for rule in ("iou", "iop")}

    n_frames = 0
    for ln in perdet.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        clf_raw = rec["clf_raw"]
        clf_flt = rec["clf_flt"]
        rgb_n_gt = rec["rgb_n_gt"]
        ir_n_gt  = rec["ir_n_gt"]
        cat = svan_category(rec["key"])
        n_frames += 1

        rgb_all = rec["rgb"]
        ir_all  = rec["ir"]

        rgb_raw = [d for d in rgb_all if d[0] >= RGB_CONF]
        ir_raw  = [d for d in ir_all  if d[0] >= IR_CONF]
        rgb_flt = [d for d in rgb_raw if d[1] < PATCH_THR]
        ir_flt  = [d for d in ir_raw  if d[1] < PATCH_THR]

        # For each classifier config, determine:
        # - which dets are kept (kr, ki)
        # - which GTs to score against (based on routing label)
        config_specs = {}

        # 1. classifier: routes on raw features
        config_specs["classifier"] = {
            "label": clf_raw,
            "rgb_dets": rgb_raw,
            "ir_dets": ir_raw,
        }

        # 2. classifier_filter (old: filter→classifier)
        config_specs["classifier_filter"] = {
            "label": clf_flt,
            "rgb_dets": rgb_flt,
            "ir_dets": ir_flt,
        }

        # 3. classifier_then_filter (new: classifier→filter)
        # Use clf_raw for routing, then filter the routed dets
        ctf_rgb = rgb_flt if clf_raw in (1, 3) else []
        ctf_ir  = ir_flt  if clf_raw in (2, 3) else []
        config_specs["classifier_then_filter"] = {
            "label": clf_raw,
            "rgb_dets": ctf_rgb,
            "ir_dets": ctf_ir,
        }

        for c_name, spec in config_specs.items():
            label = spec["label"]
            kr = spec["rgb_dets"] if label in (1, 3) else []
            ki = spec["ir_dets"]  if label in (2, 3) else []

            # Per-frame GT scope based on routing decision
            use_rgb_gt = label in (1, 3)
            use_ir_gt  = label in (2, 3)

            for rule, m_idx in (("iou", 2), ("iop", 3)):
                tp = fp = fn = 0

                # Score RGB dets
                if use_rgb_gt:
                    # Score kr against rgb GT
                    # Greedy matching simulation using stored flags
                    for det in kr:
                        if det[m_idx]:
                            tp += 1
                        else:
                            fp += 1
                    # FN = rgb GTs not matched by kr
                    rgb_tp = sum(1 for det in kr if det[m_idx])
                    fn += max(0, rgb_n_gt - rgb_tp)
                else:
                    # No RGB GT scope — any RGB det is FP
                    fp += len(kr)

                # Score IR dets
                if use_ir_gt:
                    for det in ki:
                        if det[m_idx]:
                            tp += 1
                        else:
                            fp += 1
                    ir_tp = sum(1 for det in ki if det[m_idx])
                    fn += max(0, ir_n_gt - ir_tp)
                else:
                    fp += len(ki)

                # TN: no relevant GT AND no dets
                has_gt = (use_rgb_gt and rgb_n_gt > 0) or (use_ir_gt and ir_n_gt > 0)
                has_det = len(kr) > 0 or len(ki) > 0
                tn_inc = 1 if (not has_gt and not has_det) else 0

                counters[rule][c_name]["tp"] += tp
                counters[rule][c_name]["fp"] += fp
                counters[rule][c_name]["fn"] += fn
                counters[rule][c_name]["tn"] += tn_inc
                fp_by_cat[rule][c_name][cat] += fp

    # Print results
    print(f"\n  [{ds_name}] SCOPED GT results ({n_frames:,} frames):\n")
    cats = [*SVAN_CATS, "OTHER"]
    for rule in ("iou", "iop"):
        print(f"  {rule.upper()} Match:")
        print(f"    {'config':<28s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s} "
              f"{'P':>7s} {'R':>7s} {'F1':>7s}")
        print(f"    {'-'*90}")
        for c_name in configs:
            c = counters[rule][c_name]
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
            print(f"    {c_name:<28s} {tp:>8,} {fp:>8,} {fn:>8,} {tn:>8,} "
                  f"{p:>7.4f} {r:>7.4f} {f1:>7.4f}")
        print(f"\n    FP by category:")
        print(f"    {'config':<28s} " + " ".join(f"{c:>10s}" for c in cats))
        for c_name in configs:
            vals = " ".join(f"{fp_by_cat[rule][c_name].get(c,0):>10,}" for c in cats)
            print(f"    {c_name:<28s} {vals}")
        print()


def main():
    print("Recomputing classifier configs with per-frame GT scoping...\n")
    for ds in ["antiuav", "svanstrom"]:
        process_dataset(ds)
    print("Done!")


if __name__ == "__main__":
    main()
