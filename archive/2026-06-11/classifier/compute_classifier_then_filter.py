"""
compute_classifier_then_filter.py — Compute the 7th config from cached per-det data.

classifier_then_filter:
  1. Classifier routes on RAW features (as trained) → lbl_raw
  2. Keep RGB dets if lbl_raw ∈ {1,3}, IR dets if lbl_raw ∈ {2,3}
  3. Filter vetoes: keep only dets with P(confuser) < PATCH_THR

Reads per_det.jsonl (already has conf, filter_prob, match flags, clf_raw).
No model inference needed.
"""
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "runs" / "eval_six_configs"
PATCH_THR = 0.70
RGB_CONF = 0.40
IR_CONF = 0.40

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
        return None

    print(f"  [{ds_name}] processing per_det.jsonl...")
    counters = {"iou": {"tp": 0, "fp": 0, "fn": 0},
                "iop": {"tp": 0, "fp": 0, "fn": 0}}
    fp_by_cat = {"iou": defaultdict(int), "iop": defaultdict(int)}
    n_frames = 0

    for ln in perdet.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        clf_raw = rec["clf_raw"]
        cat = svan_category(rec["key"])
        n_frames += 1

        # Decide which modalities are trusted (classifier on raw features)
        use_rgb = clf_raw in (1, 3)
        use_ir  = clf_raw in (2, 3)

        for rule, m_idx in (("iou", 2), ("iop", 3)):
            tp = fp = 0
            matched_gts_rgb = set()
            matched_gts_ir = set()

            # Process RGB detections (kept by classifier + surviving filter)
            if use_rgb:
                rgb_recs = rec["rgb"]  # [conf, filter_prob, m_iou, m_iop]
                # Filter by conf threshold first, then by patch filter
                gt_idx = 0
                for det in rgb_recs:
                    conf, fprob, m_iou, m_iop = det
                    if conf < RGB_CONF:
                        continue
                    if fprob >= PATCH_THR:  # confuser → vetoed by filter
                        continue
                    # This detection survives both classifier routing AND filter
                    matched = det[m_idx]
                    if matched:
                        tp += 1
                    else:
                        fp += 1

            # Process IR detections
            if use_ir:
                ir_recs = rec["ir"]
                for det in ir_recs:
                    conf, fprob, m_iou, m_iop = det
                    if conf < IR_CONF:
                        continue
                    if fprob >= PATCH_THR:
                        continue
                    matched = det[m_idx]
                    if matched:
                        tp += 1
                    else:
                        fp += 1

            # FN: GT boxes not matched by any surviving detection
            # Approximate: use total GT minus TPs
            # For classifier configs, GT scope is both modalities
            total_gt = rec["rgb_n_gt"] + rec["ir_n_gt"]
            fn = max(0, total_gt - tp)

            counters[rule]["tp"] += tp
            counters[rule]["fp"] += fp
            counters[rule]["fn"] += fn
            fp_by_cat[rule][cat] += fp

    # Print results
    print(f"\n  [{ds_name}] classifier_then_filter results ({n_frames:,} frames):")
    for rule in ("iou", "iop"):
        tp = counters[rule]["tp"]
        fp = counters[rule]["fp"]
        fn = counters[rule]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
        print(f"    [{rule.upper()}]  TP={tp:>8,}  FP={fp:>8,}  FN={fn:>8,}  "
              f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
        cats = [*SVAN_CATS, "OTHER"]
        fp_str = "  ".join(f"{c}={fp_by_cat[rule].get(c,0)}" for c in cats)
        print(f"           FP by cat: {fp_str}")

    return counters


def main():
    print("Computing classifier_then_filter from cached per-det data...\n")
    for ds in ["antiuav", "svanstrom"]:
        process_dataset(ds)
    print("\nDone!")


if __name__ == "__main__":
    main()
