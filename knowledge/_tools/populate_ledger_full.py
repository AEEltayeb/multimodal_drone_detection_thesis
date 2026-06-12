#!/usr/bin/env python3
"""populate_ledger_full.py — one-time, idempotent. Migrates the remaining
decision-relevant EVIDENCE_LEDGER sections (§8 Roboflow OOD, §9.4 corrected real-video,
FT4 backbone-freeze changelog, §8 limitations) into evals + ledger.

Also CORRECTS the ir-grayscale-fallback finding to the §9.4 corrected framing (the
earlier 'beats every RGB' used superseded scoring; it merely matches baseline at ~1/3
the confuser fire rate). Idempotent: skips existing ids; updates the one finding in place.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

CONFIGS = [
    dict(id="video_drone_iop", dataset="drone_video_tests", n_samples="1359", imgsz="per-model",
         scoring_rule="iop", conf_thr="0.25", notes="9 drone videos, 1234 GT; per-model imgsz (selcom 1280). §9.4.2"),
    dict(id="video_confuser", dataset="confuser_video_tests", n_samples="1250", imgsz="per-model",
         scoring_rule="", conf_thr="0.25", notes="10 confuser videos; frame-level FPR (no GT). §9.4.3"),
    dict(id="roboflow_rgb_drone_640", dataset="rgb_drone_roboflow", n_samples="", imgsz="640",
         scoring_rule="iou", conf_thr="0.25", notes="OOD; raw detector; missing-label caveat -> relative ordering only. §8.1"),
]

def V(id, target, cfg, f1="", r="", p="", halluc="", note=""):
    return dict(id=id, target=target, config_id=cfg, f1=f1, recall=r, precision=p,
                halluc_rate=halluc, extra=note, source_script="eval/eval_video_tests.py", date="2026-05-30")

EVALS = [
    # §9.4.2 real-video drone detection (IoP@0.5)
    V("vid_drone_baseline", "baseline", "video_drone_iop", "0.760", "0.749", "0.771", note="§9.4.2"),
    V("vid_drone_retrainedv2", "retrained_v2", "video_drone_iop", "0.605", "0.453", "0.909", note="low-R/high-P §9.4.2"),
    V("vid_drone_selcom1280", "selcom_1280", "video_drone_iop", "0.721", "0.812", "0.649", note="high-R/low-P; mode of selcom_mixed_ft2_1280 @1280 §9.4.2"),
    V("vid_drone_selcom640", "selcom_640", "video_drone_iop", "0.730", "0.666", "0.807", note="selcom weights @ imgsz=640; cleanest pt at FPPI<=0.3 §9.4.5"),
    V("vid_drone_ir_gray", "ir_grayscale", "video_drone_iop", "0.636", "0.557", "0.743", note="IR_final on grayscale-RGB §9.4.2"),
    V("vid_drone_ir_raw", "ir_raw", "video_drone_iop", "0.295", "0.191", "0.647", note="IR_final on raw-RGB (unusable) §9.4.2"),
    # §9.4.3 real-video confuser FPR (frame-level fire = halluc proxy)
    V("vid_conf_baseline", "baseline", "video_confuser", halluc="0.475", note="§9.4.3"),
    V("vid_conf_retrainedv2", "retrained_v2", "video_confuser", halluc="0.264", note="§9.4.3"),
    V("vid_conf_selcom1280", "selcom_1280", "video_confuser", halluc="0.593", note="WORST OOD confuser RGB §9.4.3"),
    V("vid_conf_selcom640", "selcom_640", "video_confuser", halluc="0.423", note="imgsz=640 drops fire 0.709->0.260 FPPI §9.4.5"),
    V("vid_conf_ir_gray", "ir_grayscale", "video_confuser", halluc="0.256", note="lowest among usable §9.4.3"),
    V("vid_conf_ir_raw", "ir_raw", "video_confuser", halluc="0.325", note="§9.4.3"),
    # §8.1 Roboflow OOD RGB drone (raw)
    dict(id="rob_rgb_baseline", target="baseline", config_id="roboflow_rgb_drone_640", f1="0.820",
         recall="0.746", precision="0.912", extra="raw; baseline wins OOD drone too §8.1",
         source_script="eval/run_roboflow_eval.py", date="2026-05-30"),
    dict(id="rob_rgb_retrainedv2", target="retrained_v2", config_id="roboflow_rgb_drone_640", f1="0.813",
         recall="0.726", precision="0.924", extra="raw; collapse does NOT generalize to OOD §8.1",
         source_script="eval/run_roboflow_eval.py", date="2026-05-30"),
]

def F(id, claim, outcome, evidence="", condition="", contradicts="", contrib="", notes=""):
    return dict(id=id, date="2026-05-30", claim=claim, outcome=outcome, condition=condition,
                evidence_evals=evidence, contradicts=contradicts, thesis_contribution=contrib,
                status="confirmed", notes=notes)

NEW_FINDINGS = [
    F("selcom-ood-confuser-damage", "selcom fine-tune (selcom_1280) is the WORST RGB on OOD confuser video (frame FPR 59.3% vs baseline 47.5%, retrained_v2 26.4%); fine-tuning to CCTV damages out-of-distribution confuser robustness",
      "supported", "vid_conf_selcom1280;vid_conf_baseline;vid_conf_retrainedv2", contrib="fine-tune OOD-robustness tradeoff", notes="§9.4.3"),
    F("no-pareto-dominance-video", "on real video no detector Pareto-dominates: baseline best drone-F1 (0.760, FPPI 0.512), IR-grayscale lowest confuser FPPI among usable (0.158), and selcom_640 is the cleanest single point at FPPI<=0.3 (F1 0.730, FPPI 0.260)",
      "supported", "vid_drone_baseline;vid_drone_selcom640;vid_drone_ir_gray", contrib="operating-point / Pareto framing of detector choice", notes="§9.4.5; selcom_640 not previously studied"),
    F("patch-verifier-distribution-bound", "the patch verifier is severely distribution-bound: it suppresses OOD airplane FPs only 3.1%/1.9% on Roboflow@640 vs ~52% on Svanstrom@1280, and costs net drone F1 at 640; the TRUST CLASSIFIER (not the patch) is the main confuser-rejection lever",
      "supported", "rob_rgb_baseline", contrib="patch verifier is distribution-bound; trust classifier carries confuser rejection", notes="§8.2,§8 reads"),
    F("ir-ood-recall-ceiling", "IR YOLO has a hidden OOD recall ceiling: R=0.264 on ir_drone_night and 0.519 on ir_mixed_cbam vs 0.945 Anti-UAV / 0.973 Svanstrom - a thesis limitation",
      "conditional", "", condition="ir_drone_night is heavily sensor-augmented (worst-case probe, not a fair OOD test)", notes="§8.3,§8 caveats"),
    F("ft4-backbone-freeze", "FT4 R3 (300 hard-negs, freeze=15, 3 epochs) is the only confuser-injection config passing all regression gates; freeze=15 is necessary AND sufficient to prevent catastrophic forgetting (freeze=12 regresses Svan recall/F1 even at 300 HN); -16pp confuser halluc",
      "supported", "", contrib="backbone-freeze depth controls catastrophic forgetting during hard-negative injection", notes="§[2026-05-27] ft4 ablation; R1/R2 FAIL, R3 PASS; A1-A4 ratio rescue all FAIL"),
]

# correction to an existing finding
FINDING_FIXES = {
    "ir-grayscale-fallback": {
        "claim": "IR model on grayscale-RGB matches baseline RGB on the hardest bird-cluttered drone clip (F1 0.837 vs 0.840) while cutting confuser fire rate to ~1/3, despite never seeing visible-light data in training",
        "outcome": "conditional",
        "condition": "MATCHES (not beats) RGB; lowest confuser FPPI (0.158) among usable detectors but drone F1 0.636 is 12.4pp below baseline",
        "evidence_evals": "vid_drone_ir_gray;vid_conf_ir_gray;vid_drone_baseline",
        "notes": "CORRECTED 2026-05-30 from §9.4: earlier 'beats every RGB / F1 0.901' used superseded scoring (§9.3). Cross-modal grayscale fallback still validated, reframed.",
    },
}


def _append(table, rows):
    cur = kb._read(table); existing = {r["id"] for r in cur}; added = 0
    for row in rows:
        if row["id"] in existing:
            continue
        errs = kb._validate_row(table, row)
        if errs:
            print(f"SKIP {table}.{row['id']}: {'; '.join(errs)}"); continue
        cur.append(row); existing.add(row["id"]); added += 1
    kb._write(table, cur)
    print(f"{table}: +{added} ({len(cur)} total)")


def _fix(table, fixes):
    cur = kb._read(table); by_id = {r["id"]: r for r in cur}; n = 0
    for rid, upd in fixes.items():
        if rid in by_id:
            by_id[rid].update(upd); n += 1
    kb._write(table, cur)
    print(f"{table}: {n} rows corrected")


def main():
    _append("eval_configs", CONFIGS)
    _append("evals", EVALS)
    _append("ledger", NEW_FINDINGS)
    _fix("ledger", FINDING_FIXES)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
