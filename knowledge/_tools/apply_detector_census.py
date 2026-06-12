#!/usr/bin/env python3
"""apply_detector_census.py — one-time, idempotent. Applies the detector-census agent's
COMPLETE RGB+IR lineage (on-disk ground truth + provenance from args.yaml/results.csv).

Also FIXES the ir_v3 conflation: ir_v3 = dataset-version V3 (§4.1 test F1 0.611) -> repoint
weights to the dataset-V3 run; the corrective-finetune chain (v1/v2/v3 -> v3b) is registered
separately as ir_corrective_*. Metrics in notes are final-epoch train/val (NOT thesis eval-set).
Idempotent: skips existing ids; fixes are conditional.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def M(id, name, weights, prov, type="ir_yolo", lifecycle="archived", train="", script=""):
    return dict(id=id, name=name, type=type, purpose_tags="drone-detection",
                trained_from_script=script, train_dataset=train, weights_path=weights,
                provenance_notes=prov, production="no", lifecycle=lifecycle)


MODELS = [
    # --- RGB experimental probes (present, superseded) ---
    M("rgb_unfrozen", "Yolo26n_unfrozen", "models/rgb/Yolo26n_unfrozen/weights/best.pt",
      "5ep freeze=0 imgsz=512 lr=2e-5 from baseline; P0.887 R0.766 mAP50 0.810; full-unfreeze probe",
      type="rgb_yolo", lifecycle="superseded"),
    M("rgb_freeze8", "Yolo26n_freeze8", "models/rgb/Yolo26n_freeze8/weights/best.pt",
      "4ep freeze=8 imgsz=512; P0.909 R0.840 mAP50 0.867; freeze-depth probe", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_hardneg_v1", "Yolo26n_hardneg", "models/rgb/Yolo26n_hardneg/weights/best.pt",
      "3ep freeze=10; P0.976 R0.905 mAP50 0.949; first hard-neg FT (v1 dataset)", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_hardneg_v2", "Yolo26n_hardneg_v2", "models/rgb/Yolo26n_hardneg_v2/weights/best.pt",
      "3ep freeze=10; P0.969 R0.869 mAP50 0.930; v2 hard-neg dataset", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_hardneg_v2_f5", "Yolo26n_hardneg_v2_f5", "models/rgb/Yolo26n_hardneg_v2_f5/weights/best.pt",
      "3ep freeze=5; P0.941 R0.854 mAP50 0.900", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_hardneg_v3", "Yolo26n_hardneg_v3", "models/rgb/Yolo26n_hardneg_v3/weights/best.pt",
      "3ep freeze=10; P0.980 R0.891 mAP50 0.933; direct parent of hardneg_v3more", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_selcom_ft1", "Yolo26n_selcom_ft1", "models/rgb/Yolo26n_selcom_ft1/weights/best.pt",
      "10ep pure-selcom (no general mix); P0.849 R0.758 mAP50 0.791", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_selcom_mixed_ft1", "Yolo26n_selcom_mixed_ft1", "models/rgb/Yolo26n_selcom_mixed_ft1/weights/best.pt",
      "10ep first mixed selcom FT; P0.892 R0.684 mAP50 0.751; predecessor of ft2", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_selcom_mixed_ft2_640", "Yolo26n_selcom_mixed_ft2", "models/rgb/Yolo26n_selcom_mixed_ft2/weights/best.pt",
      "imgsz=640 (vs ft2_1280); P0.545 R0.397 mAP50 0.405; imgsz=640 Svanstrom-floor failure", type="rgb_yolo", lifecycle="superseded"),
    M("rgb_selcom_mixed_ft3_960", "Yolo26n_selcom_mixed_ft3_960", "models/rgb/Yolo26n_selcom_mixed_ft3_960/weights/best.pt",
      "20ep imgsz=960; P0.883 R0.674 mAP50 0.762; imgsz variant of ft3", type="rgb_yolo", lifecycle="superseded"),
    # --- IR corrective-finetune chain (present; inits from IR_final_cleaned) ---
    M("ir_corrective_v1", "IR corrective finetune_v1", "models/ir/corrective_finetune/finetune_v1/weights/best.pt",
      "8ep freeze=10 init IR_final_cleaned; P0.917 R0.942 mAP50 0.950; first corrective FT", lifecycle="superseded",
      script="(corrective_finetune)"),
    M("ir_corrective_v2", "IR corrective finetune_v2", "models/ir/corrective_finetune/finetune_v2/weights/best.pt",
      "6ep freeze=10 init IR_final_cleaned; P0.920 R0.938 mAP50 0.947", lifecycle="superseded", script="(corrective_finetune)"),
    M("ir_corrective_v3", "IR corrective finetune_v3", "models/ir/corrective_finetune/finetune_v3/weights/best.pt",
      "corrective-finetune chain; direct parent of production ir_v3b (NOT the dataset-version V3)", lifecycle="superseded"),
    # --- IR archive lineage (archived; resurrect-worthy, NOT obsolete) ---
    M("ir_dsetV1_goldV2_300ep", "IR_dsetV1_goldV2_300ep", "archive/models/ir/IR_dsetV1_goldV2_300ep/best.pt",
      "dataset V1 (goldV2 labels); best.pt-only; P0.968 R0.962 mAP50 0.976 (from 300ep sibling)", train="IR dsetV1 goldV2"),
    M("ir_ft_goldV2_70ep", "IR_FT_goldV2_IRdsetV1_70ep", "archive/results_vastai_first_run_70ep/runs/detect/runs/IR_FT_goldV2_IRdsetV1_aug0_s0/weights/best.pt",
      "70ep init baseline(RGB); P0.961 R0.946 mAP50 0.970; first vastai goldV2 run"),
    M("ir_ft_goldV2_300ep", "IR_FT_goldV2_IRdsetV1_300ep", "archive/results_vast_ai_300ep/runs/detect/runs/IR_FT_goldV2_IRdsetV1_aug0_s0/weights/best.pt",
      "300ep continuation of 70ep; P0.968 R0.962 mAP50 0.976; likely source of ir_dsetV1_goldV2_300ep (DUP)"),
    M("ir_ft_gold_pilot", "IR_FT_gold_IRdsetV1_pilot", "archive/runs/detect/runs/IR_FT_gold_IRdsetV1_aug0_s0_pilot/weights/best.pt",
      "15ep imgsz=512 gold-label pilot"),
    M("ir_ft_goldV2_pilot", "IR_FT_goldV2_IRdsetV1_pilot", "archive/runs/detect/runs/IR_FT_goldV2_IRdsetV1_aug0_s0_pilot/weights/best.pt",
      "15ep imgsz=512 goldV2-label pilot"),
    M("ir_ft_silver_pilot", "IR_FT_silver_IRdsetV1_pilot", "archive/runs/detect/runs/IR_FT_silver_IRdsetV1_aug0_s0_pilot/weights/best.pt",
      "15ep imgsz=512 silver (IR-only) label pilot; contrast arm to gold"),
    M("ir_dsetV7_pilot", "IR_dsetV7_pilot_conservative", "archive/runs/detect/runs/IR_dsetV7_pilot_conservative/weights/best.pt",
      "5ep init yolo11n (NOT baseline) lr=0.01; conservative V7 probe on V6 data"),
    M("ir_ft_dsetV7_clahe_300ep", "IR_FT_dsetV7_aug1_clahe_300ep_s0", "archive/IR_FT_dsetV7_aug1_clahe_300ep_s0/IR_FT_dsetV7_aug1_clahe_300ep_s0/weights/best.pt",
      "~154ep aug1+CLAHE init baseline; P0.962 R0.859 mAP50 0.891; parent of cst_stage2", train="IR dsetV7"),
    M("ir_ft_dsetV8_noclahe_200ep", "IR_FT_dsetV8_aug1_noclahe_200ep_s0", "archive/IR_FT_dsetV8_aug1_noclahe_200ep_s0/IR_FT_dsetV8_aug1_noclahe_200ep_s0/weights/best.pt",
      "~177ep aug1 no-CLAHE; P0.941 R0.731 mAP50 0.794; CLAHE-ablation vs V7", train="IR dsetV8"),
    M("ir_ft_cst_stage2", "IR_FT_cst_stage2_s0", "archive/IR_FT_cst_stage2_s0/weights/best.pt",
      "30ep stage-2 init from V7-clahe best (cross-stage transfer); P0.916 R0.723 mAP50 0.767"),
    M("ir_dsetV7b_110ep", "IR_dsetV7b_110ep", "archive/models/ir/IR_dsetV7b_110ep/best.pt",
      "best.pt-only; ~110ep; name-only provenance (no args/results)"),
    M("ir_dsetV9b1_rgbcfg", "IR_dsetV9b1_rgbcfg", "archive/models2/IR_dsetV9b1_rgbcfg/weights/best.pt",
      "117ep init yolo26n (rgb-config) lr=0.001; P0.880 R0.746 mAP50 0.743; latest IR dataset V9; DUP at models2/models_/", train="IR dsetV9b1"),
    M("ir_gold_rgbcfg", "IR_gold_rgbcfg", "archive/models2/IR_gold_rgbcfg/IR_gold_rgbcfg/weights/best.pt",
      "200ep init yolo26n (rgb-config); P0.948 R0.764 mAP50 0.821; DUP at models2/models_/", train="IR gold"),
    M("ir_ft_merged_dsetV2_may22", "IR_FT_merged_IRdsetV2_aug0_s0", "archive/run_300_ep_dsetv2golden_may22_merged/ES_Drone_Detection/runs/detect/runs/IR_FT_merged_IRdsetV2_aug0_s0/weights/best.pt",
      "~266ep init baseline; P0.893 R0.803 mAP50 0.855; merged-corpus run = source of ir_v2"),
]

# Fix the ir_v3 conflation (dataset-version V3, not corrective)
IR_V3_FIX = {
    "weights_path": "runs/IR_FT_dsetV3_aug0_s0/weights/best.pt",
    "provenance_notes": "Dataset-version V3 (NOT corrective finetune_v3, which is ir_corrective_v3). §4.1 test F1 0.611; final-epoch train P0.922 R0.857 mAP50 0.900. archive dup at IR_FT_dsetV3_run_135ep/.",
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


def _fix_ir_v3():
    cur = kb._read("models"); by = {r["id"]: r for r in cur}
    if "ir_v3" in by:
        by["ir_v3"].update(IR_V3_FIX)
        kb._write("models", cur)
        print("models.ir_v3: weights_path corrected to dataset-V3 run")


def main():
    _append("models", MODELS)
    _fix_ir_v3()
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
