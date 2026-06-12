#!/usr/bin/env python3
"""detector_census.py — one-time, idempotent. COMPLETE RGB + IR detector registry from
on-disk ground truth (PowerShell best.pt enumeration 2026-05-30), not the agent report.

Closes the IR/RGB lineage gap: registers every detector weight found under runs/, models/,
training/, archive/. lifecycle=archived for anything physically under archive/ (resurrect-worthy,
not obsolete); superseded for clearly-replaced experiments; active for available probes/candidates.
Idempotent: skips existing ids. Adds notes to `ft4` + `baseline` for ablation/ancestor copies.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def M(id, name, weights, prov, type="ir_yolo", tags="drone-detection", lifecycle="archived",
      absorbed="", train=""):
    return dict(id=id, name=name, type=type, purpose_tags=tags, trained_from_script="",
                train_dataset=train, weights_path=weights, provenance_notes=prov,
                production="no", lifecycle=lifecycle, absorbed_into=absorbed)


MODELS = [
    # --- RGB experimental lineage (present, not archived) ---
    M("rgb_freeze8", "Yolo26n_freeze8", "models/rgb/Yolo26n_freeze8/weights/best.pt",
      "freeze=8 training-strategy probe", type="rgb_yolo", lifecycle="active"),
    M("rgb_unfrozen", "Yolo26n_unfrozen", "models/rgb/Yolo26n_unfrozen/weights/best.pt",
      "unfrozen-backbone training probe", type="rgb_yolo", lifecycle="active"),
    M("rgb_hardneg_v1", "Yolo26n_hardneg", "models/rgb/Yolo26n_hardneg/weights/best.pt",
      "first hard-neg RGB; best_pre_unfreeze.pt sibling", type="rgb_yolo", lifecycle="superseded", absorbed="hardneg_v3more"),
    M("rgb_hardneg_v2", "Yolo26n_hardneg_v2", "models/rgb/Yolo26n_hardneg_v2/weights/best.pt",
      "hard-neg v2", type="rgb_yolo", lifecycle="superseded", absorbed="hardneg_v3more"),
    M("rgb_hardneg_v2_f5", "Yolo26n_hardneg_v2_f5", "models/rgb/Yolo26n_hardneg_v2_f5/weights/best.pt",
      "hard-neg v2, freeze=5 variant", type="rgb_yolo", lifecycle="superseded", absorbed="hardneg_v3more"),
    M("rgb_hardneg_v3", "Yolo26n_hardneg_v3", "models/rgb/Yolo26n_hardneg_v3/weights/best.pt",
      "precursor to v3_more", type="rgb_yolo", lifecycle="superseded", absorbed="hardneg_v3more"),
    M("selcom_ft1", "Yolo26n_selcom_ft1", "models/rgb/Yolo26n_selcom_ft1/weights/best.pt",
      "pure-selcom precursor (104-img val); §3.4 note", type="rgb_yolo", lifecycle="superseded", absorbed="selcom_mixed_ft2_1280"),
    M("selcom_mixed_ft1", "Yolo26n_selcom_mixed_ft1", "models/rgb/Yolo26n_selcom_mixed_ft1/weights/best.pt",
      "587-img single-clip precursor (P0.84/R0.72/F1 0.77, not comparable)", type="rgb_yolo", lifecycle="superseded", absorbed="selcom_mixed_ft2_1280"),
    M("selcom_mixed_ft2_640", "Yolo26n_selcom_mixed_ft2", "models/rgb/Yolo26n_selcom_mixed_ft2/weights/best.pt",
      "ft2 @640 (F1 0.345); superseded by ft2_1280 resolution win", type="rgb_yolo", lifecycle="superseded", absorbed="selcom_mixed_ft2_1280"),
    M("selcom_mixed_ft3_960", "Yolo26n_selcom_mixed_ft3_960", "models/rgb/Yolo26n_selcom_mixed_ft3_960/weights/best.pt",
      "ft3 @960 (F1 0.571; +0.014 baseline vs ft2); candidate sibling of ft3_1280", type="rgb_yolo", lifecycle="active"),
    # --- IR corrective-finetune + dataset-V3 runs (present) ---
    M("ir_ft_v1", "IR corrective finetune_v1", "models/ir/corrective_finetune/finetune_v1/weights/best.pt",
      "corrective-finetune lineage step 1 (on top of Final-family)", lifecycle="superseded", absorbed="ir_v3b"),
    M("ir_ft_v2", "IR corrective finetune_v2", "models/ir/corrective_finetune/finetune_v2/weights/best.pt",
      "corrective-finetune lineage step 2", lifecycle="superseded", absorbed="ir_v3b"),
    M("ir_dsetv3_run", "IR_FT_dsetV3_aug0_s0", "runs/IR_FT_dsetV3_aug0_s0/weights/best.pt",
      "dataset-V3 finetune run (aug0); archive copy at IR_FT_dsetV3_run_135ep/", lifecycle="superseded", absorbed="ir_final"),
    # --- IR archive lineage (archived; resurrect-worthy, NOT obsolete) ---
    M("ir_v1", "IR_dsetV1_goldV2_300ep", "archive/models/ir/IR_dsetV1_goldV2_300ep/best.pt",
      "IR dataset V1 (goldV2 labels, 300ep); earliest IR detector. Only copy in archive/.", train="IR dsetV1"),
    M("ir_v7", "IR_FT_dsetV7_aug1_clahe_300ep", "archive/IR_FT_dsetV7_aug1_clahe_300ep_s0/IR_FT_dsetV7_aug1_clahe_300ep_s0/weights/best.pt",
      "IR dataset V7, aug1+CLAHE, 300ep", train="IR dsetV7"),
    M("ir_v7b", "IR_dsetV7b_110ep", "archive/models/ir/IR_dsetV7b_110ep/best.pt",
      "IR dataset V7b, 110ep", train="IR dsetV7b"),
    M("ir_v8", "IR_FT_dsetV8_aug1_noclahe_200ep", "archive/IR_FT_dsetV8_aug1_noclahe_200ep_s0/IR_FT_dsetV8_aug1_noclahe_200ep_s0/weights/best.pt",
      "IR dataset V8, aug1 no-CLAHE, 200ep", train="IR dsetV8"),
    M("ir_v9b1", "IR_dsetV9b1_rgbcfg", "archive/models2/IR_dsetV9b1_rgbcfg/weights/best.pt",
      "IR dataset V9b1 (rgb-config); DUPLICATE copy at archive/models2/models_/IR_dsetV9b1_rgbcfg/", train="IR dsetV9b1"),
    M("ir_gold_rgbcfg", "IR_gold_rgbcfg", "archive/models2/IR_gold_rgbcfg/IR_gold_rgbcfg/weights/best.pt",
      "IR gold-labels rgb-config; DUPLICATE at archive/models2/models_/IR_gold_rgbcfg/", train="IR gold"),
    M("ir_cst_stage2", "IR_FT_cst_stage2_s0", "archive/IR_FT_cst_stage2_s0/weights/best.pt",
      "IR consistency/self-training stage-2 experiment"),
    M("ir_goldv2_pilot", "IR_FT_goldV2_IRdsetV1_aug0", "archive/results_vast_ai_300ep/runs/detect/runs/IR_FT_goldV2_IRdsetV1_aug0_s0/weights/best.pt",
      "goldV2 pilot on dsetV1; vast.ai runs. DUP copies: results_vastai_first_run_70ep/ (70ep) + runs/detect/.../_pilot/"),
    M("ir_gold_pilot", "IR_FT_gold_IRdsetV1_pilot", "archive/runs/detect/runs/IR_FT_gold_IRdsetV1_aug0_s0_pilot/weights/best.pt",
      "gold-labels pilot on dsetV1"),
    M("ir_silver_pilot", "IR_FT_silver_IRdsetV1_pilot", "archive/runs/detect/runs/IR_FT_silver_IRdsetV1_aug0_s0_pilot/weights/best.pt",
      "silver-labels pilot on dsetV1"),
    M("ir_v7_pilot", "IR_dsetV7_pilot_conservative", "archive/runs/detect/runs/IR_dsetV7_pilot_conservative/weights/best.pt",
      "V7 conservative-config pilot"),
    M("ir_sweep_aug1_full", "sweep_aug1_full_s0", "archive/runs/detect/runs/sweep_aug1_full_s0/weights/best.pt",
      "augmentation sweep: aug1 full"),
    M("ir_sweep_aug_ir", "sweep_aug_ir_s0", "archive/runs/detect/runs/sweep_aug_ir_s0/weights/best.pt",
      "augmentation sweep: IR-specific aug"),
    M("ir_sweep_no_tiny8", "sweep_no_tiny8_s0", "archive/runs/detect/runs/sweep_no_tiny8_s0/weights/best.pt",
      "augmentation sweep: no tiny-object-8"),
    M("ir_merged_v2_ancestor", "IR_FT_merged_IRdsetV2_aug0", "archive/run_300_ep_dsetv2golden_may22_merged/ES_Drone_Detection/runs/detect/runs/IR_FT_merged_IRdsetV2_aug0_s0/weights/best.pt",
      "merged-dsetV2 training ancestor of ir_v2 (300ep golden may22 run)"),
]


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


def _note(id, addition):
    cur = kb._read("models"); by = {r["id"]: r for r in cur}
    if id in by and addition not in (by[id].get("provenance_notes") or ""):
        by[id]["provenance_notes"] = (by[id].get("provenance_notes") or "").rstrip(". ") + ". " + addition
        kb._write("models", cur)
        print(f"models.{id}: note appended")


def main():
    _append("models", MODELS)
    _note("ft4", "FT4 ablation checkpoints live in models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights_{A1_600hn_f15_noextra,A2_600hn_f15_4000xp,A3_600hn_f15_8800xp,failed_R1_600hn_3ep_f12_lr5e6,failed_R2_300hn_3ep_f12_lr5e6}/ - see ft4-backbone-freeze finding.")
    _note("baseline", "Archived training-ancestor copy at archive/run_300_ep_dsetv2golden_may22_merged/.../Yolo26n_trained/weights/best.pt.")
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
