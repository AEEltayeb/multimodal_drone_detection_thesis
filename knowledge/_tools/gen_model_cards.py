"""gen_model_cards.py — generate per-model `*.model_card.yaml` provenance sidecars for
the THESIS-COMPARED model set (every model that appears in a thesis ablation/comparison).

A model card is a *generated, co-located VIEW* of the knowledge system — NEVER hand-edit;
re-run this to refresh. `knowledge/models.csv` stays the single source of truth.

Each card states, plainly: what the model was **trained on** and what it was **tested on**.
A dataset that overlaps training is listed under `trained_on` (it is not presented as a
test). For detectors the train/val/test splits are recovered from the run's data.yaml; for
the filters whose corpora were reconstructed
(`docs/analysis/2026-06-18_filter_provenance_train_heldout.md`) the held-out test surfaces
carry their measured results.

Card location: next to the weight, `<weight-stem>.model_card.yaml`.

  py knowledge/_tools/gen_model_cards.py            # write all cards
  py knowledge/_tools/gen_model_cards.py --dry-run  # print, write nothing
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MODELS_CSV = REPO / "knowledge" / "models.csv"
MANIFEST = REPO / "thesis_eval" / "_filter_swap" / "final" / "swap_manifest.json"
PROV_DOC = "docs/analysis/2026-06-18_filter_provenance_train_heldout.md"

try:
    import yaml
except Exception:  # pragma: no cover
    print("PyYAML required (pip install pyyaml)"); sys.exit(1)

# kb_id -> the thesis ablation/comparison it appears in (why it earns a card)
THESIS_SET = {
    # --- RGB detectors: §3.1 / §3.4 detector ablation ---
    "baseline":              "RGB detector ablation (§3.1)",
    "hardneg_v3more":        "RGB detector ablation (§3.1, §3.3)",
    "retrained_v2":          "RGB detector ablation (§3.1)",
    "selcom_mixed_ft2_1280": "RGB CCTV detector ablation (§3.4)",
    "selcom_mixed_ft3_1280": "RGB CCTV detector ablation (§3.4)",
    "ft4":                   "RGB production detector; verifier-stack feature source (§13)",
    # --- IR detectors: §4.1 IR evolution ---
    "ir_v2":   "IR detector evolution (§4.1)",
    "ir_v3":   "IR detector evolution (§4.1)",
    "ir_v4":   "IR detector evolution (§4.1)",
    "ir_v5":   "IR detector evolution (§4.1)",
    "ir_v6":   "IR detector evolution (§4.1)",
    "ir_final":"IR detector evolution (§4.1)",
    "ir_v3b":  "IR production detector (§4.1)",
    # --- Filters: pipeline ablation / 2026-06-18 filter swap ---
    "patch_v2":                  "Patch verifier (filter ablation §6, §13)",
    "mlp_v5":                    "RGB confuser filter — predecessor (§13)",
    "mlp_v5_balanced_v4":        "RGB confuser filter — production (filter swap 2026-06-18)",
    "mlp_v5_ir":                 "IR confuser filter — native variant (native-vs-aligned ablation)",
    "mlp_v5_ir_aligned":         "IR confuser filter — predecessor (aligned)",
    "mlp_aligned_thermalonly":   "IR thermal confuser filter — production (filter swap 2026-06-18)",
    "mlp_aligned_gray_balanced": "IR grayscale confuser filter — production (filter swap 2026-06-18)",
    # --- Classifiers: trust-router ablation (§sec:classifier_results) ---
    "sa32":            "Trust-router ablation",
    "robust6":         "Trust-router ablation (6 features)",
    "robust8":         "Trust-router ablation (8 features)",
    "robust8_nr_drop": "Trust router — production (no-reject)",
}

# Models in a thesis ablation with NO knowledge/models.csv row (synthetic rows).
EXTRAS = {
    "ir_aligned_balanced": {
        "name": "mlp_v5_ir_aligned balanced (thermal-confusers, no CBAM)",
        "type": "mlp", "production": "no",
        "weights_path": "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt",
        "trained_from_script": "mri/train_aligned.py --thermal-confusers",
        "train_dataset": "thermal drones + balanced IR_confusers/train (category x size) + grayscale-harvested confusers; no CBAM",
        "provenance_notes": "",
        "_role": "IR confuser filter — balanced thermal variant (comparison)",
    },
}

# Reconstructed corpora (from PROV_DOC): explicit trained_on / tested_on, plain language.
V3B = "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
FT4 = "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
AUDITED = {
    "mlp_v5_balanced_v4": {
        "feature_detector": FT4, "deploy_threshold": 0.25,
        "trained_on": [
            "rgb_dataset drones — dataset/dataset/images/{train,val} (balanced by sub-source x size)",
            "Anti-UAV val/RGB drones",
            "SelCom CCTV — 833 drones + 149 confusers (pure-selcom; the 311 SelCom-val images excluded; imgsz 1280)",
            "Svanstrom RGB drones (imgsz 1280)",
            "confusers — rgb_confusers_merged/images/{train,val}, RGB_video, Svanstrom drone-empty frames",
            "bird.v1i — 728 of 1212 images (60% train split, seed 0), as confusers",
        ],
        "tested_on": [
            {"surface": "rgb_dataset test", "data": "dataset/dataset/images/test", "drone_recall": 0.874},
            {"surface": "Anti-UAV RGB test", "data": "Anti-UAV/test/RGB", "drone_recall": 0.982},
            {"surface": "SelCom val", "data": "_finetune_selcom_mixed_ft2/images/val", "drone_recall": 0.451},
            {"surface": "RGB confusers test", "data": "rgb_confusers_merged/images/test", "result": "confuser fires held"},
            {"surface": "bird.v1i held-out test", "data": "bird.v1i 484 of 1212 (40% test split, not trained on)",
             "result": "30 of 230 fires kept (predecessor mlp_v5: 91 of 230)"},
        ],
    },
    "mlp_v5": {
        "feature_detector": FT4, "deploy_threshold": 0.25,
        "trained_on": [
            "distilled FT4 p3+p5 ROI features (517-D)",
            "SelCom CCTV (pure_1x8), Svanstrom, Anti-UAV, rgb_confusers, RGB_video",
            "rgb_dataset drones via the parent distill (alphabetical stride-8, 8000-drone quota)",
        ],
        "tested_on": [
            {"surface": "rgb_dataset test", "data": "dataset/dataset/images/test", "drone_recall": 0.694},
        ],
        "notes": "Predecessor RGB filter; superseded in production by mlp_v5_balanced_v4 (2026-06-18).",
    },
    "mlp_aligned_thermalonly": {
        "feature_detector": V3B, "deploy_threshold": 0.05,
        "trained_on": [
            "thermal drones (8112) — Svanstrom IR, Anti-UAV val/IR, IR_dset_final/train, IR_video/train, CBAM/train (GT class D = drone)",
            "thermal confusers (2045) — IR_confusers/train, Svanstrom IR, IR_video/train, CBAM/train (balanced by category x size, cap 1000)",
        ],
        "tested_on": [
            {"surface": "CBAM", "data": "CBAM/valid", "drone_recall": 0.967},
            {"surface": "IR confusers", "data": "IR_confusers/{val,test}", "result": "94% of confuser fires removed (@0.05)"},
            {"surface": "IR_dset_final test", "data": "IR_dset_final/test", "drone_recall": 0.928},
            {"surface": "Anti-UAV IR test", "data": "Anti-UAV/test/IR", "drone_recall": 0.937},
            {"surface": "IR video test", "data": "IR_video/test", "drone_recall": 0.971},
        ],
        "notes": "Thermal head only; paired with the grayscale head mlp_aligned_gray_balanced (two checkpoints).",
    },
    "mlp_aligned_gray_balanced": {
        "feature_detector": V3B + " (RGB converted to grayscale)", "deploy_threshold": 0.25,
        "trained_on": [
            "thermal drones + grayscale-harvested confusers (rgb_confusers, RGB_video, Svanstrom RGB converted to grayscale); from the balanced run",
        ],
        "tested_on": [
            {"surface": "grayscale confusers", "data": "rgb_confusers_merged/images/test (converted to grayscale)",
             "result": "15 fires kept (predecessor grayscale head: 21)"},
        ],
        "notes": "Grayscale-fallback head; paired with the thermal head mlp_aligned_thermalonly.",
    },
    "mlp_v5_ir_aligned": {
        "feature_detector": V3B, "deploy_threshold": 0.05,
        "trained_on": [
            "thermal drones (Svanstrom IR, Anti-UAV val/IR, IR_dset_final/train, IR_video/train)",
            "grayscale-harvested confusers, per-modality z-aligned (one network, thermal + grayscale deploy scalers)",
        ],
        "tested_on": [
            {"surface": "CBAM", "data": "CBAM/valid", "drone_recall": 0.917},
            {"surface": "IR_dset_final test", "data": "IR_dset_final/test", "drone_recall": 0.965},
        ],
        "notes": "Predecessor IR aligned filter; superseded by the thermal + grayscale two-checkpoint pair (2026-06-18).",
    },
    "mlp_v5_ir": {
        "feature_detector": V3B, "deploy_threshold": 0.05,
        "trained_on": ["native thermal V5 distillation (no z-alignment) — thermal drones + thermal confusers"],
        "tested_on": [
            {"surface": "IR confusers", "data": "IR_confusers/{val,test}", "result": "~24% of confuser fires removed"},
        ],
        "notes": "Native (un-aligned) IR filter variant in the native-vs-aligned comparison.",
    },
    "ir_aligned_balanced": {
        "feature_detector": V3B, "deploy_threshold": "0.01-0.05",
        "trained_on": ["thermal drones + IR_confusers/train (balanced by category x size) + grayscale-harvested confusers; no CBAM"],
        "tested_on": [
            {"surface": "CBAM", "data": "CBAM/valid", "drone_recall": "0.717 (@0.01) / 0.600 (@0.05)"},
            {"surface": "IR confusers", "data": "IR_confusers/{val,test}", "result": "92-98% of confuser fires removed"},
        ],
        "notes": "Comparison candidate; its grayscale head was promoted (mlp_aligned_gray_balanced) while the thermal head shipped from the CBAM-trained run instead.",
    },
}


def load_rows():
    rows = {}
    with open(MODELS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = r
    return rows


def load_manifest():
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text())
    out = {}
    for slot, w in m.get("weights", {}).items():
        out[Path(w["path"]).name] = {"sha256_16": w["sha256_16"],
                                     "threshold": m.get("thresholds", {}).get(slot)}
    return out


def detector_splits(weight: Path):
    """Recover trained_on/tested_on for a YOLO detector from its run args.yaml -> data.yaml."""
    args_path = next((c for c in (weight.parent.parent / "args.yaml", weight.parent / "args.yaml")
                      if c.exists()), None)
    if not args_path:
        return None
    try:
        a = yaml.safe_load(args_path.read_text())
    except Exception:
        return None
    bm = a.get("model")
    if bm:
        bp = Path(str(bm))
        bm = bp.parent.parent.name if (bp.name in ("best.pt", "last.pt") and bp.parent.name == "weights") else bp.name
    info = {"args_yaml": str(args_path.relative_to(REPO)).replace("\\", "/"),
            "epochs": a.get("epochs"), "imgsz": a.get("imgsz"), "base_model": bm}
    data = a.get("data")
    info["data_yaml"] = data
    dy = next((p for p in (Path(str(data)), REPO / str(data)) if data and p.exists()), None)
    if dy:
        try:
            d = yaml.safe_load(dy.read_text())
            base = d.get("path")
            root = (dy.parent / base) if (base and not Path(base).is_absolute()) else (Path(base) if base else dy.parent)
            res = lambda x: None if not x else (str(x) if Path(str(x)).is_absolute() else str(root / str(x)))
            info["train"] = res(d.get("train")); info["val"] = res(d.get("val")); info["test"] = res(d.get("test"))
        except Exception:
            pass
    return info


def build_card(mid, role, row, manifest):
    weight_rel = row["weights_path"].replace("\\", "/")
    is_detector = row["type"].endswith("yolo")
    card = {"model": mid, "name": row["name"], "type": row["type"],
            "role": role, "production": row.get("production") == "yes",
            "weights": weight_rel}
    man = manifest.get(Path(weight_rel).name)
    if man:
        card["sha256_16"] = man["sha256_16"]
        if man.get("threshold"):
            card["deploy_threshold"] = man["threshold"]

    aud = AUDITED.get(mid)
    det = detector_splits(REPO / weight_rel) if is_detector else None

    if aud:
        if aud.get("feature_detector"):
            card["feature_detector"] = aud["feature_detector"]
        if aud.get("deploy_threshold") and "deploy_threshold" not in card:
            card["deploy_threshold"] = aud["deploy_threshold"]
        card["trained_on"] = aud["trained_on"]
        card["tested_on"] = aud["tested_on"]
    elif det:
        trained = []
        if row.get("train_dataset"):
            trained.append(row["train_dataset"])
        if det.get("train"):
            trained.append(f"train split: {det['train']}")
        card["trained_on"] = trained or ["(see knowledge/models.csv)"]
        tested = []
        if det.get("test"):
            tested.append({"surface": "test split", "data": det["test"]})
        if det.get("val"):
            tested.append({"surface": "val split", "data": det["val"]})
        card["tested_on"] = tested or ["(thesis §4.1 / §3.1 surfaces; see knowledge/models.csv)"]
        td = {k: det[k] for k in ("base_model", "epochs", "imgsz", "data_yaml", "args_yaml")
              if det.get(k) is not None}
        if td:
            card["training_config"] = td
    else:
        card["trained_on"] = [row["train_dataset"]] if row.get("train_dataset") else ["(see knowledge/models.csv)"]
        card["tested_on"] = ["(thesis ablation surfaces; see knowledge/models.csv)"]

    if row.get("trained_from_script"):
        card["training_script"] = row["trained_from_script"]
    card["generated_by"] = ("knowledge/_tools/gen_model_cards.py -- DO NOT hand-edit; re-run to refresh")
    return card, weight_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # cards carry unicode (§) -> utf-8 console
    except Exception:
        pass
    rows = load_rows()
    manifest = load_manifest()
    written, skipped = [], []
    items = list(THESIS_SET.items()) + [(k, v["_role"]) for k, v in EXTRAS.items()]
    for mid, role in items:
        row = rows.get(mid) or EXTRAS.get(mid)
        if row is None:
            skipped.append((mid, "no kb row and not in EXTRAS")); continue
        card, weight_rel = build_card(mid, role, row, manifest)
        weight = REPO / weight_rel
        if not weight.parent.exists():
            skipped.append((mid, f"weight dir missing: {weight.parent.relative_to(REPO)}")); continue
        out = weight.parent / (weight.stem + ".model_card.yaml")
        txt = yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=100)
        if a.dry_run:
            print(f"\n===== {out.relative_to(REPO)} =====\n{txt}")
        else:
            out.write_text(txt, encoding="utf-8")
        written.append((mid, str(out.relative_to(REPO)).replace("\\", "/")))
    print(f"\n{'DRY-RUN: would write' if a.dry_run else 'wrote'} {len(written)} model cards:")
    for mid, p in written:
        tag = "[audited]" if mid in AUDITED else ("[detector]" if rows.get(mid, {}).get("type", "").endswith("yolo") else "[kb]")
        print(f"  {tag:<11} {mid:<28} -> {p}")
    if skipped:
        print(f"\nSKIPPED {len(skipped)} (no resident weight path):")
        for mid, why in skipped:
            print(f"  {mid:<28} {why}")


if __name__ == "__main__":
    main()
