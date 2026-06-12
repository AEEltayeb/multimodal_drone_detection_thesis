"""
compare_confuser_ab.py — A/B old vs new 4-class confuser filter.

Reconstructs the same deterministic val split (seed=42) used by
train_confuser_4class.py, then scores both:
  NEW:  models/patches/confuser_filter4_{modality}.pt
  OLD:  models/patches/confuser_filter4_{modality}_v1_backup.pt
on identical val crops. Reports per-class P/R + veto-threshold sweep
side-by-side. Apples-to-apples.

Usage:
  python classifier/compare_confuser_ab.py
  python classifier/compare_confuser_ab.py --modality rgb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_confuser_4class import (
    CLASS_NAMES, CLASS_TO_IDX, CONFUSER_CLASSES,
    FourClassDataset, sequence_split, build_model, evaluate,
    manifest_category_to_class, MANIFEST_PATH, PATCH_DIR,
)


def load_state(model, pt_path, device):
    obj = torch.load(pt_path, map_location=device, weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(sd)
    return model


def score_one(modality, pt_path, device, batch_size=64):
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["modality"] == modality].copy()
    df["class_idx"] = df["category"].apply(manifest_category_to_class)
    _, va_df = sequence_split(df)

    eval_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        FourClassDataset(va_df, eval_tfm, modality),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = build_model(device, num_classes=len(CLASS_NAMES))
    model = load_state(model, pt_path, device)
    return evaluate(model, loader, device), len(va_df)


def fmt_class_row(name, n_old, n_new):
    return (f"  {name:<10s} "
            f"OLD P={n_old['P']:.3f} R={n_old['R']:.3f} "
            f"TP={n_old['TP']} FP={n_old['FP']} FN={n_old['FN']}  |  "
            f"NEW P={n_new['P']:.3f} R={n_new['R']:.3f} "
            f"TP={n_new['TP']} FP={n_new['FP']} FN={n_new['FN']}")


def fmt_sweep(thr_key, old, new):
    return (f"  {thr_key:<10s} "
            f"OLD vetoP={old['precision_veto']:.3f} vetoR={old['recall_veto']:.3f} "
            f"drones_passed={old['pass_acc_on_drones']:.3f}  |  "
            f"NEW vetoP={new['precision_veto']:.3f} vetoR={new['recall_veto']:.3f} "
            f"drones_passed={new['pass_acc_on_drones']:.3f}")


def compare_modality(modality, device):
    new_pt = PATCH_DIR / f"confuser_filter4_{modality}.pt"
    old_pt = PATCH_DIR / f"confuser_filter4_{modality}_v1_backup.pt"
    if not new_pt.exists() or not old_pt.exists():
        print(f"[skip] {modality}: missing {new_pt.name} or {old_pt.name}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  {modality.upper()} confuser filter — A/B on shared val split (seed=42)")
    print(f"{'=' * 70}")

    print(f"  Scoring NEW {new_pt.name}...")
    new_res, n_val = score_one(modality, new_pt, device)
    print(f"    val_acc={new_res['acc']:.4f}  n_val={n_val}")

    print(f"  Scoring OLD {old_pt.name}...")
    old_res, _ = score_one(modality, old_pt, device)
    print(f"    val_acc={old_res['acc']:.4f}")

    print(f"\n  Overall accuracy:  OLD={old_res['acc']:.4f}   NEW={new_res['acc']:.4f}   "
          f"Δ={new_res['acc'] - old_res['acc']:+.4f}")

    print("\n  Per-class:")
    for c in CLASS_NAMES:
        print(fmt_class_row(c, old_res["per_class"][c], new_res["per_class"][c]))

    print("\n  Veto-threshold sweep (higher vetoR = more confusers killed; "
          "higher drones_passed = safer for real drones):")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        k = f"thr={thr}"
        print(fmt_sweep(k, old_res["reject_sweep"][k], new_res["reject_sweep"][k]))

    return {"old": old_res, "new": new_res, "n_val": n_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out = {}
    mods = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in mods:
        res = compare_modality(m, device)
        if res is not None:
            out[m] = {
                "old_acc": res["old"]["acc"],
                "new_acc": res["new"]["acc"],
                "old_per_class": res["old"]["per_class"],
                "new_per_class": res["new"]["per_class"],
                "old_reject_sweep": res["old"]["reject_sweep"],
                "new_reject_sweep": res["new"]["reject_sweep"],
                "n_val": res["n_val"],
            }

    out_path = PATCH_DIR / "confuser_filter4_ab_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
