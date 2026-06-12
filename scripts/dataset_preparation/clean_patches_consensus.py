"""
clean_patches_consensus.py — Use consensus of old + new confuser filters
to identify and remove mislabeled/noisy crops from the manifest.

Logic:
  - Load v2 (backup) and v3 (current) models
  - Run both on every crop
  - If BOTH models confidently predict a DIFFERENT class than the label → flag
  - Remove flagged crops from manifest, retrain

Usage:
    python clean_patches_consensus.py --dry-run     # preview only
    python clean_patches_consensus.py               # clean and save
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"
PROJECT_ROOT = SCRIPT_DIR.parent

CLASS_NAMES = ["airplane", "helicopter", "bird", "other"]
CONFUSER_CLASSES = {"airplane", "helicopter", "bird"}


def load_model(weights_path, device):
    net = mobilenet_v3_small(weights=None)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(in_features, 4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net


def predict_crop(model, img_path, tfm, device):
    """Return (predicted_class_idx, confidence)."""
    p = img_path if Path(img_path).is_absolute() else str(PROJECT_ROOT / img_path)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return -1, 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    conf = float(probs[pred])
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview flagged crops without modifying manifest")
    parser.add_argument("--conf-thresh", type=float, default=0.7,
                        help="Min confidence for both models to agree on wrong class")
    parser.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest: {len(manifest)} rows")

    modalities = ["rgb", "ir"] if args.modality == "both" else [args.modality]

    total_flagged = 0
    rows_to_remove = set()

    for mod in modalities:
        v2_path = PATCH_DIR / f"confuser_filter4_{mod}_v2_backup.pt"
        v3_path = PATCH_DIR / f"confuser_filter4_{mod}.pt"

        if not v2_path.exists():
            # Try v1 backup
            v2_path = PATCH_DIR / f"confuser_filter4_{mod}_v1_backup.pt"
        if not v2_path.exists():
            print(f"  [SKIP] No backup model for {mod}")
            continue

        print(f"\n{'='*60}")
        print(f"Consensus cleaning: {mod.upper()}")
        print(f"{'='*60}")
        print(f"  Old model: {v2_path.name}")
        print(f"  New model: {v3_path.name}")

        model_old = load_model(v2_path, device)
        model_new = load_model(v3_path, device)

        df_mod = manifest[manifest["modality"] == mod].copy()
        print(f"  Crops: {len(df_mod)}")

        flagged = []
        agree_correct = 0
        agree_wrong = 0
        disagree = 0
        disagree_both_wrong = 0

        for idx, row in df_mod.iterrows():
            gt_cat = row["category"]
            if gt_cat in CONFUSER_CLASSES:
                gt_idx = CLASS_NAMES.index(gt_cat)
            else:
                gt_idx = CLASS_NAMES.index("other")

            pred_old, conf_old = predict_crop(model_old, row["path"], tfm, device)
            pred_new, conf_new = predict_crop(model_new, row["path"], tfm, device)

            if pred_old == -1 or pred_new == -1:
                # Missing file
                rows_to_remove.add(idx)
                continue

            if pred_old == pred_new:
                if pred_old == gt_idx:
                    agree_correct += 1
                else:
                    agree_wrong += 1
                    # Both models agree it's mislabeled AND both confident
                    if conf_old >= args.conf_thresh and conf_new >= args.conf_thresh:
                        flagged.append({
                            "idx": idx,
                            "path": row["path"],
                            "label": gt_cat,
                            "pred": CLASS_NAMES[pred_old],
                            "conf_old": round(conf_old, 3),
                            "conf_new": round(conf_new, 3),
                        })
                        rows_to_remove.add(idx)
            else:
                disagree += 1
                # Neither model predicts GT → both wrong, crop is noisy
                if pred_old != gt_idx and pred_new != gt_idx:
                    disagree_both_wrong += 1
                    flagged.append({
                        "idx": idx,
                        "path": row["path"],
                        "label": gt_cat,
                        "pred": f"{CLASS_NAMES[pred_old]}/{CLASS_NAMES[pred_new]}",
                        "conf_old": round(conf_old, 3),
                        "conf_new": round(conf_new, 3),
                    })
                    rows_to_remove.add(idx)

            if (idx + 1) % 1000 == 0:
                print(f"    [{agree_correct+agree_wrong+disagree}/{len(df_mod)}] "
                      f"flagged={len(flagged)}")

        print(f"\n  Results:")
        print(f"    Both agree correct: {agree_correct}")
        print(f"    Both agree WRONG:   {agree_wrong}")
        print(f"    Disagree:           {disagree} ({disagree_both_wrong} both wrong)")
        print(f"    Flagged total:      {len(flagged)}")
        total_flagged += len(flagged)

        # Show breakdown of flagged crops
        if flagged:
            print(f"\n  Flagged breakdown (label -> predicted):")
            transitions = Counter(
                f"{f['label']} -> {f['pred']}" for f in flagged)
            for trans, count in transitions.most_common():
                print(f"    {trans}: {count}")

            # Show some examples
            print(f"\n  Examples (first 10):")
            for f in flagged[:10]:
                print(f"    {Path(f['path']).name}: "
                      f"{f['label']} -> {f['pred']} "
                      f"(old={f['conf_old']}, new={f['conf_new']})")

    print(f"\n{'='*60}")
    print(f"TOTAL FLAGGED: {total_flagged}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] No changes made. Run without --dry-run to apply.")
    else:
        if rows_to_remove:
            before = len(manifest)
            manifest = manifest.drop(index=list(rows_to_remove))
            manifest.to_csv(MANIFEST_PATH, index=False)
            print(f"\n  Removed {before - len(manifest)} rows from manifest.")
            print(f"  New manifest: {len(manifest)} rows")
            print(f"\n  Re-run train_confuser_4class.py to retrain.")
        else:
            print("\n  Nothing to remove.")


if __name__ == "__main__":
    main()
