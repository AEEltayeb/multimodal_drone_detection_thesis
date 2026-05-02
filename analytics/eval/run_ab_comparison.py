"""
run_ab_comparison.py — Overnight A/B comparison on vast.ai.

Runs two training jobs sequentially:
  A) Baseline (standard CIoU matching)
  B) Hybrid IoG+IoP matching (relaxed assignment)

Both use the same cleaned dataset, same hyperparameters (matching dsetV6 config).
No augmentations — this is a clean baseline comparison.

Usage:
    python run_ab_comparison.py --data /IR_dsetV6/IR_dsetV6/data.yaml --weights /workspace/best.pt
"""
import sys
import argparse
from pathlib import Path

# ── Make sure patch is importable ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from iop_loss_patch import apply_patch
from ultralytics import YOLO


def train_run(name, variant, data, weights, epochs, patience, batch, imgsz, device):
    print("\n" + "=" * 70)
    print(f"  RUN: {name}")
    print(f"  Variant: {variant}")
    print(f"  Weights: {weights}")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    print("=" * 70 + "\n")

    # Apply or restore patch
    apply_patch(variant=variant)

    # Fresh model from pretrained weights
    model = YOLO(weights)

    # Train — matching dsetV6 config (YOLO default augs, no custom IR augs)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name=name,
        exist_ok=True,
        patience=patience,
        save_period=25,
        plots=True,
        val=True,
        verbose=True,
        # Hyperparams (matching previous dsetV6 run)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=20,
        single_cls=True,
        amp=False,
        seed=0,
        deterministic=True,
        # YOLO default augs are ON (mosaic, fliplr, scale, translate, etc.)
        # Only custom IR augs (polarity flip, sensor noise) are not used
    )

    # Val
    print(f"\n--- Validating {name} ---")
    metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device)

    # Test
    try:
        print(f"\n--- Testing {name} ---")
        model.val(data=data, imgsz=imgsz, batch=batch, device=device, split="test")
    except Exception as e:
        print(f"  Test split skipped: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/IR_dsetV6/IR_dsetV6/data.yaml")
    parser.add_argument("--weights", type=str, default="/workspace/best.pt",
                        help="Pretrained weights (yolo26n drone detector)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║     A/B COMPARISON: BASELINE vs IoG+IoP         ║")
    print("║     No augmentation — clean baseline             ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Data:     {args.data}")
    print(f"  Weights:  {args.weights}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Batch:    {args.batch}")

    # ── Run A: Baseline ──
    train_run(
        name="IR_dsetV7_baseline",
        variant="baseline",
        data=args.data,
        weights=args.weights,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

    # ── Run B: IoG+IoP Matching ──
    train_run(
        name="IR_dsetV7_iog_matching",
        variant="iog_matching",
        data=args.data,
        weights=args.weights,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("  BOTH RUNS COMPLETE!")
    print("  Results in: runs/IR_dsetV7_baseline/")
    print("              runs/IR_dsetV7_iog_matching/")
    print("=" * 70)


if __name__ == "__main__":
    main()
