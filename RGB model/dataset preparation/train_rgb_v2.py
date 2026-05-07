"""
train_rgb_v2.py — Train YOLOv26n from scratch on the merged dataset.
Reproduces the exact training config from the original Yolo26n_trained run.

Usage (on vast.ai):
    python train_rgb_v2.py
    python train_rgb_v2.py --data /workspace/retrain_dataset/data.yaml
    python train_rgb_v2.py --epochs 80 --batch 32
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv26n from scratch with negatives")
    ap.add_argument("--data", type=str, default="/workspace/retrain_dataset/data.yaml",
                    help="Path to data.yaml")
    ap.add_argument("--weights", type=str, default="yolo26n.pt",
                    help="Pretrained COCO weights (default: yolo26n.pt)")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--project", type=str, default="/workspace/runs")
    ap.add_argument("--name", type=str, default="Yolo26n_retrained_v2")
    args = ap.parse_args()

    print("=" * 72)
    print(f"Training YOLOv26n from scratch")
    print(f"  Weights: {args.weights}")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print("=" * 72)

    model = YOLO(args.weights)

    # Exact same config as original Yolo26n_trained (from args.yaml)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        amp=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=20,
        iou=0.7,
        patience=10,
        save_period=10,
        workers=6,
        cache=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
        deterministic=True,
        seed=0,
        # Augmentation — same as original
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.33,
        erasing=0.4,
        # Keep defaults for the rest
        plots=True,
        verbose=True,
    )

    for k, v in sorted(train_kwargs.items()):
        print(f"  {k} = {v}")

    model.train(**train_kwargs)

    print("\n" + "=" * 72)
    print("Training complete!")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print("=" * 72)

    # Run test split evaluation
    print("\nRunning test split evaluation...")
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        model_eval = YOLO(str(best_path))
        metrics = model_eval.val(
            data=args.data,
            split="test",
            device=args.device,
        )
        print(f"\nTest Results:")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
        print(f"  mAP@50:    {metrics.box.map50:.4f}")
        print(f"  mAP@50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
