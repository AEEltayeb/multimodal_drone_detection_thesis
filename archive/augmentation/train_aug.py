"""
train_aug.py — Config-driven YOLO training with IR augmentation pipeline.

This is the augmented version of train.py. It adds IR-specific augmentations
(polarity flip, contrast degradation, sensor noise, defocus, motion blur, etc.)
via Ultralytics callbacks.

Usage:
    python scripts/train_aug.py --config configs/ir_finetune_pub_v6.yaml
    python scripts/train_aug.py --config configs/ir_finetune_pub_v6.yaml --device-profile configs/devices/vastai_a100.yaml

The config MUST contain:
    augmentation_id: "aug1"
    augmentation_profiles:
      aug1:
        polarity_flip_prob: 0.3
        ...

If augmentation_id is missing, it falls back to standard train.py behavior.
"""

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Reuse all config/logging/env utilities from train.py
from scripts.train import (
    load_config,
    resolve_output_dir,
    setup_logging,
    capture_environment,
    capture_git_commit,
)


def train_with_augmentation(cfg: dict, resume_from: str = None):
    """Run YOLO training with IR-specific augmentations."""
    from ultralytics import YOLO

    run_name = cfg["run_name"]
    run_grade = cfg.get("run_grade", "PUB")
    out_dir = resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)

    # ── Log run grade ──
    if run_grade == "EXP":
        logger.warning("=" * 60)
        logger.warning("  RUN GRADE: EXP (pilot/exploratory)")
        logger.warning("  This run CANNOT be used for publication results.")
        if cfg.get("_device_profile_used"):
            logger.warning(f"  Device profile: {cfg['_device_profile_used']}")
        logger.warning("=" * 60)
    else:
        logger.info(f"  RUN GRADE: PUB (publication-quality)")

    # ── Archive resolved config ──
    config_to_save = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_archive = out_dir / "config.yaml"
    with open(config_archive, "w", encoding="utf-8") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Archived resolved config to: {config_archive}")

    logger.info("Fully resolved config:")
    for k, v in config_to_save.items():
        logger.info(f"  {k}: {v}")

    # ── Determine weights ──
    if resume_from:
        weights = resume_from
        resume = True
        logger.info(f"Resuming from: {weights}")
    else:
        weights = cfg.get("pretrained_weights", "yolo26n.pt")
        resume = False
        logger.info(f"Starting from weights: {weights}")

    # ── Build model ──
    model = YOLO(weights)

    # ── Map config → ultralytics train() kwargs ──
    train_kwargs = {
        "data": cfg["dataset_yaml"],
        "epochs": cfg.get("epochs", 70),
        "batch": cfg.get("batch_size", 48),
        "imgsz": cfg.get("image_size", 640),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": cfg.get("learning_rate", 0.001),
        "lrf": cfg.get("lr_final_fraction", 0.01),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "close_mosaic": cfg.get("close_mosaic", 20),
        "amp": cfg.get("amp", True),
        "seed": cfg.get("seed", 0),
        "device": cfg.get("device", "0"),
        "single_cls": cfg.get("single_class", True),
        "iou": cfg.get("iou_threshold", 0.7),
        "project": str(out_dir.parent),
        "name": out_dir.name,
        "exist_ok": True,
        "resume": resume,
        "deterministic": True,
        "verbose": True,
        "save": True,
        "save_period": 10,
        "plots": True,
        "val": True,
    }

    # Multi-scale training: scale=0.25 means imgsz varies +-25% (480-800 at imgsz=640)
    if cfg.get("scale") is not None:
        train_kwargs["scale"] = cfg["scale"]

    # ── Cache setting ──
    if "cache" in cfg:
        train_kwargs["cache"] = cfg["cache"]

    # ── Patience (early stopping) ──
    if "patience" in cfg:
        train_kwargs["patience"] = cfg["patience"]

    # ── Workers ──
    if "workers" in cfg:
        train_kwargs["workers"] = cfg["workers"]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  TRAINING (augmented): {run_name}")
    logger.info(f"  Grade:    {run_grade}")
    logger.info(f"  Output:   {out_dir}")
    logger.info(f"  Dataset:  {cfg['dataset_yaml']}")
    logger.info(f"  Epochs:   {train_kwargs['epochs']}")
    logger.info(f"  Batch:    {train_kwargs['batch']}")
    logger.info(f"  ImgSz:    {train_kwargs['imgsz']}")
    logger.info(f"  AMP:      {train_kwargs['amp']}")
    logger.info(f"  WD:       {train_kwargs['weight_decay']}")
    logger.info(f"  Mosaic off after epoch: {train_kwargs['epochs'] - train_kwargs['close_mosaic']}")
    logger.info(f"  Device:   {train_kwargs['device']}")
    logger.info("=" * 60)

    # ── Register IR augmentations ──
    aug_id = cfg.get("augmentation_id")
    if aug_id:
        aug_profiles = cfg.get("augmentation_profiles", {})
        if aug_id in aug_profiles:
            aug_cfg = aug_profiles[aug_id]
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"  IR AUGMENTATION PIPELINE: {aug_id}")
            logger.info(f"  {aug_cfg.get('description', '')}")
            logger.info(f"pct_norm -> local_contrast -> polarity -> bg_shift -> gamma -> contrast -> noise -> defocus -> motion_blur")
            logger.info("=" * 60)

            # Log individual augmentation probabilities
            for key, val in sorted(aug_cfg.items()):
                if key != "description":
                    logger.info(f"    {key}: {val}")

            from scripts.augmentation.ir_augmentor import register_ir_augmentations
            register_ir_augmentations(model, aug_cfg)
            logger.info(f"  ✅ Registered IR augmentation callback")
            logger.info("")
        else:
            logger.error(f"  Augmentation '{aug_id}' not found in augmentation_profiles!")
            logger.error(f"  Available profiles: {list(aug_profiles.keys())}")
            sys.exit(1)
    else:
        logger.warning("  No augmentation_id in config — running WITHOUT IR augmentations")
        logger.warning("  (Use scripts/train.py instead for non-augmented training)")

    # ── Register source-balanced sampler ──
    if cfg.get("source_balanced_sampling", False):
        from scripts.dataset.source_balanced_sampler import register_balanced_sampler
        dataset_yaml_path = cfg["dataset_yaml"]
        register_balanced_sampler(model, dataset_yaml_path)
        logger.info("  ✅ Source-balanced sampling enabled")
    else:
        logger.info("  Source-balanced sampling: OFF (default random sampling)")

    # ── YOLO default overrides for IR ──
    yolo_override_keys = [
        "hsv_h", "hsv_s", "hsv_v", "erasing",
        "fliplr", "flipud", "translate", "scale",
        "degrees", "shear", "perspective",
        "mosaic", "mixup", "cutmix", "copy_paste", "multi_scale",
    ]
    ir_overrides = {k: cfg[k] for k in yolo_override_keys if k in cfg}
    if ir_overrides:
        train_kwargs.update(ir_overrides)
        logger.info(f"  YOLO default overrides: {ir_overrides}")

    # ── Train ──
    results = model.train(**train_kwargs)

    # ── Post-training metadata ──
    meta = {
        "run_name": run_name,
        "run_grade": run_grade,
        "timestamp": datetime.now().isoformat(),
        "git_commit": capture_git_commit(),
        "dataset_version": cfg.get("dataset_version", "unknown"),
        "augmentation_id": aug_id,
        "config_path": cfg.get("_config_path", "unknown"),
        "device_profile": cfg.get("_device_profile_used", None),
        "environment": capture_environment(),
        "training_command": " ".join(sys.argv),
    }

    meta_path = out_dir / "train_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training complete. Run grade: {run_grade}")
    logger.info(f"Augmentation: {aug_id}")
    logger.info(f"Artifacts saved to: {out_dir}")
    logger.info(f"Next step: python scripts/eval.py --config {cfg.get('_config_path', '')} --split dev")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Config-driven YOLO training with IR augmentations"
    )
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config (must include augmentation_profiles)")
    parser.add_argument("--device-profile", default=None,
                        help="Path to device profile YAML (overrides config)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config, device_profile_cli=args.device_profile)
    cfg["_config_path"] = args.config

    if not cfg.get("run_name"):
        print("[ERROR] run_name must be set in the config file.")
        sys.exit(1)

    if not cfg.get("dataset_yaml"):
        print("[ERROR] dataset_yaml must be set in the config file.")
        sys.exit(1)

    train_with_augmentation(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
