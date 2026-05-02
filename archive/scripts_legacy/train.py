"""
train.py — Config-driven YOLO training wrapper.

Usage:
    python scripts/train.py --config configs/rgb_baseline.yaml
    python scripts/train.py --config configs/rgb_baseline.yaml --device-profile configs/devices/local_1050ti.yaml
    python scripts/train.py --config configs/rgb_baseline.yaml --resume runs/<run>/weights/last.pt

Config layering order:
    1. configs/base.yaml           (shared defaults)
    2. experiment config           (e.g. rgb_baseline.yaml)
    3. device_profile from config  (if set in experiment config)
    4. --device-profile CLI flag   (overrides everything)

Produces training artifacts (weights, logs, metadata).
Evaluation artifacts are produced by scripts/eval.py.
"""

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, device_profile_cli: str = None) -> dict:
    """Load base.yaml → experiment config → device profile (layered)."""
    config_dir = Path(config_path).parent
    base_path = config_dir / "base.yaml"

    # Layer 1: base defaults
    cfg = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Layer 2: experiment config
    with open(config_path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in override.items() if v is not None})

    # Layer 3: device profile (from config field)
    device_profile = device_profile_cli or cfg.get("device_profile")
    if device_profile:
        dp_path = Path(device_profile)
        if not dp_path.exists():
            # Try relative to project root
            dp_path = Path(config_path).parent.parent / device_profile
        if dp_path.exists():
            with open(dp_path, "r", encoding="utf-8") as f:
                dp = yaml.safe_load(f) or {}
            cfg.update({k: v for k, v in dp.items() if v is not None})
            cfg["_device_profile_used"] = str(dp_path)
        else:
            print(f"[WARN] Device profile not found: {device_profile}")

    return cfg


def resolve_output_dir(cfg: dict) -> Path:
    """Resolve the output directory, substituting ${run_name}."""
    template = cfg.get("output_dir", "runs/${run_name}")
    run_name = cfg["run_name"]
    return Path(template.replace("${run_name}", run_name))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path) -> logging.Logger:
    """Configure logger to write to both console and train.log."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if train() is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = out_dir / "train.log"
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Environment capture (reproducibility checklist §1.4)
# ---------------------------------------------------------------------------

def capture_environment() -> dict:
    """Capture software versions and hardware info."""
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch
        env["pytorch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["pytorch"] = "not installed"

    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except (ImportError, AttributeError):
        env["ultralytics"] = "unknown"

    return env


def capture_git_commit() -> str:
    """Try to get the current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_from: str = None):
    """Run YOLO training based on config."""
    from ultralytics import YOLO

    run_name = cfg["run_name"]
    run_grade = cfg.get("run_grade", "PUB")
    out_dir = resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)

    # ── Log run grade prominently ──
    if run_grade == "EXP":
        logger.warning("=" * 60)
        logger.warning("  RUN GRADE: EXP (pilot/exploratory)")
        logger.warning("  This run CANNOT be used for publication results.")
        if cfg.get("_device_profile_used"):
            logger.warning(f"  Device profile: {cfg['_device_profile_used']}")
        logger.warning("=" * 60)
    else:
        logger.info(f"  RUN GRADE: PUB (publication-quality)")

    # ── Archive fully resolved config ──
    # Remove internal keys before archiving
    config_to_save = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_archive = out_dir / "config.yaml"
    with open(config_archive, "w", encoding="utf-8") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Archived resolved config to: {config_archive}")

    # ── Log the full resolved config ──
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

    # ── Augmentation overrides (passthrough to YOLO) ──
    aug_keys = [
        "mosaic", "erasing", "translate", "scale", "fliplr", "flipud",
        "hsv_h", "hsv_s", "hsv_v", "degrees", "shear", "perspective",
        "bgr", "mixup", "cutmix", "copy_paste", "auto_augment",
    ]
    for key in aug_keys:
        if cfg.get(key) is not None:
            train_kwargs[key] = cfg[key]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  TRAINING: {run_name}")
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
    logger.info("")

    # ── IR polarity flip callback ──
    polarity_flip_prob = cfg.get("polarity_flip", 0.0)
    if polarity_flip_prob > 0:
        import random as _rng

        def patch_preprocess(trainer):
            """Monkey-patch preprocess_batch to add polarity flipping."""
            _original = trainer.preprocess_batch

            def preprocess_with_flip(batch):
                batch = _original(batch)
                imgs = batch["img"]  # (B, C, H, W) float32, 0-1 range
                for i in range(imgs.shape[0]):
                    if _rng.random() < polarity_flip_prob:
                        imgs[i] = 1.0 - imgs[i]
                return batch

            trainer.preprocess_batch = preprocess_with_flip

        model.add_callback("on_train_start", patch_preprocess)
        logger.info(f"  IR polarity flip enabled: p={polarity_flip_prob}")

    # ── Train ──
    results = model.train(**train_kwargs)

    # ── Post-training: save reproducibility metadata ──
    meta = {
        "run_name": run_name,
        "run_grade": run_grade,
        "timestamp": datetime.now().isoformat(),
        "git_commit": capture_git_commit(),
        "dataset_version": cfg.get("dataset_version", "unknown"),
        "config_path": cfg.get("_config_path", "unknown"),
        "device_profile": cfg.get("_device_profile_used", None),
        "environment": capture_environment(),
        "training_command": " ".join(sys.argv),
    }

    meta_path = out_dir / "train_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training complete. Run grade: {run_grade}")
    logger.info(f"Artifacts saved to: {out_dir}")
    logger.info(f"Next step: python scripts/eval.py --config {cfg.get('_config_path', 'configs/rgb_baseline.yaml')} --split dev")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Config-driven YOLO training")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--device-profile", default=None, help="Path to device profile YAML (overrides config)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config, device_profile_cli=args.device_profile)
    cfg["_config_path"] = args.config

    if not cfg.get("run_name"):
        print("[ERROR] run_name must be set in the config file.")
        sys.exit(1)

    if not cfg.get("dataset_yaml"):
        print("[ERROR] dataset_yaml must be set in the config file.")
        sys.exit(1)

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
