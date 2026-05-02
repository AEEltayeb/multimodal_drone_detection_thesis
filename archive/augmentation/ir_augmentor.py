"""
IR-specific augmentation pipeline for drone detection.

Augmentations are applied in physically correct order:
  1. Polarity flip       — scene-level (camera polarity setting)
  2. Background shift    — scene-level (ambient temperature change)
  3. Gamma / tone curve  — scene-level (camera tone mapping pipeline)
  4. Contrast degrade    — scene-level (drone ≈ background temperature)
  5. Sensor noise        — sensor-level (detector noise + striping)
  6. CLAHE               — post-processing (local contrast enhancement)
  7. Gaussian defocus    — optics-level (lens out of focus)
  8. Motion blur         — optics-level (fast-moving target)
  9. Scale / copy-paste  — spatial-level (drone size variation)

Usage:
    from scripts.augmentation.ir_augmentor import IRAugmentor

    augmentor = IRAugmentor.from_config(config)
    img, labels = augmentor(img, labels)

Or integrated into Ultralytics via callback:
    from scripts.augmentation.ir_augmentor import register_ir_augmentations
    register_ir_augmentations(trainer, config)
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class IRAugConfig:
    """Configuration for IR augmentation pipeline."""

    # Pre-Aug 0a: Percentile normalization (sensor-domain harmonization)
    percentile_normalize: bool = True
    percentile_lo: float = 2.0   # lower percentile
    percentile_hi: float = 98.0  # upper percentile

    # Pre-Aug 0b: Local contrast normalization (high-pass filter)
    local_contrast_prob: float = 0.5
    local_contrast_sigma: float = 30.0  # Gaussian sigma for background estimation

    # Aug 1: Polarity flip
    polarity_flip_prob: float = 0.3

    # Aug 2: Background temperature shift
    background_shift_prob: float = 0.2
    background_shift_range: tuple = (-35, 35)

    # Aug 3: Gamma / tone curve variation
    gamma_prob: float = 0.20
    gamma_range: tuple = (0.7, 1.4)

    # Aug 4: Contrast degradation
    contrast_degrade_prob: float = 0.2
    contrast_degrade_severity: tuple = (0.15, 0.45)

    # Aug 5: Sensor noise
    sensor_noise_prob: float = 0.15
    sensor_noise_std: tuple = (5, 20)
    sensor_stripe_prob: float = 0.3  # prob of adding horizontal striping GIVEN noise is applied

    # Aug 6: Gaussian defocus
    gaussian_defocus_prob: float = 0.10
    gaussian_defocus_kernel: tuple = (3, 7)  # min, max (odd values only)

    # Aug 7: Motion blur
    motion_blur_prob: float = 0.15
    motion_blur_kernel: tuple = (3, 15)

    # Aug 6 (new): CLAHE — local contrast enhancement
    clahe_prob: float = 0.0        # off by default; enable in config
    clahe_clip_limit: tuple = (2.0, 4.0)   # random clip limit range
    clahe_tile_grid: int = 8       # tile grid size for CLAHE

    # Aug 9: Copy-paste with thermal blending
    # *** Disabled by default for IR — pasting violates thermal physics ***
    copy_paste_prob: float = 0.0
    copy_paste_scale_range: tuple = (0.5, 3.0)
    copy_paste_thermal_blend: float = 0.3

    # Curriculum: augmentation intensity ramp (0.0 = no aug, 1.0 = full aug)
    curriculum_enabled: bool = False
    curriculum_warmup_epochs: int = 30   # epochs with minimal aug
    curriculum_ramp_epochs: int = 50     # epochs to ramp from minimal to full

    @classmethod
    def from_dict(cls, d: dict) -> "IRAugConfig":
        """Create config from a dictionary (e.g. from YAML)."""
        kwargs = {}
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                # Convert lists to tuples for range params
                if k.endswith("_range") or k.endswith("_severity") or \
                   k.endswith("_std") or k.endswith("_kernel") or \
                   k.endswith("_scale_range") or k.endswith("_limit"):
                    v = tuple(v) if isinstance(v, list) else v
                kwargs[k] = v
        return cls(**kwargs)


class IRAugmentor:
    """
    IR-specific augmentation pipeline.

    Applies augmentations in physically correct order:
    sensor-normalization → scene-level → sensor-level → optics-level → spatial-level.
    """

    def __init__(self, config: Optional[IRAugConfig] = None):
        self.cfg = config or IRAugConfig()
        self._current_epoch = 0
        self._aug_scale = 1.0  # curriculum multiplier (0.0 = no aug, 1.0 = full)

    @classmethod
    def from_config(cls, config_dict: dict) -> "IRAugmentor":
        """Create from a config dictionary (e.g. augmentation_profiles.aug1)."""
        return cls(IRAugConfig.from_dict(config_dict))

    def set_epoch(self, epoch: int):
        """Update curriculum augmentation intensity based on training epoch."""
        self._current_epoch = epoch
        if not self.cfg.curriculum_enabled:
            self._aug_scale = 1.0
            return

        warmup = self.cfg.curriculum_warmup_epochs
        ramp = self.cfg.curriculum_ramp_epochs
        if epoch < warmup:
            self._aug_scale = 0.0  # minimal augmentation
        elif epoch < warmup + ramp:
            self._aug_scale = (epoch - warmup) / ramp  # linear ramp
        else:
            self._aug_scale = 1.0  # full augmentation

    def _should_apply(self, base_prob: float) -> bool:
        """Apply augmentation with probability scaled by curriculum."""
        return random.random() < base_prob * self._aug_scale

    def __call__(self, img: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Apply the full augmentation pipeline.

        Args:
            img: Input image (H, W) or (H, W, C), uint8
            labels: Optional YOLO labels array (N, 5) [class, cx, cy, w, h] normalized

        Returns:
            img: Augmented image
            labels: Labels (unchanged for pixel-level augs, updated for spatial augs)
        """
        # Pre-augmentation: sensor normalization (always-on, not affected by curriculum)
        img = self.percentile_normalize(img)
        img = self.local_contrast_norm(img)

        # Standard augmentation pipeline (curriculum-aware)
        img = self.polarity_flip(img)
        img = self.background_shift(img)
        img = self.gamma_tone_curve(img)
        img = self.contrast_degrade(img)
        img = self.sensor_noise(img)
        img = self.clahe_enhance(img)
        img = self.gaussian_defocus(img)
        img = self.motion_blur(img)

        # Copy-paste modifies both image AND labels
        if labels is not None and len(labels) > 0:
            img, labels = self.copy_paste_drone(img, labels)

        return img, labels

    # -- Pre-Aug 0a: Percentile Normalization --------------------------

    def percentile_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 255] using percentile clipping.

        Maps the 2nd-98th percentile range to [0, 255], removing sensor-specific
        dynamic range differences. Forces the model to learn shape/texture
        instead of brightness statistics.
        """
        if not self.cfg.percentile_normalize:
            return img

        # Work on grayscale channel
        if len(img.shape) == 3:
            gray = img[:, :, 0].astype(np.float32)
        else:
            gray = img.astype(np.float32)

        p_lo = np.percentile(gray, self.cfg.percentile_lo)
        p_hi = np.percentile(gray, self.cfg.percentile_hi)
        spread = p_hi - p_lo
        if spread < 1.0:
            return img  # already flat, don't divide by zero

        normalized = (gray - p_lo) / spread
        normalized = np.clip(normalized, 0.0, 1.0)
        result = (normalized * 255.0).astype(np.uint8)

        if len(img.shape) == 3:
            return np.stack([result] * img.shape[2], axis=-1)
        return result

    # -- Pre-Aug 0b: Local Contrast Normalization ----------------------

    def local_contrast_norm(self, img: np.ndarray) -> np.ndarray:
        """
        High-pass filter: subtract Gaussian-blurred background.

        Removes large-scale temperature gradients (sky/ground/buildings)
        and amplifies small thermal anomalies (drones, birds).
        Applied as a 50% augmentation — the model learns with and without.

        Sigma is computed relative to image height (h/16) so the background
        removal scale is consistent across 320x256 and 640x512 images.
        """
        if random.random() >= self.cfg.local_contrast_prob:
            return img

        if len(img.shape) == 3:
            gray = img[:, :, 0].astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # Adaptive sigma: proportional to image height
        h = gray.shape[0]
        sigma = max(5.0, h / 16.0)  # 256->16, 512->32, 640->40

        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        highpass = gray - blurred + 128.0  # re-center around 128
        highpass = np.clip(highpass, 0, 255).astype(np.uint8)

        if len(img.shape) == 3:
            return np.stack([highpass] * img.shape[2], axis=-1)
        return highpass

    # ── Aug 1: Polarity Flip ──────────────────────────────────────

    def polarity_flip(self, img: np.ndarray) -> np.ndarray:
        """
        Invert image intensities to simulate white-hot ↔ black-hot switch.

        In IR imaging, cameras can display hot objects as bright (white-hot)
        or dark (black-hot). This is a camera setting, not a physical change,
        so it should be the first augmentation.
        """
        if self._should_apply(self.cfg.polarity_flip_prob):
            return 255 - img
        return img

    # ── Aug 2: Background Temperature Shift ───────────────────────

    def background_shift(self, img: np.ndarray) -> np.ndarray:
        """
        Shift entire image intensity to simulate different ambient temperatures.

        Morning cold sky = dark background, afternoon hot ground = bright.
        The model should detect drones by local contrast, not absolute brightness.
        """
        if self._should_apply(self.cfg.background_shift_prob):
            lo, hi = self.cfg.background_shift_range
            shift = random.randint(lo, hi)
            img = img.astype(np.int16) + shift
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ── Aug 3: Gamma / Tone Curve Variation ───────────────────────

    def gamma_tone_curve(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random gamma correction to simulate different IR camera
        tone mapping pipelines.

        Different IR cameras apply nonlinear tone curves to map scene
        radiance to display values. Two sensors can render the same
        scene very differently. Gamma variation is the single most
        effective augmentation for cross-sensor IR robustness.

        gamma < 1.0: brightens dark areas (expands shadows)
        gamma > 1.0: darkens midtones (compresses shadows)
        """
        if self._should_apply(self.cfg.gamma_prob):
            lo, hi = self.cfg.gamma_range
            gamma = random.uniform(lo, hi)
            # Normalize to [0,1], apply gamma, scale back
            img = np.power(img.astype(np.float32) / 255.0, gamma) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ── Aug 4: Contrast Degradation ───────────────────────────────

    def contrast_degrade(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce contrast by blending image toward its mean intensity.

        Simulates the hardest IR detection scenario: drone temperature ≈
        background temperature. Forces the model to use shape/texture cues
        instead of relying on high thermal contrast.
        """
        if self._should_apply(self.cfg.contrast_degrade_prob):
            lo, hi = self.cfg.contrast_degrade_severity
            severity = random.uniform(lo, hi)
            mean_val = img.mean()
            img = img.astype(np.float32) * (1 - severity) + mean_val * severity
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ── Aug 5: Sensor Noise ───────────────────────────────────────

    def sensor_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add realistic IR sensor noise patterns.

        Includes:
        - Gaussian read noise (all sensors)
        - Horizontal striping artifacts (common in uncooled LWIR microbolometers)

        Applied after scene-level transforms because noise is added by the
        sensor during image capture, not present in the scene.
        """
        if self._should_apply(self.cfg.sensor_noise_prob):
            lo, hi = self.cfg.sensor_noise_std
            noise_std = random.uniform(lo, hi)

            noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)

            # Add horizontal striping (row-correlated noise)
            if random.random() < self.cfg.sensor_stripe_prob:
                h = img.shape[0]
                stripe_std = noise_std * 0.5
                stripe = np.random.normal(0, stripe_std, (h, 1)).astype(np.float32)
                if len(img.shape) == 3:
                    stripe = stripe[:, :, np.newaxis]
                noise = noise + stripe

            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ── Aug 6: CLAHE (Contrast Limited Adaptive Histogram Equalization) ──

    def clahe_enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to amplify local contrast in low-contrast IR frames.

        CLAHE divides the image into tiles and equalizes histograms locally,
        boosting faint thermal signatures (drones with similar temperature
        to background) without over-amplifying already-bright regions.

        Applied after sensor noise because CLAHE is a post-processing step.
        Applied before defocus/motion blur to preserve the enhanced signal.

        Uses a random clip limit each time (2.0-4.0) for variety.
        """
        if not self._should_apply(self.cfg.clahe_prob):
            return img

        # Get grayscale channel
        if len(img.shape) == 3:
            gray = img[:, :, 0]
        else:
            gray = img

        # Random clip limit for variety
        lo, hi = self.cfg.clahe_clip_limit
        clip_limit = random.uniform(lo, hi)
        tile = self.cfg.clahe_tile_grid

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile, tile)
        )
        enhanced = clahe.apply(gray)

        if len(img.shape) == 3:
            return np.stack([enhanced] * img.shape[2], axis=-1)
        return enhanced

    # ── Aug 7: Gaussian Defocus ───────────────────────────────────

    def gaussian_defocus(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to simulate an out-of-focus IR camera.

        Different failure mode from motion blur — this is a uniform optical
        blur across the entire image. IR cameras often lose focus in field
        conditions due to temperature changes affecting lens geometry.
        """
        if self._should_apply(self.cfg.gaussian_defocus_prob):
            lo, hi = self.cfg.gaussian_defocus_kernel
            # Kernel must be odd
            k = random.randrange(lo, hi + 1, 2)
            if k % 2 == 0:
                k += 1
            sigma = k / 3.0  # reasonable sigma for kernel size
            img = cv2.GaussianBlur(img, (k, k), sigma)
        return img

    # ── Aug 7: Motion Blur ────────────────────────────────────────

    def motion_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply directional motion blur to simulate fast-moving targets.

        Creates the characteristic "streak" appearance of drones moving
        quickly through the IR field of view. Applied after defocus because
        motion blur occurs during exposure, after optical focusing.

        Kernel range capped at 15px to avoid unrealistic streaks that are
        longer than typical drone sizes in IR imagery.
        """
        if self._should_apply(self.cfg.motion_blur_prob):
            lo, hi = self.cfg.motion_blur_kernel
            kernel_size = random.randint(lo, hi)
            angle = random.randint(0, 180)

            # Build directional kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center = kernel_size // 2
            radian = angle * np.pi / 180
            dx, dy = np.cos(radian), np.sin(radian)

            for i in range(-center, center + 1):
                x = int(round(center + i * dx))
                y = int(round(center + i * dy))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0

            total = kernel.sum()
            if total > 0:
                kernel /= total
                img = cv2.filter2D(img, -1, kernel)
        return img

    # ── Aug 8: Copy-Paste with Thermal Blending ──────────────────

    def copy_paste_drone(self, img: np.ndarray,
                         labels: np.ndarray) -> tuple:
        """
        Copy an existing drone bbox and paste at a different size/location.

        Uses relative scaling (0.5x–3.0x of original size) instead of fixed
        target sizes to preserve drone thermal structure. Respects IR physics
        by blending the drone crop toward the local background temperature.
        """
        if random.random() >= self.cfg.copy_paste_prob:
            return img, labels

        if labels is None or len(labels) == 0:
            return img, labels

        h, w = img.shape[:2]

        # Pick a random existing drone bbox to copy
        idx = random.randint(0, len(labels) - 1)
        cls, cx, cy, bw, bh = labels[idx]

        # Convert normalized coords to pixels
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 5 or crop_h < 5:
            # Tiny crops (< 5px) have no meaningful structure to copy
            return img, labels

        # Extract drone crop
        drone_crop = img[y1:y2, x1:x2].copy()

        # Relative scaling: 0.5x–3.0x of original size (preserves structure)
        lo_scale, hi_scale = self.cfg.copy_paste_scale_range
        scale = random.uniform(lo_scale, hi_scale)
        target_w = max(5, int(crop_w * scale))
        target_h = max(5, int(crop_h * scale))

        # Clip dimensions instead of failing to preserve augmentation
        target_w = min(target_w, w // 2 - 1)
        target_h = min(target_h, h // 2 - 1)
        
        # Ensure they didn't become too small after clipping
        target_w = max(5, target_w)
        target_h = max(5, target_h)

        # Resize crop (preserves aspect ratio via independent W/H scaling)
        drone_resized = cv2.resize(drone_crop, (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)

        # Random paste location (avoid edges)
        margin_x = max(5, w // 20)
        margin_y = max(5, h // 20)
        if w - 2 * margin_x <= target_w or h - 2 * margin_y <= target_h:
            return img, labels
            
        # Compute source background stats from crop border (not center drone)
        # Use a 2px-wide border ring to estimate source local background
        border_mask = np.ones((crop_h, crop_w), dtype=bool)
        b = max(2, min(crop_h, crop_w) // 4)  # border width
        if crop_h > 2 * b and crop_w > 2 * b:
            border_mask[b:-b, b:-b] = False
        src_border = drone_crop[border_mask].astype(np.float32)
        src_mean = src_border.mean()
        src_std = max(src_border.std(), 1.0)

        # Try up to 10 random locations (increased for compatibility filtering)
        max_attempts = 10
        for _ in range(max_attempts):
            paste_x = random.randint(margin_x, w - margin_x - target_w)
            paste_y = random.randint(margin_y, h - margin_y - target_h)

            # Compute proposed bbox in normalized coords
            new_cx = (paste_x + target_w / 2) / w
            new_cy = (paste_y + target_h / 2) / h
            new_bw = target_w / w
            new_bh = target_h / h

            # Check IoU with all existing labels — reject if overlapping
            if self._has_overlap(labels, new_cx, new_cy, new_bw, new_bh,
                                 max_iou=0.3):
                continue

            # Thermal context compatibility check:
            # Reject if destination patch has very different thermal profile
            dst_patch = img[paste_y:paste_y + target_h,
                            paste_x:paste_x + target_w].astype(np.float32)
            dst_mean = dst_patch.mean()
            dst_std = max(dst_patch.std(), 1.0)

            if abs(src_mean - dst_mean) > 25:
                continue  # sky↔building mismatch
            if abs(src_std - dst_std) > 20:
                continue  # textured↔smooth mismatch

            break  # passed both checks
        else:
            # All attempts failed — skip paste
            return img, labels

        # ── Advanced copy-paste blending ──────────────────────────────
        # Addresses: drone isolation, gradient-aware blending, adaptive
        # alpha masking, and feather capping for tiny targets.

        bg_patch = img[paste_y:paste_y + target_h,
                       paste_x:paste_x + target_w].copy()
        blend = self.cfg.copy_paste_thermal_blend
        drone_f = drone_resized.astype(np.float32)
        bg_f = bg_patch.astype(np.float32)

        # --- Fix 1: Intensity-based drone isolation ---
        # Instead of pasting the entire crop (drone + source background),
        # compute a saliency mask that weights drone pixels higher than
        # background pixels within the crop.
        crop_mean = drone_f.mean()
        crop_std = max(drone_f.std(), 1.0)
        # Saliency = how far each pixel deviates from the crop mean
        # (drones are typically brighter or darker than surrounding bg)
        saliency = np.abs(drone_f - crop_mean) / (crop_std + 1e-6)
        saliency = np.clip(saliency * 1.5, 0, 1)
        # Reduce to 2D if image is multi-channel
        if saliency.ndim == 3:
            saliency = saliency.mean(axis=2)

        # --- Fix 2: Elliptical base mask (geometric prior) ---
        ellipse_mask = np.zeros((target_h, target_w), dtype=np.float32)
        cv2.ellipse(ellipse_mask,
                     center=(target_w // 2, target_h // 2),
                     axes=(target_w // 2, target_h // 2),
                     angle=0, startAngle=0, endAngle=360,
                     color=1.0, thickness=-1)

        # --- Combine: saliency-weighted ellipse mask ---
        # High saliency pixels within ellipse get full alpha,
        # low saliency (background) pixels get attenuated
        alpha = ellipse_mask * (0.1 + 0.9 * saliency)
        # --- Fix 4: Feather capping for tiny targets ---
        # Small drones get less feathering to avoid washing out
        min_dim = min(target_w, target_h)
        if min_dim <= 8:
            feather_k = 3  # minimal feather for tiny drones
        elif min_dim <= 16:
            feather_k = max(3, (min_dim // 4) | 1)
        else:
            feather_k = max(3, min(9, (min_dim // 4) | 1))
        alpha = cv2.GaussianBlur(alpha, (feather_k, feather_k), 0)

        # --- Fix 3: Gradient-aware thermal blending ---
        # Instead of blending toward scalar bg_mean, blend toward the
        # actual destination background at each pixel position.
        # This preserves local background gradients.
        drone_adapted = drone_f * (1 - blend) + bg_f * blend

        # --- Final alpha composite ---
        if len(bg_f.shape) == 3 and len(alpha.shape) == 2:
            alpha = alpha[:, :, np.newaxis]
        if len(drone_adapted.shape) == 3 and drone_adapted.shape[2] == 1 \
                and len(bg_f.shape) == 2:
            drone_adapted = drone_adapted[:, :, 0]

        result_patch = drone_adapted * alpha + bg_f * (1.0 - alpha)
        img[paste_y:paste_y + target_h, paste_x:paste_x + target_w] = \
            np.clip(result_patch, 0, 255).astype(np.uint8)

        # Add new label
        new_label = np.array([[cls, new_cx, new_cy, new_bw, new_bh]], dtype=labels.dtype)
        labels = np.vstack([labels, new_label])

        return img, labels

    @staticmethod
    def _has_overlap(labels: np.ndarray, cx: float, cy: float,
                     bw: float, bh: float, max_iou: float = 0.3) -> bool:
        """
        Check if a proposed bbox overlaps any existing label above max_iou.

        All coords are normalized [0, 1].
        """
        # Proposed box edges
        ax1 = cx - bw / 2
        ay1 = cy - bh / 2
        ax2 = cx + bw / 2
        ay2 = cy + bh / 2
        area_a = bw * bh

        for label in labels:
            _, ecx, ecy, ebw, ebh = label
            bx1 = ecx - ebw / 2
            by1 = ecy - ebh / 2
            bx2 = ecx + ebw / 2
            by2 = ecy + ebh / 2
            area_b = ebw * ebh

            # Intersection
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            if ix1 >= ix2 or iy1 >= iy2:
                continue  # No overlap

            inter = (ix2 - ix1) * (iy2 - iy1)
            union = area_a + area_b - inter
            iou = inter / max(union, 1e-6)

            if iou >= max_iou:
                return True

        return False


# ── Ultralytics Integration ───────────────────────────────────────

def register_ir_augmentations(model, config: dict):
    """
    Register IR augmentations in the Ultralytics data loading pipeline.

    Augmentations run on CPU inside each data loader worker's __getitem__,
    in parallel with GPU training — zero GPU overhead.

    The IRAugmentor.__call__() applies all 8 augmentations in physical order
    on numpy uint8 images. By injecting after Ultralytics' standard augmentations
    (mosaic, hsv, flip, etc.), we simulate IR sensor domain effects.

    Only the TRAINING dataset is augmented (uses an instance flag).
    """
    augmentor = IRAugmentor.from_config(config)

    def on_train_start(trainer_instance):
        import torch

        train_dataset = trainer_instance.train_loader.dataset
        # Flag only the training dataset — validation will NOT be augmented
        train_dataset._ir_augmentor = augmentor

        OriginalClass = train_dataset.__class__
        original_getitem = OriginalClass.__getitem__

        def _augmented_getitem(self, index):
            item = original_getitem(self, index)

            aug = getattr(self, '_ir_augmentor', None)
            if aug is None:
                return item

            img_tensor = item.get("img")
            if img_tensor is None:
                return item

            # torch (C, H, W) uint8 → numpy (H, W, C) uint8
            img_np = img_tensor.numpy().transpose(1, 2, 0).copy()

            # Build labels: [cls, cx, cy, w, h] per row
            cls_t = item.get("cls")
            bbox_t = item.get("bboxes")

            if cls_t is not None and cls_t.numel() > 0:
                cls_np = cls_t.numpy()
                bbox_np = bbox_t.numpy()
                if cls_np.ndim == 1:
                    cls_np = cls_np[:, np.newaxis]
                labels = np.hstack([cls_np, bbox_np]).astype(np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

            # Apply all 8 IR augmentations (pure numpy, runs on CPU worker)
            img_np, labels = aug(img_np, labels)

            # Ensure 3D (H, W, C)
            if img_np.ndim == 2:
                img_np = img_np[:, :, np.newaxis]

            # numpy (H, W, C) uint8 → torch (C, H, W) uint8
            item["img"] = torch.from_numpy(
                np.ascontiguousarray(img_np.transpose(2, 0, 1))
            )

            if labels is not None and len(labels) > 0:
                item["cls"] = torch.from_numpy(labels[:, 0:1].copy()).float()
                item["bboxes"] = torch.from_numpy(labels[:, 1:5].copy()).float()
                item["batch_idx"] = torch.zeros(len(labels), dtype=torch.float32)
            else:
                item["cls"] = torch.zeros((0, 1), dtype=torch.float32)
                item["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
                item["batch_idx"] = torch.zeros(0, dtype=torch.float32)

            return item

        OriginalClass.__getitem__ = _augmented_getitem

    model.add_callback("on_train_start", on_train_start)

    # Curriculum: update augmentation intensity each epoch
    def on_train_epoch_start(trainer_instance):
        epoch = trainer_instance.epoch
        augmentor.set_epoch(epoch)
        if augmentor.cfg.curriculum_enabled and epoch % 20 == 0:
            print(f"  [IR-Aug] Epoch {epoch}: aug_scale={augmentor._aug_scale:.2f}")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)


# ── Standalone Testing ────────────────────────────────────────────

if __name__ == "__main__":
    """Quick visual test: apply all augmentations to a sample image."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ir_augmentor.py <image_path> [output_path]")
        print("  Applies all IR augmentations and saves side-by-side comparison.")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ir_aug_demo.jpg"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read: {img_path}")
        sys.exit(1)

    print(f"Input: {img_path} ({img.shape})")

    # Create augmentor with all augmentations at 100% probability for demo
    demo_cfg = IRAugConfig(
        polarity_flip_prob=1.0,
        background_shift_prob=1.0,
        contrast_degrade_prob=1.0,
        sensor_noise_prob=1.0,
        gaussian_defocus_prob=1.0,
        motion_blur_prob=1.0,
        copy_paste_prob=0.0,  # skip — need labels
    )
    augmentor = IRAugmentor(demo_cfg)

    # Apply each augmentation individually for comparison
    results = [("Original", img.copy())]
    names = [
        ("1_Polarity", augmentor.polarity_flip),
        ("2_BgShift", augmentor.background_shift),
        ("3_Gamma", augmentor.gamma_tone_curve),
        ("4_Contrast", augmentor.contrast_degrade),
        ("5_Noise", augmentor.sensor_noise),
        ("6_Defocus", augmentor.gaussian_defocus),
        ("7_MotionBlur", augmentor.motion_blur),
    ]

    # Force each one to apply
    old_probs = {}
    for attr in dir(demo_cfg):
        if attr.endswith("_prob"):
            old_probs[attr] = getattr(demo_cfg, attr)
            setattr(demo_cfg, attr, 1.0)

    for name, fn in names:
        result = fn(img.copy())
        results.append((name, result))

    # Restore
    for attr, val in old_probs.items():
        setattr(demo_cfg, attr, val)

    # Stack horizontally
    max_h = max(r[1].shape[0] for r in results)
    padded = []
    for name, r in results:
        if len(r.shape) == 2:
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        pad_h = max_h - r.shape[0]
        if pad_h > 0:
            r = np.vstack([r, np.zeros((pad_h, r.shape[1], 3), dtype=np.uint8)])
        # Add label
        cv2.putText(r, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        padded.append(r)

    montage = np.hstack(padded)
    cv2.imwrite(out_path, montage)
    print(f"Saved: {out_path} ({montage.shape})")
