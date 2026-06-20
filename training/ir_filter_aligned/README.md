# IR and grayscale confuser filters: aligned MLP

The thermal-IR and grayscale confuser filters, one network trained in the thermal feature space with a
per-modality scaler. The thermal scaler runs on real thermal IR; the grayscale scaler runs when only a
visible camera is available and the frame is treated as single-channel.

- Weights:
  - `models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt` (thermal, operating point 0.05)
  - `models/verifiers/ir_aligned/mlp_aligned_gray_balanced.pt` (grayscale, operating point 0.25)
- Trainer: `mri/train_aligned.py`

The trainer lives under `mri/` because it is part of the Model MRI tool (it uses the MRI feature
extraction and alignment code). This folder is the pointer to it. Example:

```
py mri/train_aligned.py --thermal-confusers --cbam --no-gray   # thermal filter
py mri/train_aligned.py --gray-confusers --balanced            # grayscale filter
```
