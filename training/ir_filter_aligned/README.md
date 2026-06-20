# IR and grayscale confuser filters: aligned MLP

The thermal-IR and grayscale confuser filters are produced by **one script**, `mri/train_aligned.py`. It
trains a single MLP in the thermal feature space; the `--no-gray` flag decides whether the grayscale-
harvested confusers are included. There is no separate "thermal-only" script: thermal-only is just
`train_aligned.py --no-gray`.

- Weights (not published; contact the author):
  - `models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt` (thermal, operating point 0.05)
  - `models/verifiers/ir_aligned/mlp_aligned_gray_balanced.pt` (grayscale, operating point 0.25)
- Trainer: `mri/train_aligned.py` (lives under `mri/` because it uses the Model MRI feature extraction
  and alignment code)

## Train the thermal-only filter (mlp_aligned_thermalonly)

```
py mri/train_aligned.py --thermal-confusers --cbam --no-gray
```

`--no-gray` drops the grayscale-harvested confuser groups, so the network trains on thermal drones and
thermal confusers only. It writes `mlp_aligned.pt` under the output dir; that thermal-only checkpoint is
deployed as `mlp_aligned_thermalonly.pt`.

## Train the grayscale filter (mlp_aligned_gray_balanced)

```
py mri/train_aligned.py --thermal-confusers --cbam
```

Without `--no-gray`, the same run also harvests grayscale confusers and additionally writes
`mlp_aligned_gray.pt` (the same network with a grayscale scaler). The balanced grayscale variant is
deployed as `mlp_aligned_gray_balanced.pt`.

## Flags

| Flag | Effect |
|---|---|
| `--no-gray` | thermal-only (drop grayscale-harvested confusers) |
| `--thermal-confusers` | add the thermal-native IR_confusers train split |
| `--cbam` | add CBAM train drones and confusers (CBAM-valid stays held out for evaluation) |
| `--out DIR` | output dir (default `mri/results/ir_aligned[_nogray|_balanced|_cbam]`) |
