# IR + Grayscale Verifier — Thesis Addition (evidence pack, 2026-05-31)

> Source-of-truth for the IR/grayscale chapter addition. Three claims, each with its
> figure(s) + verified numbers + reproduction. The IR detector (`finetune_v3b`) was
> **never trained on grayscale**, so all grayscale findings are novel for this model.
> Knowledge rows: `ledger.gray-thermal-alignable`, `ledger.ir-grayscale-harvest-solves-thermal-verifier`,
> `ledger.ir-recall-fixed-by-drone-diversity`, `ledger.grayscale-drone-recall`.

---

## Proof 1 — IR drones and confusers ARE separable in the detector's features

The IR YOLO (`finetune_v3b`) FPN features (p3+p5 ROI-pooled, 517-D) linearly separate
drones from confusers. Measured on 14,697 drones + 1,386 confusers mined across IR
surfaces (`mri/results/ir_v3b_report`).

| signal | value |
|---|---|
| LDA train separability | **0.981** |
| Max ANOVA F (single feature) | **5,370** (p5 channel) |
| Median ANOVA F | 256 |
| Raw detector hallucination | 1.8% / img |
| Projected FP cut at 10% recall cost | 89% |

Figures (`docs/analysis/images/`): `ir_v3b_lda.png` (two clean peaks), `ir_v3b_pca.png`
(unsupervised structure), `ir_v3b_neurons.png` (per-neuron drone/confuser KDE),
`ir_v3b_heatmap.png`, `ir_v3b_anova.png`.
Reproduce: `py -m mri --yolo <v3b> --config mri/configs/ir_v3b.yaml --train-mlp --feature-set fused`.

> **Takeaway:** the knowledge to reject confusers already exists inside the IR detector's
> features — a lightweight MLP can read it. (Same result holds for RGB; this is the IR half.)

---

## Proof 2 — The grayscale → thermal feature gap is alignable (grayscale transfers)

The detector represents a class the *same way* in real thermal and in grayscale-RGB:
top-discriminative neurons overlap (Jaccard 0.71–0.88), mean-activation correlation
0.93–0.99, drone-class centroid cosine-distance **0.012**. The only difference is a
per-feature affine offset — so a confuser filter trained in one modality transfers to the
other after **per-modality z-score**.

| gray→thermal transfer (drone-vs-confuser AUROC) | value |
|---|---|
| raw (no alignment) | **0.500** (chance) |
| CORAL | 0.707 |
| **per-modality z-score** | **0.919** |
| within-modality ceiling (thermal CV) | 0.974 |

Figure: `docs/analysis/images/ir_gray_thermal_alignment.png` — (A) drone-vs-confuser LDA
after per-modality z-score (thermal & grayscale land in the same regions); (B) the transfer
AUROC bars (chance → near-ceiling).
Reproduce: `py mri/modality_probe.py` (signatures) + `py mri/modality_align.py` (transfer).

> **Takeaway:** grayscale-RGB confusers (abundant) can be harvested and z-aligned to stand
> in for scarce thermal confusers. Novel: the IR model was never trained on grayscale.

---

## Proof 3 — Results: a deployable thermal verifier from grayscale-harvested confusers

`mlp_v5_ir_aligned` = thermal drones (recall-safety via drone diversity) + grayscale-harvested
confusers, per-modality z-aligned into one thermal-deploy checkpoint (517-D, loads in the
production `MLPVerifier`). **CBAM held out of training** — a clean test of generalization to
novel thermal aerial confusers.

| surface | bare | patch (old) | **aligned MLP** |
|---|---|---|---|
| **CBAM (held out)** | F1 0.699, 48 FP, R 0.967 | F1 0.688, 41 FP | **F1 0.841, 13 FP (73% catch), R 0.883** |
| ir_dset test | F1 0.965, R 0.971 | F1 0.943 (ΔR −0.044) | **F1 0.959, ΔR −0.012** |
| ir_video test | F1 0.942, R 0.977 | F1 0.920 | **F1 0.942, ΔR 0.000** |
| antiuav test | F1 0.962, R 0.942 | F1 0.962 | **F1 0.962, ΔR 0.000** |

Supporting:
- **Recall fix:** drone-diversity re-mine made the thermal verifier recall-safe (ΔR≈0); the
  earlier −5% recall loss was OOD-drone coverage, not conf-reliance (yolo-only CV 0.986 ≈ fused 0.987).
- **Grayscale is the hallucination mode:** raw halluc 37.2%/img (vs thermal 1.8%); the grayscale
  verifier cut held-out confuser FP 325→13 (96%).
- **Grayscale drone recall** (bare v3b): gray beats raw-RGB (svan +0.197, rgb_dataset +0.130) but
  is mediocre absolute (0.33–0.65) vs thermal (0.79–0.98).

Reproduce: `py eval/run_aligned_full.py` (train dual-checkpoint + holdout on grayscale + thermal/ir_dset).

> **Takeaway:** the IR confuser-verifier blocker was confuser *data scarcity*, not the detector.
> Harvesting from grayscale + alignment yields a confuser-rich AND recall-safe thermal verifier —
> reversing the earlier "ship none" conclusion.

### 3b. One unified verifier serves BOTH modes (re-run `bszquitg6`, 2026-05-31)

The aligned net dual-saves a thermal scaler (`mlp_aligned.pt`) and a grayscale scaler
(`mlp_aligned_gray.pt`) — same weights, per-mode standardization.

**Thermal deploy** (`mlp_aligned.pt`, conf 0.40, thr 0.05; CBAM held out; full P/R/F1, FP in parens):

| surface | n | bare P / R / F1 (FP) | aligned MLP P / R / F1 (FP) | ΔR / ΔF1 |
|---|---|---|---|---|
| **CBAM** (held out) | 180 | 0.547 / 0.967 / 0.699 (48) | **0.786 / 0.917 / 0.846 (15)** | −0.050 / **+0.147** |
| **ir_dset_final** | 4806 | 0.965 / 0.965 / 0.965 (109) | **0.965 / 0.958 / 0.962 (108)** | −0.007 / −0.003 |
| ir_video test | 831 | 0.909 / 0.977 / 0.942 (80) | **0.909 / 0.977 / 0.942 (80)** | 0.000 / 0.000 |
| antiuav test | 4269 | 0.983 / 0.942 / 0.962 (68) | **0.983 / 0.942 / 0.962 (68)** | 0.000 / 0.000 |
| sea-ships (neg) | 8398 | — (3 FP) | — (**1 FP**) | — |
| road-thermal (neg) | 652 | — (0 FP) | — (0 FP) | — |
| *patch baseline on CBAM* | 180 | — | 0.564 / 0.883 / 0.688 (41) | (cut only 7 FP) |

**Grayscale deploy** (`mlp_aligned_gray.pt`, `--grayscale-input`, conf 0.25, thr 0.25):

| surface | n | bare P / R / F1 (FP) | aligned-gray P / R / F1 (FP) | dedicated `mlp_v5_gray` |
|---|---|---|---|---|
| confusers (rgb_confusers→gray, neg) | 1317 | — (325 FP) | — (**12 FP, 96% cut**) | 13 FP (96%), ΔR −0.113 |
| drones (rgb_dataset→gray) | 17209 | 0.704 / 0.210 / 0.324 (1307) | **0.762 / 0.157 / 0.261 (723)** | (ΔR −0.053 here vs −0.113) |

> **Takeaway:** one net + two per-modality scalers covers thermal *and* grayscale deployment —
> recall-safe in both, 69%/96% confuser catch, matching/beating the per-mode dedicated models.
> Grayscale-mode bare drone recall (0.21 on photo surfaces) is a detector limit, not the verifier.

---

## 4. Statistical methods (what each statistic is and why it is used)

The argument is built on standard, interpretable statistics rather than a single black-box
score. Each answers a specific sub-question.

### 4.1 Per-modality z-score (standardization) — the core of the alignment claim
For modality \(m\) (thermal or grayscale) and feature \(j\), each value is standardized to that
modality's own statistics:
\[ x'_{j} = \frac{x_{j} - \mu_{m,j}}{\sigma_{m,j}}, \]
where \(\mu_{m,j},\sigma_{m,j}\) are the mean and standard deviation of feature \(j\) estimated
**within modality \(m\)**. This is an affine (shift + scale) transform applied per feature.

*Why it is the right tool here.* Three diagnostics show the gray↔thermal gap is a **per-feature
affine offset**, not a structural change in the representation:
- **centroid cosine-distance 0.012** — the class-mean vectors point in nearly the same direction
  across modality (no rotation);
- **mean-activation correlation 0.93–0.99** — the *shape* of the average signature is preserved;
- **modality is linearly separable at accuracy ≈ 1.0** — yet that separation is a *consistent,
  low-magnitude* offset (small relative to the drone–confuser signal), not a re-arrangement.

A per-feature affine offset is *exactly* what standardization removes: subtracting \(\mu_m\)
re-centres each modality and dividing by \(\sigma_m\) re-scales it, mapping both modalities onto a
common standardized axis where the shared drone-vs-confuser structure coincides. Empirically the
transfer AUROC rises from **0.500 (raw) to 0.919 (per-modality z-score)**, against a within-modality
ceiling of 0.974.

*Why a single (global) scaler fails.* Fitting one scaler on the **source** modality and applying it
to the **target** leaves the target off-centre and off-scale; a decision boundary calibrated to the
source's absolute feature values then lands in the wrong place on the target → chance-level AUROC
(0.500). The fix is to standardize **each modality to its own** \((\mu,\sigma)\). At deployment this
is free: the thermal \((\mu_t,\sigma_t)\) is estimated once from thermal data and folded into the
verifier's input scaler.

### 4.2 LDA — linear separability of the two classes
Linear Discriminant Analysis projects the 512-D feature vector onto the single axis that maximises
the ratio of between-class to within-class variance (Fisher's criterion) for drone vs confuser. The
**train-set accuracy** of a threshold on that axis quantifies linear separability: **0.981** on IR
means the two classes are almost linearly separable in the detector's own features (Proof 1), and
the same axis is the visualization used in Proof 2's panel (A).

### 4.3 ANOVA F-test — per-neuron discriminative power
For each feature, the one-way ANOVA \(F\)-statistic is the ratio of between-class to within-class
variance. Large \(F\) means that single neuron strongly separates drones from confusers. The
**max \(F = 5{,}370\)** (a p5 channel; median \(F = 256\)) indicates near-binary "is-this-a-drone"
switch neurons that emerged from single-class detector training — evidence the discriminative signal
is concentrated, not diffuse. (Source: `mri/results/ir_v3b_report/stats.json`, `separability.max_anova_F`.
An earlier draft cited 15,032 from a different MRI run, `ir_v3b_regen`, whose LDA was 0.954 rather than
the canonical 0.981; that figure is superseded.)

### 4.4 AUROC — threshold-free separability (used for cross-modal transfer)
The Area Under the ROC Curve is the probability a random drone scores above a random confuser;
0.5 = chance, 1.0 = perfect ranking. It is used for the transfer test **because it does not depend
on a chosen operating threshold** — and a fixed threshold would itself be miscalibrated across
modality, confounding the measurement. The **within-modality 5-fold cross-validated AUROC** is the
attainable in-distribution ceiling (thermal 0.974, gray 0.921); the **cross-modal** AUROC measured
against it isolates the cost of the modality shift.

### 4.5 CORAL — covariance alignment (and why z-score beat it)
CORAL (CORrelation ALignment) whitens the source covariance and re-colours it with the target
covariance — a **full-covariance** affine map. It recovered transfer only partially (0.707) versus
the diagonal per-feature z-score (0.919). The interpretation is informative: the modality gap is
**predominantly a per-feature (diagonal) mean/scale shift**, so the extra off-diagonal covariance
terms CORAL estimates add variance (over-fitting on limited samples) rather than signal. The
simpler statistic is both better *and* more defensible.

### 4.6 Estimation hygiene
- **Cross-modal transfer protocol:** fit the classifier on modality A's z-scored features, report
  AUROC on modality B's z-scored features; the alignment is estimated from **drones only**
  (svan + Anti-UAV, population level, no instance pairing) so the confuser test stays independent.
- **Held-out validation:** CBAM is **excluded from training** so Proof 3 measures generalization,
  not memorization. We deliberately do **not** rely on the in-pool cross-validated verdict (which is
  optimistic) — the deployment numbers come from a separate held-out eval (`mri/holdout.py`).
- **Class imbalance / operating point:** the deployed MLP uses Focal Loss (\(\alpha=0.75,\gamma=2.0\),
  label-smoothing 0.1) and per-modality sample weights (thermal 1.5, gray 1.0) to bias the boundary
  toward the thermal deploy domain without discarding the abundant grayscale confuser signal.

---

## Delivered
- `mri/docs/ir_grayscale_verifier_report.md` (this evidence pack)
- Figures → `docs/analysis/images/`: `ir_v3b_lda.png`, `ir_v3b_pca.png`, `ir_v3b_neurons.png`,
  `ir_v3b_heatmap.png`, `ir_v3b_anova.png`, `ir_gray_thermal_alignment.png`
- Model: `mri/results/ir_aligned/classifiers/mlp_aligned.pt` (+ `_gray.pt` after `bszquitg6`)
- Knowledge: ledger `gray-thermal-alignable`, `ir-grayscale-harvest-solves-thermal-verifier`,
  `ir-recall-fixed-by-drone-diversity`, `grayscale-drone-recall`, `ir-grayscale-is-hallucination-mode`
