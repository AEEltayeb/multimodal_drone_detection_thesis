"""
build_dashboard_nb.py — Generate docs/analysis/eval_1000_results.ipynb
from the eval CSVs. Re-run whenever CSVs change.

Reads from:
  - eval/results/detector_eval/*.csv          (antiuav, svanstrom: per step)
  - docs/analysis/full_pipeline_ablations/csv/softveto_ablation_selcom_960.csv
        (drone-video aggregate stages with soft-veto)
  - docs/analysis/full_pipeline_ablations/csv/drone_video_tests.csv
        (drone-video confuser stages by category)
  - docs/analysis/full_pipeline_ablations/csv/eval_drone_video_per_clip.csv
  - docs/analysis/full_pipeline_ablations/csv/eval_drone_video_confuser_per_clip.csv

Writes:
  docs/analysis/eval_1000_results.ipynb
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "analysis" / "eval_1000_results.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex,
        "metadata": {},
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex,
        "metadata": {},
        "outputs": [],
        "source": src.rstrip("\n").splitlines(keepends=True),
    }


cells: list[dict] = []

# ── Header ──────────────────────────────────────────────────────────
cells.append(md("""# Detector Ablation Results — Dashboard

Plots and per-step charts backing [`eval_1000_results.md`](eval_1000_results.md).
All numbers are read directly from CSVs on disk so this notebook stays in sync
with the underlying evals.

**Sources**

| Step | CSV |
|---|---|
| 1–4 (Anti-UAV, Svanström) | `eval/results/detector_eval/` |
| Drone-video aggregate stages | `docs/analysis/full_pipeline_ablations/csv/softveto_ablation_selcom_960.csv` |
| Confuser by category | `docs/analysis/full_pipeline_ablations/csv/drone_video_tests.csv` |
| Per-clip drone | `docs/analysis/full_pipeline_ablations/csv/eval_drone_video_per_clip.csv` |
| Per-clip confuser | `docs/analysis/full_pipeline_ablations/csv/eval_drone_video_confuser_per_clip.csv` |

To refresh: rerun the relevant `eval/*.py` scripts, then `python eval/build_dashboard_nb.py`.
"""))

# ── Setup cell ──────────────────────────────────────────────────────
cells.append(code("""# Setup
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110

# Walk up to repo root from notebook location
NB_DIR = Path.cwd()
REPO = NB_DIR
while not (REPO / '.git').exists() and REPO.parent != REPO:
    REPO = REPO.parent
print('Repo root:', REPO)

DET = REPO / 'eval' / 'results' / 'detector_eval'
FPA = REPO / 'docs' / 'analysis' / 'full_pipeline_ablations' / 'csv'


def load_det(model, ds):
    df = pd.read_csv(DET / f'{model}_{ds}_detection.csv')
    return df[df['size'] == 'all'].iloc[0]


def load_fl(model, ds):
    df = pd.read_csv(DET / f'{model}_{ds}_frame_level.csv')
    return df[df['size'] == 'all'].iloc[0]


def load_stage(prefix, model, ds):
    return pd.read_csv(DET / f'{prefix}_{model}_{ds}.csv').iloc[0]


# Drone-video aggregate stages (RGB-only, soft-veto sweep)
SV960 = pd.read_csv(FPA / 'softveto_ablation_selcom_960.csv')
DV_TESTS = pd.read_csv(FPA / 'drone_video_tests.csv')
# Aggregate (per-stage, all sizes + per-size) for drone-video AT IMGSZ=960
# — use this for any drone-video metrics so all sections use the same detector config.
DV_AGG = pd.read_csv(FPA / 'eval_drone_video_aggregate.csv')
DV_CONF_AGG = pd.read_csv(FPA / 'eval_drone_video_confuser_aggregate.csv')

RGB = 'selcom_1280_960imgsz'
IR = 'ir_v3b'
"""))

# ── Soft-veto reference card ────────────────────────────────────────
cells.append(md("""## Soft-Veto Classifier — reference card

A deployment-time decision rule layered on top of the sa32 trust classifier's
4-class output (`reject_both`, `trust_RGB`, `trust_IR`, `trust_both`).
Same trained model, different routing logic.

**Rule (τ = 0.95):**
- If RGB has **≥1 detection**: *keep RGB*, **unless** `P(reject_both) ≥ τ` (very-confident reject).
- If RGB silent **and** classifier argmax votes IR-only or both: *fall back to IR detections*.
- Otherwise: empty.

**When we use it:**
| Operating mode | Rule | Why |
|---|---|---|
| Paired data (real IR) | **argmax** (full trust-aware) | IR-side features are real → classifier's modality arbitration works as designed |
| RGB-only / grayscale fallback | **soft-veto τ=0.95** | IR branch sees grayscale-RGB → OOD → argmax over-rejects legit drone frames; soft-veto fail-opens for RGB |

**Datasets in this notebook:**
- antiuav, svanstrom, ir_test → **argmax**
- drone_video, rgb_test → **soft-veto**"""))

# ── Example frames ──────────────────────────────────────────────────
cells.append(md("""## Example frames per dataset

One composite per dataset showing representative frames with bounding-box overlays.

**Legend** — *cyan dashed* = ground-truth box · *yellow* = raw detector · *lime* = boxes surviving the full pipeline (classifier + alert-gate patch) · *red dashed* = boxes dropped by classifier or patch.

*Click any image to open it full-resolution in a new tab.*

<table>
<tr><th>Dataset</th><th>Image (click to zoom)</th></tr>
<tr><td><b>antiuav</b> (paired)</td><td><a href="full_pipeline_ablations/plots/antiuav_examples.png" target="_blank"><img src="full_pipeline_ablations/plots/antiuav_examples.png" width="600"/></a></td></tr>
<tr><td><b>svanstrom</b> (paired, confuser-heavy)</td><td><a href="full_pipeline_ablations/plots/svanstrom_examples.png" target="_blank"><img src="full_pipeline_ablations/plots/svanstrom_examples.png" width="600"/></a></td></tr>
<tr><td><b>drone_video tests</b> (RGB-only)</td><td><a href="full_pipeline_ablations/plots/drone_video_examples.png" target="_blank"><img src="full_pipeline_ablations/plots/drone_video_examples.png" width="600"/></a></td></tr>
<tr><td><b>rgb_test</b> (cross-domain RGB)</td><td><a href="full_pipeline_ablations/plots/rgb_test_examples.png" target="_blank"><img src="full_pipeline_ablations/plots/rgb_test_examples.png" width="600"/></a></td></tr>
<tr><td><b>ir_test</b> (IR-primary)</td><td><a href="full_pipeline_ablations/plots/ir_test_examples.png" target="_blank"><img src="full_pipeline_ablations/plots/ir_test_examples.png" width="600"/></a></td></tr>
</table>
"""))

# ── Headline ────────────────────────────────────────────────────────
cells.append(md("""## Headline — best F1 per dataset"""))

cells.append(code("""# Best F1 across pipeline per dataset
rows = []
for ds in ('antiuav', 'svanstrom'):
    rgb = load_det(RGB, ds)
    ir = load_det(IR, ds)
    clf = load_stage('classifier_sa32', RGB, ds) if False else pd.read_csv(DET / f'classifier_sa32_{ds}.csv').iloc[0]
    rows.append({'dataset': ds, 'stage': 'RGB only',   'F1': rgb['f1']})
    rows.append({'dataset': ds, 'stage': 'IR only',    'F1': ir['f1']})
    rows.append({'dataset': ds, 'stage': 'Classifier', 'F1': clf['F1']})

# Drone-video from soft-veto CSV
dv = SV960[SV960['dataset'] == 'drone_video_drone']
def stage_f1(name):
    sub = dv[dv['stage'] == name]
    return float(sub['F1'].iloc[0]) if len(sub) else None
rows += [
    {'dataset': 'drone_video', 'stage': 'RGB only',          'F1': stage_f1('rgb_only')},
    {'dataset': 'drone_video', 'stage': 'IR-gray',           'F1': stage_f1('ir_grayscale')},
    {'dataset': 'drone_video', 'stage': 'Soft-veto (τ=0.95)', 'F1': stage_f1('softveto_0.95')},
]
df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(df, x='dataset', y='F1', hue='stage', ax=ax, palette='deep')
ax.set_ylim(0, 1.0)
ax.set_title('Headline — best stage F1 per dataset')
for c in ax.containers:
    ax.bar_label(c, fmt='%.3f', fontsize=8)
plt.tight_layout(); plt.show()
"""))

# ── Step 1 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 1 — Base Detector Performance (raw YOLO)

P, R, F1 and frame-level FP% / TN% for each modality on each dataset before any pipeline components."""))

cells.append(code("""# Step 1: base detector P/R/F1
rows = []
for ds in ('antiuav', 'svanstrom'):
    for model, label in [(RGB, 'RGB (selcom@960)'), (IR, 'IR (ir_v3b)')]:
        d = load_det(model, ds)
        fl = load_fl(model, ds)
        rows.append({'dataset': ds, 'model': label,
                     'P': d['precision'], 'R': d['recall'], 'F1': d['f1'],
                     'FP%': fl['FP_pct'], 'TN%': fl['TN_pct']})
# Drone-video
for stage, label in [('rgb_only', 'RGB (selcom@960)'),
                      ('ir_grayscale', 'IR-gray (ir_v3b)')]:
    s = dv[dv['stage'] == stage].iloc[0]
    rows.append({'dataset': 'drone_video', 'model': label,
                 'P': s['P'], 'R': s['R'], 'F1': s['F1'],
                 'FP%': None, 'TN%': None})
step1 = pd.DataFrame(rows)
display(step1.round(4))
"""))

cells.append(code("""# Plot Step 1 — F1 / P / R panels
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, metric in zip(axes, ('F1', 'P', 'R')):
    sns.barplot(step1, x='dataset', y=metric, hue='model', ax=ax, palette='deep')
    ax.set_ylim(0, 1.0); ax.set_title(metric)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=7)
    ax.legend(loc='lower right', fontsize=8)
plt.suptitle('Step 1 — base detector by dataset & modality')
plt.tight_layout(); plt.show()
"""))

# ── Step 2 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 2 — Temporal Voting (+ TROI on paired)

Shows base → +temporal → +TROI F1 progression. Drone-video has no TROI run."""))

cells.append(code("""# Step 2: temporal + TROI
def tempvals(ds, model):
    base = load_det(model, ds)
    temp = load_stage('temporal', model, ds)
    troi = load_stage('troi', model, ds)
    return [
        {'dataset': ds, 'model': model, 'stage': 'base', 'P': base['precision'], 'R': base['recall'], 'F1': base['f1']},
        {'dataset': ds, 'model': model, 'stage': '+temporal', 'P': temp['P'], 'R': temp['R'], 'F1': temp['F1']},
        {'dataset': ds, 'model': model, 'stage': '+TROI', 'P': troi['P'], 'R': troi['R'], 'F1': troi['F1']},
    ]

rows = []
for ds in ('antiuav', 'svanstrom'):
    for model in (RGB, IR):
        rows += tempvals(ds, model)
# Drone-video temporal — pull from the IMGSZ=960 aggregate CSV
for stage_key, model_label, stage_label in [
    ('S0_rgb', RGB, 'base'),
    ('temporal__S0_rgb', RGB, '+temporal'),
    ('S0_ir_grayscale', IR, 'base'),
    ('temporal__S0_ir_grayscale', IR, '+temporal'),
]:
    sub = DV_AGG[DV_AGG['stage'] == stage_key]
    if len(sub):
        # Pick the 'all' / 'segment' row depending on stage
        if 'temporal' in stage_key:
            row = sub.iloc[0]
        else:
            row = sub[sub['size'] == 'all'].iloc[0]
        rows.append({'dataset': 'drone_video', 'model': model_label,
                     'stage': stage_label, 'P': row['P'], 'R': row['R'], 'F1': row['F1']})

step2 = pd.DataFrame(rows)
display(step2.round(4))
"""))

cells.append(code("""# Plot Step 2 — F1 progression line plot (base → +temporal → +TROI)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
stage_order = ['base', '+temporal', '+TROI']
for ax, ds in zip(axes, ('antiuav', 'svanstrom', 'drone_video')):
    sub = step2[step2['dataset'] == ds].copy()
    if sub.empty:
        ax.set_title(f'{ds} (no data)'); continue
    sub['stage'] = pd.Categorical(sub['stage'], categories=stage_order, ordered=True)
    sub = sub.sort_values(['model', 'stage'])
    for model in sub['model'].unique():
        m = sub[sub['model'] == model]
        ax.plot(m['stage'].astype(str), m['F1'], marker='o', label=str(model))
        for _, row in m.iterrows():
            ax.annotate(f'{row["F1"]:.3f}', (str(row['stage']), row['F1']),
                        textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title(ds); ax.set_ylabel('F1')
    ax.legend(fontsize=8, loc='lower right')
plt.suptitle('Step 2 — F1 progression: base → +temporal → +TROI (drone_video has no TROI)')
plt.tight_layout(); plt.show()
"""))

# ── Step 3 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 3 — Patch Verifier & Alert Gate

Patch verifier (`rgb_filter`/`ir_filter`, threshold 0.70) on raw detections, vs. the production
alert-gate path (patch only on the temporal decision boundary).

> *Drone-video patch / alert-gate impact is reported separately in **Step 6** (confuser categories)
> and **Step 7** (per-clip soft-veto + filter). The patch-verifier CSV format here only covers
> Anti-UAV + Svanström.*"""))

cells.append(code("""# Step 3: pipeline cascade base → +temporal → +alert_gate (= +temporal+patch)
# Ablation: +patch (per-frame patch, not pipeline).
rows = []
for ds in ('antiuav', 'svanstrom'):
    for model in (RGB, IR):
        base = load_det(model, ds)
        rows.append({'dataset': ds, 'model': model, 'stage': 'base',
                     'P': base['precision'], 'R': base['recall'], 'F1': base['f1'],
                     'kind': 'pipeline'})
        try:
            temp = load_stage('temporal', model, ds)
            rows.append({'dataset': ds, 'model': model, 'stage': '+temporal',
                         'P': temp['P'], 'R': temp['R'], 'F1': temp['F1'],
                         'kind': 'pipeline'})
        except FileNotFoundError:
            pass
        try:
            ag = load_stage('alert_gate', model, ds)
            rows.append({'dataset': ds, 'model': model, 'stage': '+alert_gate',
                         'P': ag['P'], 'R': ag['R'], 'F1': ag['F1'],
                         'kind': 'pipeline'})
        except FileNotFoundError:
            pass
        try:
            patch = load_stage('patch', model, ds)
            rows.append({'dataset': ds, 'model': model, 'stage': '+patch',
                         'P': patch['P'], 'R': patch['R'], 'F1': patch['F1'],
                         'kind': 'ablation'})
        except FileNotFoundError:
            pass
step3 = pd.DataFrame(rows)
display(step3.round(4))
"""))

cells.append(code("""# Plot Step 3 — Svanström pipeline cascade with temporal + alert-gate; +patch as ablation triangle
sub = step3[step3['dataset'] == 'svanstrom'].copy()
order = ['base', '+temporal', '+alert_gate', '+patch']
sub['stage'] = pd.Categorical(sub['stage'], categories=order, ordered=True)
sub = sub.sort_values(['model', 'stage'])

PIPELINE_STAGES_S3 = {'base', '+temporal', '+alert_gate'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {RGB: 'tab:blue', IR: 'tab:orange'}
for ax, metric in zip(axes, ('F1', 'P')):
    for model in sub['model'].unique():
        m = sub[sub['model'] == model]
        # Pipeline line (skip ablation +patch)
        pipe = m[m['stage'].isin(PIPELINE_STAGES_S3)].copy()
        ax.plot(pipe['stage'].astype(str), pipe[metric], '-o', markersize=10,
                color=colors[model], label=f"{model} (pipeline)", linewidth=2, zorder=3)
        # Ablation triangles
        abl = m[~m['stage'].isin(PIPELINE_STAGES_S3)]
        ax.scatter(abl['stage'].astype(str), abl[metric], marker='^', s=140,
                   color=colors[model], edgecolor='black', linewidth=0.7,
                   alpha=0.75, label=f"{model} (ablation)", zorder=4)
        for _, row in m.iterrows():
            ax.annotate(f'{row[metric]:.3f}', (str(row['stage']), row[metric]),
                        textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    ax.axvline(x=2.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(2.5, 0.02, '  ← pipeline | ablation →  ', ha='center', fontsize=8, color='dimgrey')
    ax.set_ylim(0, 1.0); ax.set_ylabel(metric); ax.set_title(f'Svanström — {metric}')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='lower right')
plt.suptitle('Step 3 — pipeline cascade (base → +temporal → +alert_gate). △ ablation = +patch every frame')
plt.tight_layout(); plt.show()
"""))

# ── Step 4 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 4 — Scene-Aware Trust Classifier (SA32)

Trust-aware multi-modality routing. Argmax on paired data; soft-veto (τ=0.95) on RGB-only drone-video."""))

cells.append(code("""# Step 4: classifier impact
rows = []
for ds in ('antiuav', 'svanstrom'):
    rgb_base = load_det(RGB, ds)
    ir_base = load_det(IR, ds)
    clf = pd.read_csv(DET / f'classifier_sa32_{ds}.csv').iloc[0]
    rows += [
        {'dataset': ds, 'stage': 'RGB only',   'F1': rgb_base['f1'], 'P': rgb_base['precision'], 'R': rgb_base['recall']},
        {'dataset': ds, 'stage': 'IR only',    'F1': ir_base['f1'],  'P': ir_base['precision'],  'R': ir_base['recall']},
        {'dataset': ds, 'stage': 'Classifier', 'F1': clf['F1'],      'P': clf['P'],              'R': clf['R']},
    ]

# Drone-video: rgb, ir-gray, argmax, soft-veto
for stage_name, label in [('rgb_only', 'RGB only'),
                            ('ir_grayscale', 'IR-gray only'),
                            ('classifier_argmax', 'Classifier (argmax)'),
                            ('softveto_0.95', 'Soft-veto τ=0.95')]:
    sub = dv[dv['stage'] == stage_name]
    if len(sub):
        s = sub.iloc[0]
        rows.append({'dataset': 'drone_video', 'stage': label,
                     'F1': s['F1'], 'P': s['P'], 'R': s['R']})

step4 = pd.DataFrame(rows)
display(step4.round(4))
"""))

cells.append(code("""# Plot Step 4 — grouped bars per dataset (P, R, F1)
# Mark off-pipeline stages with a hatched pattern: argmax on RGB-only or softveto on paired.
OFF_PIPELINE = {
    'antiuav':     {'Soft-veto τ=0.95'},               # paired -> argmax is pipeline
    'svanstrom':   {'Soft-veto τ=0.95'},
    'drone_video': {'Classifier (argmax)'},            # RGB-only -> softveto is pipeline
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ('antiuav', 'svanstrom', 'drone_video')):
    sub = step4[step4['dataset'] == ds].reset_index(drop=True)
    sub_long = sub.melt(id_vars=['stage'], value_vars=['P', 'R', 'F1'],
                        var_name='metric', value_name='value')
    bars = sns.barplot(sub_long, x='stage', y='value', hue='metric',
                        ax=ax, palette='Set2')
    # Hatch ablation-only stages
    off = OFF_PIPELINE.get(ds, set())
    stages_in_order = sub['stage'].tolist()
    for cont_idx, container in enumerate(ax.containers):
        for bar_idx, bar in enumerate(container):
            if stages_in_order[bar_idx] in off:
                bar.set_hatch('//')
                bar.set_edgecolor('black')
                bar.set_alpha(0.85)
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
    ax.set_ylim(0, 1.1); ax.set_xlabel('')
    ax.set_title(ds); ax.tick_params(axis='x', rotation=20)
plt.suptitle('Step 4 — classifier impact (P / R / F1).  Hatched bars = off-pipeline ablation '
             '(argmax on RGB-only, softveto on paired)')
plt.tight_layout(); plt.show()
"""))

# ── Step 5 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 5 — Per-Size Detection Breakdown"""))

cells.append(code("""# Step 5: per-size — for each model x dataset
def load_persize(model, ds):
    df = pd.read_csv(DET / f'{model}_{ds}_detection.csv')
    return df[df['size'].isin(['small', 'medium', 'large'])]


fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
for col, ds in enumerate(('antiuav', 'svanstrom')):
    for row, (model, label) in enumerate([(RGB, 'RGB'), (IR, 'IR')]):
        ax = axes[row, col]
        sub = load_persize(model, ds)
        sub_long = sub.melt(id_vars=['size'], value_vars=['precision', 'recall', 'f1'],
                            var_name='metric', value_name='value')
        sns.barplot(sub_long, x='size', y='value', hue='metric', ax=ax, palette='Set2')
        ax.set_ylim(0, 1.05); ax.set_title(f'{ds} — {label}')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', fontsize=7)
plt.suptitle('Step 5 — per-size P/R/F1')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# Per-size for drone_video — sourced from eval_drone_video_aggregate.csv (selcom@960)
# One subplot per model so RGB and IR-gray are read separately, not averaged.
rgb = DV_AGG[(DV_AGG['stage']=='S0_rgb') & (DV_AGG['size'].isin(['small','medium','large']))].assign(model='RGB (selcom@960)')
ir = DV_AGG[(DV_AGG['stage']=='S0_ir_grayscale') & (DV_AGG['size'].isin(['small','medium','large']))].assign(model='IR-gray (ir_v3b)')

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (label, sub) in zip(axes, [('RGB (selcom@960)', rgb), ('IR-gray (ir_v3b)', ir)]):
    long = sub.melt(id_vars=['size'], value_vars=['P', 'R', 'F1'],
                    var_name='metric', value_name='value')
    sns.barplot(long, x='size', y='value', hue='metric', ax=ax, palette='Set2')
    ax.set_ylim(0, 1.05); ax.set_title(f'drone_video — {label}')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=7)
plt.tight_layout(); plt.show()
"""))

# ── Step 6 ──────────────────────────────────────────────────────────
cells.append(md("""## Step 6 — Confuser-Clip Suppression (drone-video confuser categories)

Lower fire rate is better. Stage = pipeline configuration; bars = segment-level FR%."""))

cells.append(code("""# Step 6: confuser FR% by category — pipeline cascade with temporal made explicit
# Pipeline cascade (circles, connected):
#   RGB raw (frame)  → IR-gray raw (frame)  → +temporal  → +classifier soft-veto  → +alert_gate
# Frame-level for the raw detectors (pre-temporal), segment-level for everything after temporal.
# Plotting them on the same y-axis (FR%) is meaningful because lower = better in both worlds.
pipeline_rows = []
abl_rows = []
for cat in ('birds', 'airplanes', 'helicopters'):
    sub = DV_CONF_AGG[DV_CONF_AGG['category'] == cat]
    if sub.empty: continue
    def _get(stage, col):
        r = sub[sub['stage'] == stage]
        return float(r[col].iloc[0]) if len(r) else None
    # Pipeline path
    pipeline_rows += [
        {'category': cat, 'stage': 'RGB (frame)',        'order': 0, 'fr': _get('S0_rgb', 'fr_frame_pct')},
        {'category': cat, 'stage': 'IR-gray (frame)',    'order': 1, 'fr': _get('S0_ir_grayscale', 'fr_frame_pct')},
        {'category': cat, 'stage': '+ temporal (2/3)',   'order': 2, 'fr': _get('S0_rgb', 'fr_seg_pct')},
        {'category': cat, 'stage': '+ soft-veto',         'order': 3, 'fr': _get('S4_clf_softveto', 'fr_seg_pct')},
        {'category': cat, 'stage': '+ alert_gate',        'order': 4, 'fr': _get('S4_softveto_patch', 'fr_seg_pct')},
    ]
    # Ablation triangles (segment-level, off-pipeline)
    abl_rows += [
        {'category': cat, 'stage': 'RGB+patch',          'order': 5, 'fr': _get('S2_rgb_patch', 'fr_seg_pct')},
        {'category': cat, 'stage': 'IR-gray+patch',      'order': 6, 'fr': _get('S2_ir_patch', 'fr_seg_pct')},
        {'category': cat, 'stage': 'Clf argmax',          'order': 7, 'fr': _get('S4_clf_argmax', 'fr_seg_pct')},
    ]
pipe_df = pd.DataFrame(pipeline_rows).sort_values(['category', 'order'])
abl_df = pd.DataFrame(abl_rows).sort_values(['category', 'order'])
all_stages = list(pipe_df.drop_duplicates('order').sort_values('order')['stage']) + \
             list(abl_df.drop_duplicates('order').sort_values('order')['stage'])

# Pipeline (circles + connecting line) on the LEFT; ablation (triangles, unconnected) on the RIGHT
n_pipe = pipe_df['stage'].nunique()

fig, ax = plt.subplots(figsize=(14, 5.5))
for cat, color in zip(('birds', 'airplanes', 'helicopters'),
                       ('tab:blue', 'tab:orange', 'tab:green')):
    pipe = pipe_df[pipe_df['category'] == cat]
    abl = abl_df[abl_df['category'] == cat]
    if pipe.empty: continue
    ax.plot(pipe['stage'], pipe['fr'], marker='o', markersize=10,
            label=f'{cat} (pipeline)', linewidth=2.2, color=color)
    ax.scatter(abl['stage'], abl['fr'], marker='^', s=140,
                color=color, edgecolor='black', linewidth=0.8, alpha=0.7,
                label=f'{cat} (ablation)')
    for _, row in pd.concat([pipe, abl]).iterrows():
        if row['fr'] is None: continue
        ax.annotate(f'{row["fr"]:.1f}%', (row['stage'], row['fr']),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=7)

ax.axvline(x=n_pipe - 0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
y_top = ax.get_ylim()[1] * 0.97
ax.text(n_pipe - 0.5, y_top, '  ← pipeline  |  ablation →  ',
        ha='center', fontsize=9, color='dimgrey')
ax.set_ylabel('FR% (frame-level for first two stages, segment-level after)')
ax.set_title('Step 6 — confuser fire-rate across pipeline (RGB→IR-gray→+temporal→+soft-veto→+alert_gate).  ○ pipeline, △ ablation')
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', ncol=3, fontsize=8)
ax.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.show()
"""))

# ── Step 7a — Per-clip drone ─────────────────────────────────────────
cells.append(md("""## Step 7 — Per-Video Breakdown

Per-clip RGB vs IR-grayscale on the realistic mixed-scene drone clips (and the confuser clips).
The slope chart highlights which clips IR-grayscale actually wins."""))

cells.append(code("""# Per-clip drone F1: slope chart RGB -> IR-gray, ★ where IR-gray wins
per_clip = pd.read_csv(FPA / 'eval_drone_video_per_clip.csv')

# We want one F1 per clip for rgb (S0_rgb) and ir-gray (S0_ir_grayscale) and softveto (S4_clf_softveto)
def pivot_stage(stage):
    sub = per_clip[per_clip['stage'] == stage][['clip', 'F1']].rename(columns={'F1': stage})
    return sub

p = pivot_stage('S0_rgb').merge(pivot_stage('S0_ir_grayscale'), on='clip').merge(pivot_stage('S4_clf_softveto'), on='clip')
p['delta'] = p['S0_ir_grayscale'] - p['S0_rgb']
p = p.sort_values('delta', ascending=False)
p
"""))

cells.append(code("""# Slope chart RGB -> IR-gray per drone clip
fig, ax = plt.subplots(figsize=(11, 6))
for _, row in p.iterrows():
    color = 'tab:green' if row['delta'] > 0 else 'tab:red'
    ax.plot([0, 1], [row['S0_rgb'], row['S0_ir_grayscale']], color=color, marker='o', alpha=0.7)
    ax.text(1.02, row['S0_ir_grayscale'], row['clip'][:40], fontsize=8, va='center')
ax.set_xticks([0, 1])
ax.set_xticklabels(['RGB', 'IR-gray'])
ax.set_ylabel('F1')
ax.set_title('Per-clip RGB → IR-gray F1 (green = IR-gray wins)')
ax.set_ylim(0, 1.0)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# Bar of softveto F1 vs RGB F1 per clip — shows where softveto retains the win
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(p))
w = 0.28
ax.bar(x - w, p['S0_rgb'], w, label='RGB', color='tab:blue')
ax.bar(x, p['S0_ir_grayscale'], w, label='IR-gray', color='tab:orange')
ax.bar(x + w, p['S4_clf_softveto'], w, label='Soft-veto', color='tab:green')
ax.set_xticks(x)
ax.set_xticklabels([c[:25] for c in p['clip']], rotation=30, ha='right')
ax.set_ylabel('F1'); ax.set_ylim(0, 1.0)
ax.legend()
ax.set_title('Per-clip — RGB vs IR-gray vs Soft-veto')
plt.tight_layout(); plt.show()
"""))

# ── Step 7b — Per-clip confuser ──────────────────────────────────────
cells.append(code("""# Per-clip confuser FR%: RGB vs IR-gray vs softveto+patch
conf_clip = pd.read_csv(FPA / 'eval_drone_video_confuser_per_clip.csv')
# pivot fr_seg_pct per stage
def pivot_fr(stage):
    s = conf_clip[conf_clip['stage'] == stage][['category', 'clip', 'fr_seg_pct']].rename(columns={'fr_seg_pct': stage})
    return s

p_conf = pivot_fr('S0_rgb').merge(pivot_fr('S0_ir_grayscale'), on=['category', 'clip'])
p_conf = p_conf.merge(pivot_fr('S4_softveto_patch'), on=['category', 'clip'])
p_conf['delta'] = p_conf['S0_ir_grayscale'] - p_conf['S0_rgb']
p_conf = p_conf.sort_values(['category', 'delta'])
p_conf
"""))

cells.append(code("""# Bar chart confuser FR% per clip
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, cat in zip(axes, ('birds', 'airplanes', 'helicopters')):
    sub = p_conf[p_conf['category'] == cat].reset_index(drop=True)
    x = np.arange(len(sub))
    w = 0.28
    ax.bar(x - w, sub['S0_rgb'], w, label='RGB', color='tab:blue')
    ax.bar(x, sub['S0_ir_grayscale'], w, label='IR-gray', color='tab:orange')
    ax.bar(x + w, sub['S4_softveto_patch'], w, label='Soft-veto+patch', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:25] for c in sub['clip']], rotation=35, ha='right', fontsize=8)
    ax.set_title(f'{cat} — confuser FR% per clip')
    ax.set_ylabel('Segment FR%')
    ax.legend(fontsize=8)
plt.suptitle('Step 7 — per-clip confuser suppression (lower is better)')
plt.tight_layout(); plt.show()
"""))

cells.append(md("""---

**Read:** IR-grayscale beats RGB on every confuser clip (or ties at 0%). On drone clips, IR-grayscale wins outright on the hardest scenes (mountain/sky, distant drone) and loses badly on close-up takeoff. Soft-veto picks the better side per clip with no re-training.
"""))

# ───────────────────────────────────────────────────────────────────────
# Step 8 — RGB Test Split (cross-domain mixed dataset, soft-veto classifier)
# ───────────────────────────────────────────────────────────────────────
cells.append(md("""## Step 8 — RGB Test Split (cross-domain mixed RGB dataset)

`G:/drone/dataset/dataset/test` — 17,209 images, uniform-stride-sampled to ~1,000.
**Mixed dataset**: drone-positive (`mav`, `anti`, `dut`, `wosdetc`) + confuser categories
(`AirBird`, `BDD100K`, `VIRAT`, `UA-DETRAC`, `FBD-SV`).

Pipeline (RGB-only deployment): **selcom_1280@960 → ir_v3b on grayscale-RGB →
+temporal → +classifier (soft-veto τ=0.95) → +alert_gate (rgb_filter at decision)**.

Two plots per dataset: pipeline-layer P/R/F1, then per-confuser-category FR%.
"""))

cells.append(code("""# Step 8 — RGB test pipeline cascade
RGB_AGG = pd.read_csv(FPA / 'eval_rgb_test_aggregate.csv')
RGB_CONF = pd.read_csv(FPA / 'eval_rgb_test_confuser.csv')

# Pipeline cascade for rgb_test, temporal stages inserted between base detector and classifier.
# We synthesize segment-level P/R from the aggregate CSV's seg_* columns and add them as
# virtual stages 'S0_rgb__temporal' and 'S0_ir__temporal' in the dataframe.
def _add_temporal_stages(df):
    extra = []
    for s in ('S0_rgb', 'S0_ir'):
        row = df[(df['stage'] == s) & (df['size'] == 'all')]
        if row.empty: continue
        r = row.iloc[0]
        tp, fp, fn = int(r['seg_TP']), int(r['seg_FP']), int(r['seg_FN'])
        P = tp/(tp+fp) if (tp+fp) else 0.0
        R = tp/(tp+fn) if (tp+fn) else 0.0
        F = 2*P*R/(P+R) if (P+R) else 0.0
        extra.append({'stage': f'{s}__temporal', 'size': 'all',
                       'TP': tp, 'FP': fp, 'FN': fn, 'n_gt': tp+fn,
                       'P': round(P,4), 'R': round(R,4), 'F1': round(F,4)})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

RGB_AGG = _add_temporal_stages(RGB_AGG)

PIPELINE_RGB = ['S0_rgb', 'S0_ir', 'S0_rgb__temporal', 'S4_clf', 'S4_clf_patch']
ABLATION_RGB = ['S2_rgb_patch', 'S2_ir_patch', 'S4_clf_other_mode', 'S0_ir__temporal']
STAGE_LABEL = {
    'S0_rgb': 'RGB (frame)', 'S0_ir': 'IR-gray (frame)',
    'S0_rgb__temporal': '+temporal (RGB)',
    'S0_ir__temporal': 'IR-gray + temporal',
    'S2_rgb_patch': 'RGB+patch', 'S2_ir_patch': 'IR-gray+patch',
    'S4_clf': '+soft-veto',
    'S4_clf_patch': '+alert_gate',
    'S4_clf_other_mode': 'Clf argmax (ablation)',
}

# Get per-stage P, R, F1 (all-size row)
all_rgb = RGB_AGG[RGB_AGG['size'] == 'all'].copy()
all_rgb['label'] = all_rgb['stage'].map(STAGE_LABEL)
display(all_rgb[['stage', 'label', 'TP', 'FP', 'FN', 'P', 'R', 'F1']].round(4))
"""))

cells.append(code("""# Step 8 plot 1 — P / R per pipeline layer (line plot, pipeline ○ vs ablation △)
def plot_pipeline_PR(df, pipeline_stages, ablation_stages, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ('P', 'R')):
        pipe = df[df['stage'].isin(pipeline_stages)].copy()
        pipe = pipe.set_index('stage').loc[pipeline_stages].reset_index()
        abl = df[df['stage'].isin(ablation_stages)].copy()
        # Pipeline line
        ax.plot(pipe['label'], pipe[metric], '-o', markersize=11, linewidth=2.2,
                color='tab:blue', label='Pipeline')
        # Ablation triangles
        ax.scatter(abl['label'], abl[metric], marker='^', s=140,
                    color='tab:red', edgecolor='black', linewidth=0.8, alpha=0.75,
                    label='Ablation')
        for _, row in pd.concat([pipe, abl]).iterrows():
            ax.annotate(f"{row[metric]:.3f}", (row['label'], row[metric]),
                        textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric)
        ax.set_title(metric); ax.tick_params(axis='x', rotation=20)
        ax.legend(loc='lower left', fontsize=8)
    plt.suptitle(title)
    plt.tight_layout(); plt.show()

plot_pipeline_PR(all_rgb, PIPELINE_RGB, ABLATION_RGB,
                  'Step 8 — RGB test pipeline: P and R per layer (○ pipeline, △ ablation)')
"""))

cells.append(md("""**Confuser fire-rate on rgb_test (bird / airplane / helicopter frames only).**
Categories below are filtered to filenames containing those keywords; other empty-label
frames (e.g. background driving scenes) are excluded.
"""))

cells.append(code("""# Step 8 — per-confuser-category FR% across pipeline (BIRD / AIRPLANE / HELICOPTER only)
CONF_CATS = ['bird', 'airplane', 'helicopter']
sub_conf = RGB_CONF[RGB_CONF['category'].isin(CONF_CATS)].copy()
display(sub_conf.pivot(index='category', columns='stage', values='fr_seg_pct')
        .reindex(columns=['S0_rgb','S2_rgb_patch','S0_ir_grayscale','S2_ir_patch',
                          'S4_clf','S4_clf_patch','S4_clf_other_mode']).round(2))
"""))

cells.append(code("""# Step 8 plot — confuser FR% per category across pipeline (○ pipeline, △ ablation)
PIPELINE_RGB_CONF = [s for s in PIPELINE_RGB if not s.endswith('__temporal')]
ABLATION_RGB_CONF = [s for s in ABLATION_RGB if not s.endswith('__temporal')]
order = PIPELINE_RGB_CONF + ABLATION_RGB_CONF
labels = [STAGE_LABEL[s] for s in order]
plot_cats = [c for c in CONF_CATS if c in sub_conf['category'].unique()]

if plot_cats:
    fig, ax = plt.subplots(figsize=(14, 5))
    palette = sns.color_palette('tab10', len(plot_cats))
    for cat, color in zip(plot_cats, palette):
        s = sub_conf[sub_conf['category'] == cat].set_index('stage').reindex(order).reset_index()
        pipe_mask = s['stage'].isin(PIPELINE_RGB_CONF)
        ax.plot(labels[:len(PIPELINE_RGB_CONF)], s[pipe_mask]['fr_seg_pct'].values,
                '-o', markersize=8, linewidth=1.8, color=color, label=cat)
        ax.scatter(labels[len(PIPELINE_RGB_CONF):],
                    s[~pipe_mask]['fr_seg_pct'].values,
                    marker='^', s=110, color=color, edgecolor='black', linewidth=0.6, alpha=0.7)
    ax.axvline(x=len(PIPELINE_RGB_CONF) - 0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Segment FR% (lower = better confuser suppression)')
    ax.set_title('Step 8 — RGB test confuser FR% (bird / airplane / helicopter filenames only)')
    ax.legend(loc='upper right', fontsize=9); ax.tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.show()
else:
    print('No bird/airplane/helicopter categories in rgb_test confuser CSV.')
"""))

# ───────────────────────────────────────────────────────────────────────
# Step 9 — IR Test Split (IR-only mixed dataset, argmax classifier)
# ───────────────────────────────────────────────────────────────────────
cells.append(md("""## Step 9 — IR Test Split (IR-only mixed dataset)

`G:/drone/IR_dset_final/test` — 9,612 images, uniform-stride-sampled to ~1,000.
**Mixed dataset**: drone-positive (mostly `dv5_auv`, some `dv5_dv4`) + confuser categories
(`flir_video-*`, `dv5_dv4_bird_*`, etc.).

Pipeline (IR-primary deployment): **ir_v3b on IR → +temporal → +classifier (argmax, with
synthetic RGB = selcom on IR-as-RGB) → +alert_gate (ir_filter at decision)**.

The classifier's RGB branch is fed **selcom on IR-as-3-channel** — a deliberately noisy input
to test whether the classifier correctly routes to IR (label 2/3 = IR-trust).
"""))

cells.append(code("""# Step 9 — IR test pipeline cascade
IR_AGG = pd.read_csv(FPA / 'eval_ir_test_aggregate.csv')
IR_CONF = pd.read_csv(FPA / 'eval_ir_test_confuser.csv')

IR_AGG = _add_temporal_stages(IR_AGG)

# Pipeline for ir_test (IR-primary, argmax = pipeline, softveto = ablation)
PIPELINE_IR = ['S0_ir', 'S0_rgb', 'S0_ir__temporal', 'S4_clf', 'S4_clf_patch']
ABLATION_IR = ['S2_ir_patch', 'S2_rgb_patch', 'S4_clf_other_mode', 'S0_rgb__temporal']
STAGE_LABEL_IR = {
    'S0_ir': 'IR-native (frame)', 'S0_rgb': 'selcom-on-IR (noisy)',
    'S0_ir__temporal': '+temporal (IR)',
    'S0_rgb__temporal': 'selcom-on-IR + temporal',
    'S2_ir_patch': 'IR+patch', 'S2_rgb_patch': 'selcom-on-IR+patch',
    'S4_clf': '+classifier (argmax)',
    'S4_clf_patch': '+alert_gate',
    'S4_clf_other_mode': 'Clf softveto (ablation)',
}

all_ir = IR_AGG[IR_AGG['size'] == 'all'].copy()
all_ir['label'] = all_ir['stage'].map(STAGE_LABEL_IR)
display(all_ir[['stage', 'label', 'TP', 'FP', 'FN', 'P', 'R', 'F1']].round(4))
"""))

cells.append(code("""# Step 9 plot 1 — P / R per pipeline layer
def plot_pipeline_PR_labelled(df, pipeline_stages, ablation_stages, stage_label_map, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ('P', 'R')):
        pipe = df[df['stage'].isin(pipeline_stages)].copy()
        pipe['lbl'] = pipe['stage'].map(stage_label_map)
        pipe = pipe.set_index('stage').loc[pipeline_stages].reset_index()
        pipe['lbl'] = pipe['stage'].map(stage_label_map)
        abl = df[df['stage'].isin(ablation_stages)].copy()
        abl['lbl'] = abl['stage'].map(stage_label_map)
        ax.plot(pipe['lbl'], pipe[metric], '-o', markersize=11, linewidth=2.2,
                color='tab:blue', label='Pipeline')
        ax.scatter(abl['lbl'], abl[metric], marker='^', s=140,
                    color='tab:red', edgecolor='black', linewidth=0.8, alpha=0.75,
                    label='Ablation')
        for _, row in pd.concat([pipe, abl]).iterrows():
            ax.annotate(f"{row[metric]:.3f}", (row['lbl'], row[metric]),
                        textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric); ax.set_title(metric)
        ax.tick_params(axis='x', rotation=20)
        ax.legend(loc='lower left', fontsize=8)
    plt.suptitle(title)
    plt.tight_layout(); plt.show()

plot_pipeline_PR_labelled(all_ir, PIPELINE_IR, ABLATION_IR, STAGE_LABEL_IR,
    'Step 9 — IR test pipeline: P and R per layer (○ pipeline, △ ablation)')
"""))

cells.append(code("""# Step 9 plot 2 — per-confuser-category segment FR% on IR test (BIRD / AIRPLANE / HELICOPTER only)
PIPELINE_IR_CONF = [s for s in PIPELINE_IR if not s.endswith('__temporal')]
ABLATION_IR_CONF = [s for s in ABLATION_IR if not s.endswith('__temporal')]
order_ir = PIPELINE_IR_CONF + ABLATION_IR_CONF
labels_ir = [STAGE_LABEL_IR[s] for s in order_ir]
CONF_CATS_IR = ['bird', 'airplane', 'helicopter']
conf_cats_ir = [c for c in CONF_CATS_IR if c in IR_CONF['category'].unique()]

if conf_cats_ir:
    fig, ax = plt.subplots(figsize=(15, 5.5))
    palette = sns.color_palette('tab10', len(conf_cats_ir))
    for cat, color in zip(conf_cats_ir, palette):
        sub = IR_CONF[IR_CONF['category'] == cat]
        sub = sub.set_index('stage').reindex(order_ir).reset_index()
        pipe_mask = sub['stage'].isin(PIPELINE_IR_CONF)
        ax.plot(labels_ir[:len(PIPELINE_IR_CONF)], sub[pipe_mask]['fr_seg_pct'].values,
                '-o', markersize=8, linewidth=1.8, color=color, label=cat)
        ax.scatter(labels_ir[len(PIPELINE_IR_CONF):],
                    sub[~pipe_mask]['fr_seg_pct'].values,
                    marker='^', s=110, color=color, edgecolor='black', linewidth=0.6, alpha=0.7)
    ax.axvline(x=len(PIPELINE_IR_CONF) - 0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Segment FR% (lower = better confuser suppression)')
    ax.set_title('Step 9 — IR test confuser FR% (bird / airplane / helicopter filenames only)')
    ax.legend(loc='upper right', fontsize=9); ax.tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.show()
else:
    print('No bird/airplane/helicopter categories in ir_test confuser CSV.')
"""))

# ── Assemble notebook ───────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(cells)} cells)")
