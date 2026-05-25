"""Patch eval_1000_results.ipynb in place:
   1. Add style_cmp() helper for red→green per-dataset gradient
   2. Fill drone_video FP%/TN% in Step 1
   3. Add +TROI rows for drone_video in Step 2
   4. Apply styling to Step 1/2/3/4/8/9 tables
   5. Append an aggregate section at the end
"""
import nbformat
from pathlib import Path

NB = Path('docs/analysis/eval_1000_results.ipynb')
nb = nbformat.read(NB, as_version=4)


def code(src):
    return nbformat.v4.new_code_cell(src)


def md(src):
    return nbformat.v4.new_markdown_cell(src)


# ── 1. setup helper ──────────────────────────────────────────────────
SETUP_APPEND = '''

# ── Styling helper: saturated RdYlGn gradient, auto-zoom on tight columns ──
# Spec: dark red → red → orange → yellow → green → dark green. Strong colors.
# When a column's values are tightly clustered (small variance), we narrow the
# vmin/vmax to the column's own min/max so tones still differentiate.
from matplotlib.colors import LinearSegmentedColormap as _LSC
_RYG_GOOD = _LSC.from_list('ryg_good', [
    '#67000d',  # dark red
    '#cb181d',  # red
    '#f16913',  # orange
    '#fdd835',  # yellow
    '#7cb342',  # green
    '#1b5e20',  # dark green
])
_RYG_BAD = _LSC.from_list('ryg_bad', list(reversed([
    '#67000d', '#cb181d', '#f16913', '#fdd835', '#7cb342', '#1b5e20',
])))

# Absolute metric bounds (used unless the column is tightly clustered)
_METRIC_BOUNDS = {
    'P': (0, 1), 'R': (0, 1), 'F1': (0, 1),
    'precision': (0, 1), 'recall': (0, 1), 'f1': (0, 1),
    'FP%': (0, 100), 'TN%': (0, 100), 'FR%': (0, 100),
    'FP_pct': (0, 100), 'TN_pct': (0, 100),
    'fr_seg_pct': (0, 100), 'fr_frame_pct': (0, 100), 'tn_seg_pct': (0, 100),
}

def _bounds_for(col, series):
    """Return (vmin, vmax). If the column's range covers <25% of the absolute
    scale, zoom to the column's own min/max so close values still get distinct
    tones. Otherwise use the absolute scale.
    """
    s = series.dropna()
    if len(s) == 0: return (0, 1)
    lo, hi = float(s.min()), float(s.max())
    if col in _METRIC_BOUNDS:
        amin, amax = _METRIC_BOUNDS[col]
        scale = amax - amin
        # If the data range is <25% of the absolute scale, zoom in
        if (hi - lo) < 0.25 * scale and hi > lo:
            pad = (hi - lo) * 0.05
            return (lo - pad, hi + pad)
        return (amin, amax)
    # Counts: always use column range
    return (lo, hi if hi > lo else hi + 1)


def style_cmp(df, higher_better=None, lower_better=None, group=None, fmt='{:.3f}'):
    """Saturated red→yellow→green gradient. Absolute scale for known metric
    columns, with auto-zoom when values are tightly clustered. White text on
    dark cells for readability.
    """
    higher_better = higher_better or []
    lower_better = lower_better or []
    if not higher_better and not lower_better:
        higher_better = [c for c in df.columns
                          if df[c].dtype.kind in 'fi' and c != group]
    sty = df.style
    for col in higher_better:
        if col in df.columns and df[col].dtype.kind in 'fi':
            vmin, vmax = _bounds_for(col, df[col])
            sty = sty.background_gradient(cmap=_RYG_GOOD, subset=[col],
                                            axis=0, vmin=vmin, vmax=vmax,
                                            text_color_threshold=0.4)
    for col in lower_better:
        if col in df.columns and df[col].dtype.kind in 'fi':
            vmin, vmax = _bounds_for(col, df[col])
            sty = sty.background_gradient(cmap=_RYG_BAD, subset=[col],
                                            axis=0, vmin=vmin, vmax=vmax,
                                            text_color_threshold=0.4)
    num_cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    sty = sty.format({c: fmt for c in num_cols}, na_rep='—')
    return sty
'''
nb.cells[1].source += SETUP_APPEND


# ── 2. Step 1 (cell 7): fill drone_video FP%/TN%, style ─────────────
CELL7 = '''# Step 1: base detector P/R/F1 + frame-level FP%/TN% (n = frame count)
rows = []
for ds in ('antiuav', 'svanstrom'):
    for model, label in [(RGB, 'RGB (selcom@960)'), (IR, 'IR (ir_v3b)')]:
        d = load_det(model, ds)
        fl = load_fl(model, ds)
        rows.append({'dataset': ds, 'model': label, 'n': int(fl['total']),
                     'P': d['precision'], 'R': d['recall'], 'F1': d['f1'],
                     'FP%': fl['FP_pct'], 'TN%': fl['TN_pct']})
# Drone-video — pull frame-level FP%/TN% from DV_AGG (size='all' rows of each base stage)
for stage_key, label in [('S0_rgb', 'RGB (selcom@960)'),
                          ('S0_ir_grayscale', 'IR-gray (ir_v3b)')]:
    s = DV_AGG[(DV_AGG['stage'] == stage_key) & (DV_AGG['size'] == 'all')].iloc[0]
    rows.append({'dataset': 'drone_video', 'model': label, 'n': int(s['n_frames']),
                 'P': s['P'], 'R': s['R'], 'F1': s['F1'],
                 'FP%': s['FP_pct_frame'], 'TN%': s['TN_pct_frame']})
step1 = pd.DataFrame(rows)
style_cmp(step1, higher_better=['P','R','F1','TN%'], lower_better=['FP%'], group='dataset')
'''
nb.cells[7].source = CELL7


# ── 3. Step 2 (cell 10): add +TROI for drone_video, style ───────────
CELL10 = '''# Step 2: temporal + TROI (alert-gate = temporal + patch on drone_video)
DS_N = {'antiuav': 1000, 'svanstrom': 1000, 'drone_video': int(DV_AGG[(DV_AGG['stage']=='S0_rgb') & (DV_AGG['size']=='all')]['n_frames'].iloc[0])}
def tempvals(ds, model):
    base = load_det(model, ds)
    temp = load_stage('temporal', model, ds)
    troi = load_stage('troi', model, ds)
    return [
        {'dataset': ds, 'model': model, 'n': DS_N[ds], 'stage': 'base', 'P': base['precision'], 'R': base['recall'], 'F1': base['f1']},
        {'dataset': ds, 'model': model, 'n': DS_N[ds], 'stage': '+temporal', 'P': temp['P'], 'R': temp['R'], 'F1': temp['F1']},
        {'dataset': ds, 'model': model, 'n': DS_N[ds], 'stage': '+TROI', 'P': troi['P'], 'R': troi['R'], 'F1': troi['F1']},
    ]

rows = []
for ds in ('antiuav', 'svanstrom'):
    for model in (RGB, IR):
        rows += tempvals(ds, model)

# Drone-video: base / +temporal / +TROI(=alert_gate__S2_*_patch)
DV_STAGES = [
    ('S0_rgb',                       RGB, 'base'),
    ('temporal__S0_rgb',             RGB, '+temporal'),
    ('alert_gate__S2_rgb_patch',     RGB, '+TROI'),
    ('S0_ir_grayscale',              IR, 'base'),
    ('temporal__S0_ir_grayscale',    IR, '+temporal'),
    ('alert_gate__S2_ir_patch',      IR, '+TROI'),
]
for stage_key, model_label, stage_label in DV_STAGES:
    sub = DV_AGG[DV_AGG['stage'] == stage_key]
    if sub.empty: continue
    row = sub[sub['size'] == 'all'].iloc[0] if (sub['size']=='all').any() else sub.iloc[0]
    rows.append({'dataset': 'drone_video', 'model': model_label, 'n': DS_N['drone_video'],
                 'stage': stage_label, 'P': row['P'], 'R': row['R'], 'F1': row['F1']})

step2 = pd.DataFrame(rows)
style_cmp(step2, higher_better=['P','R','F1'], group='dataset')
'''
nb.cells[10].source = CELL10


# ── 4. Step 3 (cell 13) styling + n column ──────────────────────────
nb.cells[13].source = nb.cells[13].source.replace(
    'display(step3.round(4))',
    "step3['n'] = step3['dataset'].map({'antiuav':1000,'svanstrom':1000,"
    "'drone_video':int(DV_AGG[(DV_AGG['stage']=='S0_rgb') & (DV_AGG['size']=='all')]['n_frames'].iloc[0])})\n"
    "step3 = step3[['dataset','model','n','stage','P','R','F1','kind']]\n"
    "style_cmp(step3, higher_better=['P','R','F1'], group='dataset')"
)


# ── 5. Step 4 (cell 16) styling + n column ──────────────────────────
nb.cells[16].source = nb.cells[16].source.replace(
    'display(step4.round(4))',
    "step4['n'] = step4['dataset'].map({'antiuav':1000,'svanstrom':1000,"
    "'drone_video':int(DV_AGG[(DV_AGG['stage']=='S0_rgb') & (DV_AGG['size']=='all')]['n_frames'].iloc[0])})\n"
    "step4 = step4[['dataset','n','stage','F1','P','R']]\n"
    "style_cmp(step4, higher_better=['P','R','F1'], group='dataset')"
)


# ── 6. Step 8 (cell 31) styling ─────────────────────────────────────
nb.cells[31].source = nb.cells[31].source.replace(
    "display(all_rgb[['stage', 'label', 'TP', 'FP', 'FN', 'P', 'R', 'F1']].round(4))",
    "all_rgb['n'] = all_rgb['stage'].map(RGB_AGG.set_index('stage')['n_frames'].to_dict()).fillna(0).astype(int)\n"
    "style_cmp(all_rgb[['stage','label','n','TP','FP','FN','P','R','F1']].copy(), "
    "higher_better=['P','R','F1','TP'], lower_better=['FP','FN'])"
)


# ── 7. Step 8 confuser FR (cell 34) styling — lower is better ───────
nb.cells[34].source = '''# Step 8 — per-confuser-category FR% across pipeline (BIRD / AIRPLANE / HELICOPTER only)
# Note: rgb_test only has bird-filename frames in the confuser CSV (no airplane/helicopter file matches).
CONF_CATS = ['bird', 'airplane', 'helicopter']
sub_conf = RGB_CONF[RGB_CONF['category'].isin(CONF_CATS)].copy()
piv = (sub_conf.pivot(index='category', columns='stage', values='fr_seg_pct')
        .reindex(columns=['S0_rgb','S2_rgb_patch','S0_ir_grayscale','S2_ir_patch',
                          'S4_clf','S4_clf_patch','S4_clf_other_mode']).round(2))
piv.style.background_gradient(cmap=_RYG_BAD, axis=None, vmin=0, vmax=100).format('{:.2f}', na_rep='—')
'''


# ── 7b. Step 7 per-clip tables (cells 24, 27) styling + n column ────
nb.cells[24].source = nb.cells[24].source.rstrip().rsplit('\n', 1)[0] + (
    "\n# Attach n (frames per clip) from per_clip CSV\n"
    "_n = per_clip.drop_duplicates('clip')[['clip','frames']].rename(columns={'frames':'n'})\n"
    "p = p.merge(_n, on='clip', how='left')\n"
    "p = p[['clip','n','S0_rgb','S0_ir_grayscale','S4_clf_softveto','delta']]\n"
    "style_cmp(p, higher_better=['S0_rgb','S0_ir_grayscale','S4_clf_softveto','delta'])"
)
nb.cells[27].source = nb.cells[27].source.rstrip().rsplit('\n', 1)[0] + (
    "\n_n2 = conf_clip.drop_duplicates(['category','clip'])[['category','clip','frames']].rename(columns={'frames':'n'})\n"
    "p_conf = p_conf.merge(_n2, on=['category','clip'], how='left')\n"
    "p_conf = p_conf[['category','clip','n','S0_rgb','S0_ir_grayscale','S4_softveto_patch','delta']]\n"
    "style_cmp(p_conf, lower_better=['S0_rgb','S0_ir_grayscale','S4_softveto_patch','delta'])"
)


# ── 7c. Step 8 confuser plot (cell 35): replace flat-zero plot ──────
nb.cells[35].source = '''# Step 8 — rgb_test bird confusers: base detector fires <1% of frames, segment FR = 0 across
# the entire pipeline. The flat-zero plot is uninformative; show a compact table instead.
sub = RGB_CONF[RGB_CONF['category'].isin(['bird','airplane','helicopter'])].copy()
n_frames_by_cat = sub.groupby('category')['n_frames'].first().to_dict()
print('rgb_test confuser frames per category:', n_frames_by_cat)
print('(Only bird-filenames present — airplane/helicopter substrings absent from this split.)\\n')
piv_fr = (sub.pivot(index='category', columns='stage', values='fr_frame_pct')
            .reindex(columns=['S0_rgb','S0_ir','S2_rgb_patch','S2_ir_patch',
                              'S4_clf','S4_clf_patch','S4_clf_other_mode']).round(2))
print('Frame-level FR% (lower = better):')
display(piv_fr.style.background_gradient(cmap=_RYG_BAD, axis=None, vmin=0, vmax=10)
            .format('{:.2f}', na_rep='—'))
print('Segment-level FR% is 0.00 for every stage — selcom@960 never triggers a 2/3 segment vote on these frames.')
'''


# ── 8. Step 9 (cell 37) styling ─────────────────────────────────────
nb.cells[37].source = nb.cells[37].source.replace(
    "display(all_ir[['stage', 'label', 'TP', 'FP', 'FN', 'P', 'R', 'F1']].round(4))",
    "all_ir['n'] = all_ir['stage'].map(IR_AGG.set_index('stage')['n_frames'].to_dict()).fillna(0).astype(int)\n"
    "style_cmp(all_ir[['stage','label','n','TP','FP','FN','P','R','F1']].copy(), "
    "higher_better=['P','R','F1','TP'], lower_better=['FP','FN'])"
).replace(
    "display(all_ir.round(4))",
    "style_cmp(all_ir, higher_better=['P','R','F1','TP'], lower_better=['FP','FN'])"
)


# ── 9. New aggregate section appended at end ────────────────────────
nb.cells.append(md('''---
## Aggregate across all datasets

Pipeline stages rolled up across every evaluated dataset, **with IR native and IR grayscale tracked separately** (real IR sensor vs cross-modal grayscale-RGB fallback — very different inputs).

- `base_ir_native` = ir_v3b on real IR frames (antiuav, svanstrom, ir_test)
- `base_ir_grayscale` = ir_v3b run on grayscale-RGB (drone_video, rgb_test)
- `+classifier` = **production deployment per dataset**: argmax for paired/IR-primary (antiuav, svanstrom, ir_test) + softveto τ=0.95 for RGB-only/grayscale (drone_video, rgb_test). This is the rule the system actually ships.

The `n_ds` column shows how many datasets contributed to each stage's row — useful for spotting apples-to-oranges comparisons.

**Read:** one row per architecture/stage, columns are the global P/R/F1 and frame-level FP%/TN%. Color: red→green per column.
'''))

nb.cells.append(code('''# Aggregate: drone P/R/F1 + frame-level FP%/TN% across ALL datasets per stage
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

# 1) Pull TP/FP/FN per stage per dataset
agg_rows = []

# antiuav / svanstrom — use per-dataset CSV via per_size_detail.csv (or rebuild from detection)
def _det_tpfn(model, ds):
    df = pd.read_csv(DET / f'{model}_{ds}_detection.csv')
    r = df[df['size']=='all'].iloc[0]
    return int(r['TP']), int(r['FP']), int(r['FN'])

def _fl_fp_tn(model, ds):
    df = pd.read_csv(DET / f'{model}_{ds}_frame_level.csv')
    r = df[df['size']=='all'].iloc[0]
    fp_frames = int(r['FP']); tn_frames = int(r['TN'])
    return fp_frames, tn_frames, fp_frames + tn_frames

# antiuav & svanstrom: argmax classifier; drone_video & rgb_test: soft-veto; ir_test: argmax IR.
# Aggregate stages we care about: base_rgb, base_ir, +temporal, +alert_gate, +classifier(final)
DS_N_ALL = {'antiuav': 1000, 'svanstrom': 1000,
             'drone_video': int(DV_AGG[(DV_AGG['stage']=='S0_rgb') & (DV_AGG['size']=='all')]['n_frames'].iloc[0]),
             'rgb_test': 796, 'ir_test': 690}
def _add(stage, ds, tp, fp, fn, fp_frames=None, tn_frames=None, n_neg=None):
    agg_rows.append({'stage': stage, 'dataset': ds, 'n': DS_N_ALL.get(ds, 0),
                      'TP': tp, 'FP': fp, 'FN': fn,
                      'FP_frames': fp_frames, 'TN_frames': tn_frames, 'n_neg': n_neg})

# Anti-UAV / Svanstrom — IR is NATIVE (real IR sensor)
for ds in ('antiuav', 'svanstrom'):
    tp,fp,fn = _det_tpfn(RGB, ds); fpf,tnf,nn = _fl_fp_tn(RGB, ds)
    _add('base_rgb', ds, tp, fp, fn, fpf, tnf, nn)
    tp,fp,fn = _det_tpfn(IR, ds); fpf,tnf,nn = _fl_fp_tn(IR, ds)
    _add('base_ir_native', ds, tp, fp, fn, fpf, tnf, nn)
    # +temporal (RGB), +alert_gate (RGB), classifier (production = argmax for paired)
    for stage_name, prefix in [('+temporal', 'temporal'),
                                ('+alert_gate', 'alert_gate'),
                                ('+classifier', 'classifier_sa32')]:
        try:
            f = DET / (f'{prefix}_{RGB}_{ds}.csv' if prefix != 'classifier_sa32'
                       else f'classifier_sa32_{ds}.csv')
            r = pd.read_csv(f).iloc[0]
            _add(stage_name, ds, int(r['TP']), int(r['FP']), int(r['FN']))
        except FileNotFoundError:
            pass

# drone_video — IR is GRAYSCALE (cross-modal fallback). Production = softveto.
for stage_key, stage_name in [('S0_rgb','base_rgb'),
                                ('S0_ir_grayscale','base_ir_grayscale'),
                                ('temporal__S0_rgb','+temporal'),
                                ('alert_gate__S2_rgb_patch','+alert_gate'),
                                ('S4_clf_softveto','+classifier')]:
    sub = DV_AGG[(DV_AGG['stage']==stage_key) & (DV_AGG['size']=='all')]
    if sub.empty:
        sub = DV_AGG[DV_AGG['stage']==stage_key]
    if not sub.empty:
        r = sub.iloc[0]
        fpf = r.get('frame_FP'); tnf = r.get('frame_TN'); nf = r.get('n_frames')
        n_neg = (fpf + tnf) if (pd.notna(fpf) and pd.notna(tnf)) else None
        _add(stage_name, 'drone_video', int(r['TP']), int(r['FP']), int(r['FN']),
             int(fpf) if pd.notna(fpf) else None,
             int(tnf) if pd.notna(tnf) else None,
             int(n_neg) if n_neg is not None else None)

# rgb_test — IR is GRAYSCALE. Production classifier = softveto (the S4_clf rows in this CSV are softveto).
RGB_AGG_FULL = pd.read_csv(FPA / 'eval_rgb_test_aggregate.csv')
for stage_key, stage_name in [('S0_rgb','base_rgb'),
                                ('S0_ir','base_ir_grayscale'),
                                ('S4_clf','+classifier'),
                                ('S4_clf_patch','+alert_gate')]:
    sub = RGB_AGG_FULL[(RGB_AGG_FULL['stage']==stage_key) & (RGB_AGG_FULL['size']=='all')]
    if not sub.empty:
        r = sub.iloc[0]
        _add(stage_name, 'rgb_test', int(r['TP']), int(r['FP']), int(r['FN']))

# ir_test — IR is NATIVE. Production classifier = argmax (the S4_clf row is argmax here).
IR_AGG_FULL = pd.read_csv(FPA / 'eval_ir_test_aggregate.csv')
for stage_key, stage_name in [('S0_ir','base_ir_native'),
                                ('S0_rgb','base_rgb'),
                                ('S4_clf','+classifier'),
                                ('S4_clf_patch','+alert_gate')]:
    sub = IR_AGG_FULL[(IR_AGG_FULL['stage']==stage_key) & (IR_AGG_FULL['size']=='all')]
    if not sub.empty:
        r = sub.iloc[0]
        _add(stage_name, 'ir_test', int(r['TP']), int(r['FP']), int(r['FN']))

agg = pd.DataFrame(agg_rows)
# Roll up per stage across all datasets
roll = agg.groupby('stage').agg(n=('n','sum'),
                                  TP=('TP','sum'), FP=('FP','sum'), FN=('FN','sum'),
                                  FP_frames=('FP_frames','sum'), TN_frames=('TN_frames','sum'),
                                  n_neg=('n_neg','sum')).reset_index()
roll['P']  = roll.apply(lambda r: _safe_div(r.TP, r.TP+r.FP), axis=1)
roll['R']  = roll.apply(lambda r: _safe_div(r.TP, r.TP+r.FN), axis=1)
roll['F1'] = roll.apply(lambda r: _safe_div(2*r.P*r.R, r.P+r.R), axis=1)
roll['FP%'] = roll.apply(lambda r: 100*_safe_div(r.FP_frames, r.n_neg) if r.n_neg else None, axis=1)
roll['TN%'] = roll.apply(lambda r: 100*_safe_div(r.TN_frames, r.n_neg) if r.n_neg else None, axis=1)
# Order stages logically
order = ['base_rgb','base_ir_native','base_ir_grayscale','+temporal','+classifier','+alert_gate']
roll['_o'] = roll['stage'].map({s:i for i,s in enumerate(order)})
roll = roll.sort_values('_o').drop(columns='_o').reset_index(drop=True)

print('Drone — all datasets aggregated (n_datasets per stage shown in n_ds)')
roll_ds_counts = agg.groupby('stage')['dataset'].nunique().rename('n_ds').reset_index()
roll = roll.merge(roll_ds_counts, on='stage')
style_cmp(roll[['stage','n_ds','n','TP','FP','FN','P','R','F1','FP%','TN%']].copy(),
           higher_better=['P','R','F1','TN%','TP'],
           lower_better=['FP','FN','FP%'])
'''))

nb.cells.append(md('''### Aggregate per size — drone (across paired datasets that have per-size CSVs)

`drone_video` per-size also included where available (small/medium/large) for stage S0 base RGB and S0 base IR-gray.
'''))

nb.cells.append(code('''# Aggregate per-size for drones, IR native vs grayscale split
rows = []
# Anti-UAV + Svanstrom: IR is NATIVE
for ds in ('antiuav', 'svanstrom'):
    for model, mod_label in [(RGB, 'base_rgb'), (IR, 'base_ir_native')]:
        df = pd.read_csv(DET / f'{model}_{ds}_detection.csv')
        for size in ('small','medium','large'):
            r = df[df['size']==size]
            if r.empty: continue
            r = r.iloc[0]
            rows.append({'stage': mod_label, 'size': size,
                          'TP': int(r['TP']), 'FP': int(r['FP']), 'FN': int(r['FN'])})
# drone_video: IR is GRAYSCALE
for stage_key, stage_label in [('S0_rgb','base_rgb'),('S0_ir_grayscale','base_ir_grayscale')]:
    sub = DV_AGG[(DV_AGG['stage']==stage_key) & (DV_AGG['size'].isin(['small','medium','large']))]
    for _, r in sub.iterrows():
        rows.append({'stage': stage_label, 'size': r['size'],
                      'TP': int(r['TP']), 'FP': int(r['FP']), 'FN': int(r['FN'])})

ps = pd.DataFrame(rows).groupby(['stage','size']).sum().reset_index()
ps['n_gt'] = ps['TP'] + ps['FN']
ps['P'] = ps.apply(lambda r: r.TP/(r.TP+r.FP) if (r.TP+r.FP) else 0.0, axis=1)
ps['R'] = ps.apply(lambda r: r.TP/(r.TP+r.FN) if (r.TP+r.FN) else 0.0, axis=1)
ps['F1'] = ps.apply(lambda r: 2*r.P*r.R/(r.P+r.R) if (r.P+r.R) else 0.0, axis=1)
ps['size'] = pd.Categorical(ps['size'], ['small','medium','large'], ordered=True)
ps = ps.sort_values(['stage','size']).reset_index(drop=True)
ps = ps[['stage','size','n_gt','TP','FP','FN','P','R','F1']]
style_cmp(ps, higher_better=['P','R','F1','TP'], lower_better=['FP','FN'], group='stage')
'''))

nb.cells.append(md('''### Aggregate confuser FR% across all datasets, per category — split by IR type

Lower FR% = better. **Two tables** — one for datasets where IR is native (ir_test) and one for grayscale-fallback IR (drone_video + rgb_test). The `+classifier` column is **production deployment**: argmax for ir_test, softveto τ=0.95 for drone_video / rgb_test.

This is why the previous mixed table looked weird — it summed apples (rgb_test 0% FR on birds) into oranges (drone_video 21% FR on birds against attacking flocks). Splitting by IR type also separates the deployment rules, so each cell is a fair rollup.
'''))

nb.cells.append(code('''# Aggregate confuser FR% — production stages only, split IR-native vs IR-grayscale
def _conf_long_v2(df, ir_type):
    """Returns one row per (cat, common_stage) with fp_boxes/n_frames and ir_type label."""
    sub = df.copy()
    sub['cat'] = sub['category'].replace({'birds':'bird','airplanes':'airplane','helicopters':'helicopter'})
    sub = sub[sub['cat'].isin(['bird','airplane','helicopter'])]
    sub['ir_type'] = ir_type
    return sub

dv    = pd.read_csv(FPA / 'eval_drone_video_confuser_aggregate.csv')
rgb_c = pd.read_csv(FPA / 'eval_rgb_test_confuser.csv')
ir_c  = pd.read_csv(FPA / 'eval_ir_test_confuser.csv')

# Per dataset, the meaningful pipeline stages (production)
# drone_video (softveto): S0_rgb, S0_ir_grayscale, S4_clf_softveto, S4_softveto_patch
# rgb_test (softveto):    S0_rgb, S0_ir,           S4_clf,           S4_clf_patch
# ir_test (argmax):       S0_rgb, S0_ir,           S4_clf,           S4_clf_patch

def _label_stages(df, mapping):
    """Map per-CSV stage names to canonical pipeline labels."""
    df = df.copy()
    df['stage_canon'] = df['stage'].map(mapping)
    return df[df['stage_canon'].notna()]

GRAY_MAP = {
    'S0_rgb': 'base_rgb', 'S0_ir_grayscale': 'base_ir', 'S0_ir': 'base_ir',
    'S4_clf_softveto': '+classifier', 'S4_clf': '+classifier',
    'S4_softveto_patch': '+alert_gate', 'S4_clf_patch': '+alert_gate',
}
NATIVE_MAP = {
    'S0_rgb': 'base_rgb', 'S0_ir': 'base_ir',
    'S4_clf': '+classifier', 'S4_clf_patch': '+alert_gate',
}

# Grayscale-IR datasets: drone_video + rgb_test
gray = pd.concat([
    _conf_long_v2(_label_stages(dv, GRAY_MAP), 'grayscale'),
    _conf_long_v2(_label_stages(rgb_c, GRAY_MAP), 'grayscale'),
], ignore_index=True)
roll_gray = (gray.groupby(['cat','stage_canon']).agg(fp_boxes=('fp_boxes','sum'),
                                                       n_frames=('n_frames','sum')).reset_index())
roll_gray['FR%'] = 100 * roll_gray['fp_boxes'] / roll_gray['n_frames'].replace(0, np.nan)
piv_gray = (roll_gray.pivot(index='cat', columns='stage_canon', values='FR%')
            .reindex(columns=['base_rgb','base_ir','+classifier','+alert_gate'])
            .reindex(index=['bird','airplane','helicopter']).round(2))
n_gray = gray.groupby('cat')['n_frames'].first().to_dict()
piv_gray.insert(0, 'n', piv_gray.index.map(n_gray).fillna(0).astype('Int64'))

# Native-IR dataset: ir_test
native = _conf_long_v2(_label_stages(ir_c, NATIVE_MAP), 'native')
roll_nat = (native.groupby(['cat','stage_canon']).agg(fp_boxes=('fp_boxes','sum'),
                                                       n_frames=('n_frames','sum')).reset_index())
roll_nat['FR%'] = 100 * roll_nat['fp_boxes'] / roll_nat['n_frames'].replace(0, np.nan)
piv_nat = (roll_nat.pivot(index='cat', columns='stage_canon', values='FR%')
            .reindex(columns=['base_rgb','base_ir','+classifier','+alert_gate'])
            .reindex(index=['bird','airplane','helicopter']).round(2))
n_nat = native.groupby('cat')['n_frames'].first().to_dict()
piv_nat.insert(0, 'n', piv_nat.index.map(n_nat).fillna(0).astype('Int64'))

print('IR-native confusers (ir_test — production = argmax classifier)')
display(piv_nat.style.background_gradient(cmap=_RYG_BAD, axis=None, vmin=0, vmax=100).format('{:.2f}', na_rep='—'))
print()
print('IR-grayscale confusers (drone_video + rgb_test — production = softveto τ=0.95)')
display(piv_gray.style.background_gradient(cmap=_RYG_BAD, axis=None, vmin=0, vmax=100).format('{:.2f}', na_rep='—'))
'''))

nbformat.write(nb, NB)
print(f'Patched → {NB}  (now {len(nb.cells)} cells)')
