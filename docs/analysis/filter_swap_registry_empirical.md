# Confuser-Filter Swap Registry — `empirical.tex`

**Scope:** every number, figure, table, and qualitative claim in
`docs/thesis_working_distilling_overleaf/chapters/empirical.tex` that depends on the two
**confuser filters** (detections-removed-only verifiers; never raise recall):

- **RGB filter** = `mlp_v5`, P(drone) ≥ 0.25 (cells: `filt_mlp`, `filt_mlp_rgb`, `filt (mlp_v5,RGB)`).
- **IR filter** = `mlp_v5_ir_aligned`, one net + two scalers:
  - **thermal** scaler `aligned` / `mlp_aligned`, thr ≥ 0.05 (cells: `filt_mlp_ir`, `filt (aligned,IR)`, `filt (aligned, thermal)`).
  - **grayscale** scaler `aligned_gray` / `mlp_aligned_gray`, thr ≥ 0.25 (cells: `filt (aligned_gray,IR)`).
- Composed cells `clf->filt[...]` and `filt->clf[...]` **also embed a filter** (so they are IN scope);
  the bare `clf[...]` / `clf only` rows and **patch** cells (`filt_patch`, `filt (patch)`) are NOT these
  filters but are listed where they share a table so a swap-diff is complete.

**Audit backbone:** `thesis_eval/_audit_headline_numbers.py`. Its `CHECKS` list hard-codes the claimed
value of every audited cell beside a JSON lookup. JSON sources:
- `thesis_eval/results/tier1_results.json` (`T`), `…/temporal_results.json` (`V`), `…/video_thr_sweep.json` (`W`), `…/notes_round1_results.json` (`N`), `…/failure_profile_results.json` (`F`), `…/negative_frame_fire.json` (`G`), `…/conf_sweep/conf_sweep_results.json` (`S`)
- `thesis_eval/results_noreject/{tier1,temporal,notes_round1}_results.json` (`TN`/`VN`/`NN`) — shipped `robust8-nr`
- `runs/results_dut/tier1_results.json` (`D`/frozen DUT), `runs/clean_split/clean_split_results.json` (`C`), `eval/results/filter_operating_sweep.json` (`FS`)

> **Cell-key legend (JSON):** the shipped no-reject router `robust8-nr` is keyed `robust8_nr_drop`.
> `filt_mlp` = RGB+IR filter both-on; `filt_mlp_rgb` / `filt_mlp_ir` = RGB-only / IR-only filter rows;
> `filt_patch` = patch CNN (not in scope as a swap target, but co-located).

---

## (1) NUMBERS TABLE

`metric` codes: R=recall, P=precision, F1, FP=false-positive count, fire=frame-fire-rate, thr=threshold, %=percentage/delta, Δ=delta. `filter` column: RGB=`mlp_v5`; IR-th=`aligned` thermal scaler; IR-gray=`aligned_gray` grayscale scaler; (mixed)=composed cell touching both. Lines are `empirical.tex` line numbers.

### Table `tab:ablation_svanstrom` (svanstrom paired, `T["svanstrom"]["B_pipeline"]`)

| line | value | metric | surface | filter | table/inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 31 | 3043 | TP | svanstrom | RGB | `tab:ablation_svanstrom` filt only (mlp_v5,RGB) | T svanstrom B_pipeline `filt_mlp_rgb` TP | UN-AUDITED |
| 31 | 353 | FP | svanstrom | RGB | filt only (mlp_v5,RGB) | T … `filt_mlp_rgb` FP | UN-AUDITED |
| 31 | 271 | FN | svanstrom | RGB | filt only (mlp_v5,RGB) | T … `filt_mlp_rgb` FN | UN-AUDITED |
| 31 | 0.896 | P | svanstrom | RGB | filt only (mlp_v5,RGB) | T … `filt_mlp_rgb` precision | UN-AUDITED |
| 31 | 0.918 | R | svanstrom | RGB | filt only (mlp_v5,RGB) | T … `filt_mlp_rgb` recall | UN-AUDITED |
| 31 | 0.907 | F1 | svanstrom | RGB | filt only (mlp_v5,RGB) | T … `filt_mlp_rgb` f1 | **`svan filt_mlp_rgb F1`** |
| 32 | 3142 | TP | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` TP | UN-AUDITED |
| 32 | 2018 | FP | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` FP | UN-AUDITED |
| 32 | 172 | FN | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` FN | UN-AUDITED |
| 32 | 0.609 | P | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` precision | UN-AUDITED |
| 32 | 0.948 | R | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` recall | UN-AUDITED |
| 32 | 0.742 | F1 | svanstrom | IR-th | filt only (aligned,IR) | T … `filt_mlp_ir` f1 | **`svan filt_mlp_ir F1`** |
| 38 | 3037 / 330 / 126 / 0.902 / 0.960 / 0.930 | TP/FP/FN/P/R/F1 | svanstrom | mixed | clf→filt [robust8-nr] | TN svanstrom B_pipeline `clf->filt[robust8_nr_drop]` | F1 **`NR svan composed F1`**; R **`NR svan composed R`**; P **`NR svan composed P`**; TP/FP/FN UN-AUDITED |
| 39 | 2993 / 195 / 130 / 0.939 / 0.958 / 0.949 | TP/FP/FN/P/R/F1 | svanstrom | mixed | clf→filt [robust8] | T svanstrom B_pipeline `clf->filt[robust8]` | F1 **`svan composed F1`**; P **`svan composed P`**; R **`svan composed R`**; TP/FP/FN UN-AUDITED |
| 40 | 3039 / 162 / 128 / 0.949 / 0.960 / 0.955 | TP/FP/FN/P/R/F1 | svanstrom | mixed | clf→filt [sa32] | T … `clf->filt[sa32]` | UN-AUDITED |
| 41 | 3037 / 332 / 29 / 0.901 / 0.991 / 0.944 | TP/FP/FN/P/R/F1 | svanstrom | mixed | **filt→clf [robust8-nr]** (SHIPPED) | TN svanstrom B_pipeline `filt->clf[robust8_nr_drop]` | F1 **`NR svan filt->clf F1`**; P/R/TP/FP/FN UN-AUDITED |
| 42 | 2996 / 196 / 35 / 0.939 / 0.989 / 0.963 | TP/FP/FN/P/R/F1 | svanstrom | mixed | filt→clf [robust8] | T svanstrom B_pipeline `filt->clf[robust8]` | F1 **`svan filt->clf F1`**; rest UN-AUDITED |

### Table `tab:ablation_antiuav` (antiuav paired, `T["antiuav"]["B_pipeline"]`)

| line | value | metric | surface | filter | table/inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 60 | 7430 / 172 / 243 / 0.977 / 0.968 / 0.973 | TP/FP/FN/P/R/F1 | antiuav | RGB | filt only (mlp_v5,RGB) | T antiuav B_pipeline `filt_mlp_rgb` | F1 **`antiuav filt_mlp_rgb F1`**; rest UN-AUDITED |
| 61 | 7432 / 173 / 241 / 0.977 / 0.969 / 0.973 | TP/FP/FN/P/R/F1 | antiuav | IR-th | filt only (aligned,IR) | T antiuav B_pipeline `filt_mlp_ir` | F1 **`antiuav filt_mlp_ir F1`**; rest UN-AUDITED |
| 67 | 7428 / 157 / 83 / 0.979 / 0.989 / 0.984 | TP/FP/FN/P/R/F1 | antiuav | mixed | clf→filt [robust8-nr] | TN antiuav B_pipeline `clf->filt[robust8_nr_drop]` | F1 **`NR antiuav composed F1`**; rest UN-AUDITED |
| 68 | 7406 / 139 / 95 / 0.982 / 0.987 / 0.984 | TP/FP/FN/P/R/F1 | antiuav | mixed | clf→filt [robust8] | T antiuav B_pipeline `clf->filt[robust8]` | F1 **`antiuav composed F1`**; rest UN-AUDITED |
| 69 | 7424 / 134 / 77 / 0.982 / 0.990 / 0.986 | TP/FP/FN/P/R/F1 | antiuav | mixed | clf→filt [sa32] | T … `clf->filt[sa32]` | UN-AUDITED |
| 70 | 7428 / 157 / 81 / 0.979 / 0.989 / 0.984 | TP/FP/FN/P/R/F1 | antiuav | mixed | **filt→clf [robust8-nr]** (SHIPPED) | TN antiuav B_pipeline `filt->clf[robust8_nr_drop]` | UN-AUDITED (only `clf->filt` nr antiuav is checked) |
| 71 | 7406 / 139 / 93 / 0.982 / 0.988 / 0.985 | TP/FP/FN/P/R/F1 | antiuav | mixed | filt→clf [robust8] | T antiuav B_pipeline `filt->clf[robust8]` | UN-AUDITED |

### Prose around lines 79–87 (paired readings — filter-dependent numbers)

| line | value | metric | surface | filter | inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 79 | 1,845 | FP (RGB-only bare) | svanstrom | RGB-only baseline removed by filter | inline "removed by the mlp_v5 filter" | T svanstrom A_bare `ft4/rgb` FP | UN-AUDITED (RGB-only FP not in CHECKS; antiuav ft4 FP=41 is) |
| 81 | 2019 → 1997 | FP | svanstrom | mixed (nr router) | inline (no-reject removes almost none) | TN svanstrom B_pipeline `clf[robust8_nr_drop]` FP | UN-AUDITED |
| 81 | 2019 → 353 (alone), → 330 (composed) | FP | svanstrom | RGB | inline "the mlp_v5 filter does that work" | T `filt_mlp_rgb` FP=353; TN `clf->filt[robust8_nr_drop]` FP=330 | UN-AUDITED |
| 81 | 2019 → 354 | FP | svanstrom | IR/router context (robust8 clf-only) | inline | T svanstrom B_pipeline `clf[robust8]` FP | UN-AUDITED |
| 81 | 195 | FP | svanstrom | mixed | inline "composing robust8 with the filter reaches 195 FP" | T `clf->filt[robust8]` FP | UN-AUDITED |
| 83 | 0.991 / 0.960 | R | svanstrom | mixed | inline filt→clf vs clf→filt recall | TN `filt->clf[robust8_nr_drop]`.recall / `clf->filt[robust8_nr_drop]`.recall | 0.991 via NR composed-R chain UN-AUDITED for filt→clf; 0.960 = **`NR svan composed R`** |
| 83 | 0.944 vs 0.930 | F1 | svanstrom | mixed | inline filt→clf vs clf→filt | TN `filt->clf` / `clf->filt` [robust8_nr_drop] f1 | **`NR svan filt->clf F1`** / **`NR svan composed F1`** |
| 83 | 0.837 vs 0.792 | F1 | dut_antiuav_960 | mixed | inline DUT filt→clf vs clf→filt | TN dut `filt->clf` / `clf->filt`[robust8_nr_drop] | **`NR dut filt->clf F1`** / **`NR dut composed F1`** |
| 85 | 0.963 vs 0.944 | F1 | svanstrom | mixed | inline robust8 vs nr at filt→clf | T `filt->clf[robust8]`=0.963 / TN `filt->clf[robust8_nr_drop]`=0.944 | **`svan filt->clf F1`** / **`NR svan filt->clf F1`** |
| 85 | 0.834 vs 0.733 | F1 (mean composed) | drone surfaces (mean) | mixed | inline composed router+filter mean | derived mean over TN composed cells; no single JSON key | UN-AUDITED (aggregate) |

### Table `tab:ablation_dut` (`runs/results_dut` frozen `D`; nr from `TN["dut_antiuav_960"]`)

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 120 | 2820 / 455 / 1670 / 0.861 / 0.628 / 0.726 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | RGB | filt only (mlp_v5,RGB) | D B_pipeline `filt_mlp_rgb` | F1 **`dut filt_mlp_rgb F1`**; rest UN-AUDITED |
| 121 | 2796 / 423 / 1694 / 0.869 / 0.623 / 0.725 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | IR-gray | filt only (aligned_gray,IR) | D B_pipeline `filt_mlp_ir` | F1 **`dut filt_mlp_ir F1`**; rest UN-AUDITED |
| 127 | 2580 / 389 / 966 / 0.869 / 0.728 / 0.792 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | mixed | clf→filt [robust8-nr] | TN dut B_pipeline `clf->filt[robust8_nr_drop]` | F1 **`NR dut composed F1`**; R **`NR dut composed R`**; rest UN-AUDITED |
| 128 | 2234 / 253 / 1653 / 0.898 / 0.575 / 0.701 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | mixed | clf→filt [robust8] | D B_pipeline `clf->filt[robust8]` | P **`DUT clf->filt[robust8] P`** (0.898); F1/R UN-AUDITED |
| 129 | 2265 / 361 / 1866 / 0.863 / 0.548 / 0.670 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | mixed | clf→filt [sa32] | D B_pipeline `clf->filt[sa32]` | UN-AUDITED |
| 130 | 2580 / 389 / 614 / 0.869 / 0.808 / 0.837 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | mixed | **filt→clf [robust8-nr]** (SHIPPED) | TN dut B_pipeline `filt->clf[robust8_nr_drop]` | F1 **`NR dut filt->clf F1`**; rest UN-AUDITED |
| 131 | 2182 / 243 / 1564 / 0.900 / 0.583 / 0.707 | TP/FP/FN/P/R/F1 | dut_antiuav_960 | mixed | filt→clf [robust8] | D B_pipeline `filt->clf[robust8]` | P **`DUT filt->clf[robust8] P`** (0.900); F1/R UN-AUDITED |
| 137 | 0.726 | F1 | dut_antiuav_960 | RGB | inline "filt only (mlp) is weakest cell at 0.726" | D `filt_mlp_rgb` f1 | **`dut filt_mlp_rgb F1`** |
| 137 | 0.837 / 0.807 / 0.707 / 0.583 | F1/R | dut_antiuav_960 | mixed | inline shipped vs robust8 recovery | TN filt→clf nr / D filt→clf robust8 | **`NR dut filt->clf F1`** + UN-AUDITED |
| 140 | 0.575 → 0.728 (R), 0.701 → 0.792 (F1) | R/F1 | dut_antiuav_960 | mixed | inline reject-drop recovery | D clf→filt[robust8] / TN clf→filt[nr] | **`NR dut composed R`** (0.728) + UN-AUDITED |

### Table `tab:ablation_confusers` (`T`/`TN` `C_confuser`) — fire rates

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 160 | 0.011 | fire | rgb_confuser | RGB | filt only (mlp) | T rgb_confuser C `filt_mlp` fire_rate (0.0106) | **`rgbconf mlp fire`** |
| 160 | 0.237 | fire | ir_confusers | IR-th | filt only (mlp) | T ir_confusers C `filt_mlp` fire_rate | UN-AUDITED (only composed irconf checked) |
| 160 | 0.008 | fire | gray_confuser | IR-gray | filt only (mlp) | T gray_confuser C `filt_mlp` fire_rate (0.0076) | **`grayconf mlp fire`** |
| 162 | 0.011 | fire | rgb_confuser | mixed | **filt→clf [robust8-nr]** (SHIPPED) | TN rgb_confuser C `clf->filt[robust8_nr_drop]` / `filt->clf[…]` | **`NR rgb_conf fire`** + **`NR rgb_conf fire filt->clf`** |
| 162 | 0.237 | fire | ir_confusers | mixed | filt→clf [robust8-nr] | TN ir_confusers C `clf->filt`/`filt->clf`[robust8_nr_drop] | **`NR ir_conf fire`** + **`NR ir_conf fire filt->clf`** |
| 163 | 0.0011 | fire | ir_confusers→(RGB col) | mixed | clf→filt [robust6] | T rgb_confuser C `clf->filt[robust6]` | UN-AUDITED (rgb col); irconf r6 0.1792 = **`irconf composed r6 fire`** |
| 164 | 0.0015 | fire | rgb_confuser | mixed | clf→filt [robust8] | T rgb_confuser C `clf->filt[robust8]` | **`rgbconf composed fire`** |
| 164 | 0.217 | fire | ir_confusers | mixed | clf→filt [robust8] | T ir_confusers C `clf->filt[robust8]` | **`irconf composed r8 fire`** |

### Prose lines 170–174 (confuser observations)

| line | value | metric | surface | filter | inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 170 | 835 → 29 (−96.5%) | FP | rgb_confuser | RGB | "the per-frame mlp_v5 cuts FP detections" | T rgb_confuser C `filt_mlp` FP=29; bare FP=835 | FP=29 **`rgbconf mlp FP`**; 835 UN-AUDITED |
| 170 | 835 → 282 | FP | rgb_confuser | patch (context) | "the patch CNN manages" | T rgb_confuser C `filt_patch` FP | UN-AUDITED |
| 170 | 37–72× faster, ~10× stronger | % | rgb_confuser | RGB vs patch | inline | speed ledger / fire ratio | UN-AUDITED (speed; see tab:speed) |
| 170 | 1.1% (29 of 2,633) | fire | rgb_confuser | mixed (nr) | "shipped robust8-nr sits at 1.1%" | TN rgb_confuser C `clf->filt[robust8_nr_drop]` (0.0106) | **`NR rgb_conf fire`** |
| 170 | 0.15% (4 of 2,633) | fire | rgb_confuser | mixed (robust8) | "robust8 ablation drives to 0.15%" | T rgb_confuser C `clf->filt[robust8]` (FP=4) | **`rgbconf composed FP`** (4) + **`rgbconf composed fire`** |
| 170 | 0.11% / 0.08% | fire | rgb_confuser | mixed | robust6 / sa32 compositions | T rgb_confuser C `clf->filt[robust6]` (0.0011) / `clf[sa32]`? | 0.0011 UN-AUDITED in rgb col; sa32 0.003 = **`rgbconf clf[sa32]`?** see note |
| 170 | 656 → 21 (−96.8%) | FP | gray_confuser | IR-gray | "filter weights transfer to grayscale" | T gray_confuser C `filt_mlp` FP=21 | **`grayconf mlp FP`** (21) + **`grayconf mlp fire`** |
| 170 | 656 → 280 | FP | gray_confuser | IR-th (thermal scaler on gray) | "thermal scaler manages only" | thermal-scaler-on-gray cell | UN-AUDITED |
| 173 | 0.179 | fire | ir_confusers | mixed | "best composition (robust6) cuts 39%" | T ir_confusers C `clf->filt[robust6]` | **`irconf composed r6 fire`** |
| 173 | 0.237 (−19%) | fire | ir_confusers | mixed (nr) | "shipped no-reject cell" | TN ir_confusers C `clf->filt[robust8_nr_drop]` | **`NR ir_conf fire`** |
| 173 | 76% | % | ir_confusers | (airplane share, not filter) | inline | meta | UN-AUDITED |

> **Note (line 170, 0.11%/0.08%):** the thesis attributes 0.08% to "sa32 compositions". In `T` the `clf->filt[sa32]` rgb_confuser cell is the source; the audit only pins `rgbconf clf[sa32] fire`=0.003 (clf-only router), not the composed 0.08%. Treat composed sa32/robust6 rgb-confuser percentages as **UN-AUDITED**.

### Table `tab:ablation_solo` (`T … S4_verifier`) — single-modality filter rows

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 199 | 0.943 / 0.973 / 0.958 | P/R/F1 | ir_dset_final | IR-th | filt (aligned, thermal) | T ir_dset_final S4 `filt_mlp` | F1 **`irtest mlp F1`** (0.9578); P/R UN-AUDITED |
| 204 | 0.976 / 0.691 / 0.809 | P/R/F1 | rgb_dataset_test | RGB | filt (mlp) | T rgb_dataset_test S4 `filt_mlp` | F1 **`rgbtest mlp F1`** (0.8092); P/R UN-AUDITED |
| 209 | 0.950 / 0.451 / 0.612 | P/R/F1 | selcom_val | RGB | filt (mlp) | T selcom_val S4 `filt_mlp` | F1 **`selcom mlp F1`** (0.6115); P/R UN-AUDITED |
| 216 | 0.8 pp / 5.0 pp recall cost | Δ | ir_dset_final | IR-th vs patch | inline reading | derived from `filt_mlp` vs `filt_patch` | UN-AUDITED (deltas) |
| 216 | −11.7 pp F1, 0.899 → 0.691 | Δ/R | rgb_dataset_test | RGB | inline carve-out | T `bare` f1 vs `filt_mlp` | **`rgbtest mlp F1`**/`rgbtest bare F1` (deltas UN-AUDITED) |
| 216 | 22 → 7 FP; 0.858 → 0.950 P | FP/P | selcom_val | RGB | inline | T selcom_val S4 `bare`/`filt_mlp` FP & P | UN-AUDITED (FP counts) |

### Table `tab:per_size` (`N … SZ_per_size`) — per-size +filter recall

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 232 | 0.256 | +filt R (<16px) | rgb_dataset_test | RGB | tab:per_size | N rgb_dataset_test SZ `filt` `<16px` recall | **`SZ rgbtest <16 filt R`** (0.2562) |
| 233 | 0.447 | +filt R (16–32px) | rgb_dataset_test | RGB | tab:per_size | N … `filt` `16-32px` recall | **`SZ rgbtest 16-32 filt R`** (0.4465) |
| 234 | 0.829 | +filt R (32–64px) | rgb_dataset_test | RGB | tab:per_size | N … `filt` `32-64px` recall | UN-AUDITED |
| 235 | 0.951 | +filt R (≥64px) | rgb_dataset_test | RGB | tab:per_size | N … `filt` `>=64px` recall | **`SZ rgbtest >=64 filt R`** (0.9506) |
| 237 | 0.593 | +filt R (<16px) | svanstrom | RGB | tab:per_size svan ft4 | N svanstrom SZ `filt` `<16px` recall | UN-AUDITED |
| 238 | 0.823 | +filt R (16–32px) | svanstrom | RGB | tab:per_size | N svanstrom SZ `filt` `16-32px` recall | UN-AUDITED |
| 239 | 0.918 | +filt R (32–64px) | svanstrom | RGB | tab:per_size | N svanstrom SZ `filt` `32-64px` recall | UN-AUDITED |
| 241 | 0.970 | +filt R (<16px) | svanstrom | IR-th | tab:per_size svan v3b | N svanstrom SZ v3b `filt` `<16px` recall | UN-AUDITED |
| 242 | 0.998 | +filt R (16–32px) | svanstrom | IR-th | tab:per_size | N svanstrom SZ v3b `filt` `16-32px` recall | UN-AUDITED |
| 248 | 0.782 → 0.256, 0.865 → 0.447, 0.956 → 0.951 | R deltas | rgb_dataset_test | RGB | inline localisation | N SZ bare vs filt | bare/filt CHECKS exist for these buckets (see audit) |

### §sec:lowconf_mode — filter × detector-floor sweep (`S` conf_sweep)

| line | value | metric | surface | filter | inline / `tab:lowconf_selcom` | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 282 | 0.531 → 0.707 | P (filter restores) | selcom_val @floor 0.05 | RGB | inline | S selcom_val sweep `filt` P @0.05 | UN-AUDITED (P); F1 part = **`SWEEP selcom filt@0.05`** (0.692) |
| 282 | 0.692 [0.654–0.730] | F1 | selcom_val @0.05 +filt | RGB | inline | S selcom_val `filt` @0.05 f1 | **`SWEEP selcom filt@0.05`** |
| 282 | 0.612 [0.551–0.659] | F1 | selcom_val @0.25 +filt | RGB | inline (filtered default) | T selcom_val S4 `filt_mlp` f1 | **`selcom mlp F1`** |
| 282 | 1,281 → 45 (96.5%); default 29 | FP | rgb_confuser @floor 0.05 | RGB | inline confuser-safe | S rgb_confuser sweep `filt`@0.05 FP | UN-AUDITED (45/1281); default 29 = **`rgbconf mlp FP`** |
| 293 | 0.950 / 0.451 / 0.612 | P/R/F1 | selcom_val @0.25 | RGB | `tab:lowconf_selcom` row 0.25 | S/T selcom `filt`@0.25 | F1 **`selcom mlp F1`** / **`SWEEP selcom bare@0.25`** context |
| 294 | 0.816 / 0.573 / 0.673 | P/R/F1 | selcom_val @0.10 | RGB | `tab:lowconf_selcom` row 0.10 | S selcom `filt`@0.10 | UN-AUDITED |
| 295 | 0.707 / 0.678 / 0.692 | P/R/F1 | selcom_val @0.05 | RGB | `tab:lowconf_selcom` row 0.05 | S selcom `filt`@0.05 | F1 **`SWEEP selcom filt@0.05`**; P/R UN-AUDITED |
| 301 | −12.5 pp | Δ F1 | rgb_dataset_test | RGB | inline (carve-out at every floor) | S rgb_dataset_test sweep | UN-AUDITED |
| 301 | 1,051 → 1,911 FP; ~18–20% absorbed | FP/% | ir_confusers @floor drop | IR-th | inline (thermal floor must stay 0.40) | S ir_confsweep | UN-AUDITED |

### §sec:pipeline_speed — `tab:speed`

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 316 | 1.3–2.1 ms/det | latency | (runtime) | RGB `mlp_v5` | `tab:speed` confuser filter | ledger `v5-ship-per-frame` / `eval/bench_speed.py` | UN-AUDITED (no JSON; ledger only) |
| 316 | 59–112 ms/det | latency | (runtime) | patch (predecessor) | `tab:speed` | ledger | UN-AUDITED |
| 316 | 37–72× | speedup | (runtime) | RGB vs patch | `tab:speed` | derived | UN-AUDITED |
| 322 | 1–4% | % overhead | (runtime) | both filters | inline | ledger | UN-AUDITED |

### §sec:temporal_results — `tab:temporal_production` (`V`/`VN`)

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 339 | 0.954 / 0.541 / 0.690 / 0.226 (−35%) | winP/R/F1/fire | video_drone+confuser | RGB+IR (mlp both) | filt only (mlp) | V video_drone `filt_mlp` window[2]; video_confuser `filt_mlp` window_fire | UN-AUDITED (mlp filt-only window cells not in CHECKS) |
| 345 | 0.954 / 0.537 / 0.687 / 0.225 (−36%) | winP/R/F1/fire | video | mixed (nr) | clf→filt [robust8-nr] | VN video_drone `clf->filt[robust8_nr_drop]` window[2]; VN video_confuser window_fire | F1 **`NR video composed F1`** (0.687); fire **`NR video_conf fire`** (0.2252) |
| 346 | 0.972 / 0.393 / 0.560 / 0.073 (−79%) | winP/R/F1/fire | video | mixed | clf→filt [robust8] | V video_drone `clf->filt[robust8]` window[2]=0.56; video_confuser fire=0.0732 | F1 **`video composed r8 F1`**; fire **`video composed r8 fire`** |
| 347 | 0.998 / 0.443 / 0.614 / 0.057 (−84%) | winP/R/F1/fire | video | mixed | clf→filt [robust6] | V video_drone `clf->filt[robust6]` window[2]; video_confuser fire | UN-AUDITED (composed r6 video F1/fire not in CHECKS) |
| 348 | 0.969 / 0.535 / 0.689 / 0.080 (−77%) | winP/R/F1/fire | video | mixed (patch) | clf→filt patch [sa32] | V video_drone `clf->filt_patch[sa32]` window[2]=0.6891 | F1 **`video replica F1`** (0.6891) |
| 355 | ΔR ≈ ±0.01 | Δ | video | filters | inline temporal voting neutral | V window vs frame | UN-AUDITED |
| 357 | R 0.393 → 0.507 (robust8), 0.443 → 0.577 (robust6) | R | video | mixed (filt thr sweep) | inline thr 0.25→0.01 | W `robust8@0.01`[1]=0.507; `robust6@0.01` | **`sweep r8@0.01 R`** (0.507); robust6 R UN-AUDITED, but **`sweep r6@0.01 F1`** (0.7309) pins F1 |
| 357 | window fire trimmed 7–9% | % | video_confuser | filters | inline near-zero thr | W | UN-AUDITED |
| 357 | [0.01, 0.25) smear | thr | video | filters | inline | W | UN-AUDITED (qualitative) |

### §sec:verifier_results — `tab:distill_verifier` (mlp_v5 vs patch; ledger-sourced)

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 589 | 0.869 | +mlp_v5 F1 | svanstrom (s9) | RGB | `tab:distill_verifier` | ledger `v5-beats-patch`; eval `v5_svan_mlp` (no tier1 key — s9 config) | UN-AUDITED (different config: svan_iop_1280_s9) |
| 589 | 0.037 | mlp_v5 halluc | svanstrom | RGB | `tab:distill_verifier` | ledger | UN-AUDITED |
| 590 | 0.985 / 0.010 | F1 / halluc | antiuav | RGB | `tab:distill_verifier` | ledger | UN-AUDITED |
| 591 | 0.607 / 0.019 | F1 / halluc | selcom | RGB | `tab:distill_verifier` | ledger `v5_selcom`; cf. T selcom `filt_mlp` f1=0.6115 | UN-AUDITED (this table) — close to **`selcom mlp F1`** |
| 592 | 0.792 / 0.010 | F1 / halluc | rgb_dataset | RGB | `tab:distill_verifier` | ledger `v5-rgbds-ceiling`; cf. T rgb_dataset_test `filt_mlp`=0.8092 | UN-AUDITED (this config IoU@640) |
| 593 | 0.008 | confuser halluc (mlp_v5) | confuser-only | RGB | `tab:distill_verifier` | ledger | UN-AUDITED |
| 609 | 0.792 (−11 pp), 0.896 → 0.664 | F1 / R collapse | rgb_dataset | RGB | inline carve-out | ledger `mlp-v5-recall-drop-is-ood-coverage` | UN-AUDITED (note: differs from tab:ablation_solo 0.809/0.691 — different scoring) |
| 609 | centroid dist 16.5 / 11.1 / 15.4 | dist | rgb_dataset | RGB | inline MRI diagnosis | `eval/diagnose_mlp_recall_drop.py` cache | UN-AUDITED |
| 612 | 91–100% recovered @5–10% leak; 0.887 → 0.631 P | R/P | svanstrom/rgb_dataset | RGB (fail-open, not adopted) | inline | ledger `filter-recall-precision-decision` | UN-AUDITED |

### `tab:ir_aligned` (CBAM held-out; ledger-sourced, has a regex audit pin)

| line | value | metric | surface | filter | table | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 634 | 0.786 / 0.917 / 0.846 | P/R/F1 | CBAM | IR-th | `tab:ir_aligned` +aligned MLP | ledger `ir_aligned_cbam_heldout` (no replay JSON) | regex pin in audit (CBAM table FP) |
| 634 | 15 | FP | CBAM | IR-th | `tab:ir_aligned` (held out) | evals.csv canonical = 48→15 | **`CBAM aligned FP (empirical table)`** (regex, =15) |
| 634 | −0.050 / +0.147 | ΔR / ΔF1 | CBAM | IR-th | `tab:ir_aligned` | ledger | UN-AUDITED |
| 634 | 48 | FP (bare IR) | CBAM | IR-th baseline | `tab:ir_aligned` | evals.csv | UN-AUDITED (cross-pinned via methodology regex) |
| 635 | 0.965 / 0.958 / 0.962 (108) | P/R/F1 (FP) | ir_dset_final | IR-th | `tab:ir_aligned` | ledger `ir_aligned_irdset_final` | UN-AUDITED (FP 108 vs tab:ablation_solo basis differs) |
| 636 | 0.909/0.977/0.942 (80) | P/R/F1(FP) | ir_video test | IR-th | `tab:ir_aligned` | ledger | UN-AUDITED |
| 637 | 0.983/0.942/0.962 (68) | P/R/F1(FP) | antiuav test | IR-th | `tab:ir_aligned` | ledger | UN-AUDITED |
| 639 | 0.564/0.883/0.688 (41); cut only 7 FP | P/R/F1(FP) | CBAM | patch (context) | `tab:ir_aligned` | ledger | UN-AUDITED |
| 645 | 96.8% FP cut grayscale | % | gray_confuser | IR-gray | inline | T gray_confuser `filt_mlp` (656→21) | **`grayconf mlp FP`** |

### §sec:verifier_results filter-operating sweep (`FS`) + Figure `fig:filter_operating`

| line | value | metric | surface | filter | inline / fig caption | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 649/655 | 0.956 | drone-det recall @0.25 | (pooled RGB drone surfaces) | RGB | "keeps 95.6%" | FS `RGB mlp_v5` shipped[0] | **`FIG rgb recall@0.25`** (0.956) |
| 649/655 | 0.011 | confuser fire @0.25 | rgb_confuser | RGB | "1.1% confuser fire" | FS `RGB mlp_v5` shipped[1] | **`FIG rgb fire@0.25`** (0.011) |
| 649 | 0.21–0.24 | fire (flat) | ir_confusers | IR-th | "flat in fire at every threshold" | FS aligned thermal | UN-AUDITED (range) |
| 649/655 | 0.467 | drone-det recall @0.25 | gray (fallback) | IR-gray | "retains only 46.7%" | FS `grayscale aligned` shipped[0] | **`FIG gray recall@0.25`** (0.467) |
| 649 | 0.47 → 0.62 (recall), 0.008 → 0.036 (fire) @0.05 | R/fire | gray | IR-gray | inline 0.05 vs 0.25 | FS `grayscale aligned` t0.05 / shipped | **`FIG gray recall@0.05`** (0.618), **`FIG gray fire@0.25`** (0.008), **`FIG gray fire@0.05`** (0.036) |

### §sec:grayscale (filter on grayscale channel)

| line | value | metric | surface | filter | inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 769 | 37.2% vs 1.8% | halluc (per image) | gray vs thermal | (detector, motivates filter) | inline harvesting | mri/results/ir_v3b_report | UN-AUDITED here (MRI 1.8% = **`MRI ir halluc`** elsewhere) |
| 772 | 96.8% (656 → 21) | FP cut | gray_confuser | IR-gray | inline | T gray_confuser `filt_mlp` FP=21 | **`grayconf mlp FP`** (21) + **`grayconf mlp fire`** |
| 772 | recall ≤ 0.27 even @thr 0.02 | R | svanstrom_gray | IR-gray | inline over-veto (GRAY_SWEEP) | T svanstrom_gray S4 + GRAY_SWEEP | UN-AUDITED |
| 772 | 0.580 bare | F1 | svanstrom_gray | (detector bare, unfiltered) | inline | T svanstrom_gray A v3b/ir | **`3way gray F1`** (0.5796) |

### §sec:threats (leakage bound touches the filter/composed cell)

| line | value | metric | surface | filter | inline | harness JSON cell | audit CHECK |
|---|---|---|---|---|---|---|---|
| 786 | 0.984 → 0.986 pipeline | F1 | antiuav clean | mixed | inline clean-split | C antiuav_clean B_pipeline `clf->filt[robust8]` | **`CLEAN auv pipeline`** (0.9862) |
| 786 | robust8 −1.4 pp (0.949 → 0.935); robust8-nr −3.3 pp (0.944 → 0.911) | Δ F1 | svanstrom clean | mixed | inline cascade leakage bound | C svanstrom_clean pipeline (0.9348) vs T 0.9485; nr 0.944→0.911 | **`CLEAN svan pipeline`** (0.9348) pins robust8 clean; nr 0.911 **UN-AUDITED** |

---

## (2) FIGURES / TABLES INDEX

| line | \label | caption (short) | filter cells | image / generating script |
|---|---|---|---|---|
| 19 | `tab:ablation_svanstrom` | Full-pipeline ablation, Svanstrom paired (n=4000) | `filt_mlp_rgb`, `filt_mlp_ir`, `clf->filt[*]`, `filt->clf[*]` (+filt_patch ctx) | `thesis_eval/results/tier1_results.json` + `_noreject/`; run `thesis_eval/pipeline_eval_unified.py` |
| 48 | `tab:ablation_antiuav` | Full-pipeline ablation, Anti-UAV paired (n=4000) | same cell set | `thesis_eval/results/tier1_results.json` + `_noreject/`; `pipeline_eval_unified.py` |
| 89 | `tab:rq3` | RQ3 single modality vs routed (production cell embeds filter) | `filt->clf[robust8_nr_drop]` production row | `thesis_eval/results/tier1_results.json` (A_bare + B_pipeline) |
| 108 | `tab:ablation_dut` | Full-pipeline ablation, DUT test split @960 | `filt_mlp_rgb`, `filt_mlp_ir`(gray), `clf->filt[*]`, `filt->clf[*]` | `runs/results_dut/tier1_results.json` + `thesis_eval/results_noreject/`; `pipeline_eval_unified.py` |
| 147 | `tab:ablation_confusers` | Confuser-surface FP reduction (RGB/IR/gray fire) | `filt_mlp` (mlp), `clf->filt[*]`, `filt->clf[robust8_nr_drop]` (+filt_patch) | `thesis_eval/results/tier1_results.json` + `_noreject/` (C_confuser) |
| 176 | `fig:pipeline_ablation` | Pipeline ablation at a glance (robust8 cascade drawn) | bare→patch→mlp_v5→router→composed | `fig_pipeline_ablation` ; gen `thesis_eval/gen_ablation_figure.py` from `tier1_results.json` |
| 188 | `tab:ablation_solo` | Solo-surface ablation (filt mlp + patch rows) | `filt (aligned,thermal)`, `filt (mlp)` (RGB), patch | `thesis_eval/results/tier1_results.json` (S4_verifier) |
| 224 | `tab:per_size` | Per-size recall bare→+filter | RGB `mlp_v5` filt R; IR-th `aligned` filt R | `thesis_eval/results/notes_round1_results.json` (SZ_per_size); `thesis_eval/notes_round1_replays.py` |
| 255 | `tab:failure_profile` | Background failure profile (bare detectors — NO filter) | none (bare only) | `thesis_eval/results/failure_profile_results.json`; `failure_profile_aggregate.py` |
| 285 | `tab:lowconf_selcom` | Detector-floor sweep × filter on SelCom | `+mlp_v5` P/R/F1 column | `thesis_eval/results/conf_sweep/conf_sweep_results.json`; `conf_sweep_replay.py` |
| 307 | `tab:speed` | Per-stage runtime (filter latency) | confuser filter `mlp_v5` 1.3–2.1 ms/det | ledger `v5-ship-per-frame`; `eval/bench_speed.py` |
| 330 | `tab:temporal_production` | Segment-level (2-of-3) real-video eval | `filt_mlp`, `clf->filt[*]`, `clf->filt_patch[sa32]` | `thesis_eval/results/temporal_results.json` + `_noreject/`; `temporal_replay.py` |
| 364 | `fig:cascade_segment_fig` | Design-evolution: segment F1/FPR predecessor (alert-gated patch) | predecessor patch filter (not mlp_v5) | `fig8_cascade_segment`; predecessor config |
| 381 | `tab:rgb_comparison` | RGB variants on Svanstrom (NO filter) | none | ledger `retrainedv2-recall-collapse` |
| 401 | `tab:selcom` | SelCom CCTV fine-tune (NO filter) | none | `runs/rgb_finetune_eval/.../comparison.json` |
| 430 | `tab:ir_evolution` | IR detector versions (NO filter) | none | `eval/results/ir_version_comparison/...csv` |
| 449 | `fig:ir_evolution` | IR P/R trajectory (NO filter) | none | `fig4_ir_evolution` |
| 472 | `tab:robust6_pipeline` | robust6 vs sa32 under cascade (predecessor; "filter only" row) | predecessor patch filter; `filter->classifier[robust6]` | `eval/results/_overnight_ablation/ablation_results.json` (predecessor config) |
| 513 | `tab:classifiers` | Classifier comparison predecessor config (patch filter) | predecessor patch filter | predecessor config |
| 528 | `fig:classifier_reversal` | No trust classifier wins both surfaces | none (classifier only) | `fig8_classifier_reversal` |
| 549 | `tab:patch_audit` | Patch v2 catch/veto by bucket (PATCH, not mlp_v5) | patch filter | ledger `patch-catch-below-bar` |
| 567 | `fig:patch_catchbar` | Patch per-bucket catch rate vs decisiveness bar | patch filter | `fig8_patch_catchbar` |
| 579 | `tab:distill_verifier` | `mlp_v5` vs patch v2, per surface | **RGB `mlp_v5`** vs patch | `fig8_distill_verifier`; ledger `v5-beats-patch`; `eval/eval_v4_vs_patch.py` |
| 599 | `fig:distill_verifier_bar` | mlp_v5 vs patch vs bare, per surface | RGB `mlp_v5` | `fig8_distill_verifier`; from `knowledge/evals.csv` |
| 615 | `fig:failopen_expanded` | Fail-open frontier (mlp_v5 carve-out, not adopted) | RGB `mlp_v5` fail-open gate | `fig8_failopen_expanded`; `eval/failopen_expanded_ref.py` |
| 626 | `tab:ir_aligned` | Thermal-deploy filter `mlp_v5_ir_aligned` vs bare IR | **IR-th `aligned`** (+ patch ctx) | ledger `ir-grayscale-harvest-solves-thermal-filter`; CBAM FP regex-pinned |
| 652 | `fig:filter_operating` | Filter operating points (all 3 filters, det level) | **RGB + IR-th + IR-gray** | `fig_filter_operating`; `eval/filter_operating_sweep.py` → `eval/results/filter_operating_sweep.json` |
| 665 | `fig:mri_stats` | MRI LDA/ANOVA of FT4 ROI features (justifies filter) | (feature space; filter rationale) | `fig8_mri_lda`/`fig8_mri_anova`; `mri/cli.py` |
| 678 | `fig:mri_activation` | MRI activation "brain scan" (what filter reads) | (filter rationale) | `fig8_mri_act_*` |
| 691 | `fig:ir_gray_align` | Gray↔thermal feature alignment (enables aligned filter) | IR filter training rationale | `fig9_ir_gray_align`; `mri/modality_align.py` |
| 713 | `tab:gray_threeway` | Grayscale three-way on Svanstrom (bare detectors, NO filter) | none | `thesis_eval/results/tier1_results.json` (A_bare) |
| 732 | `tab:realvideo_master` | Real-video six-mode diagnostic (bare detectors, NO filter) | none | `eval/eval_video_tests.py` |
| 753 | `fig:grayscale_qualitative` | Cross-modal transfer night frame (NO filter) | none | `fig_grayscale_panel` |

---

## (3) QUALITATIVE CLAIMS A SWAP COULD FALSIFY

Each claim is an assertion whose truth depends on the filter's measured behaviour. Swapping a filter
(weights, threshold, or scaler) could **flip the sign** of the claim while the prose stays unchanged —
the silent-corruption risk.

1. **"The RGB filter does all the work; the IR filter does nothing on Svanstrom."**
   - L81: *"the mlp_v5 filter does that work (2019 → 353 alone, → 330 composed)"*; L31–32 table:
     `filt (mlp_v5,RGB)` F1 0.907 vs `filt (aligned,IR)` F1 0.742 (= bare, **identical**).
   - *Why it could flip:* the IR thermal filter being inert on Svanstrom is the whole basis for splitting
     the row. A new IR filter that fires on Svanstrom RGB-fed detections would break "does nothing."

2. **"The filter raises recall over bare on Svanstrom (0.948 → 0.991)."**
   - L21/L79: framed as a *routing* effect, not the filter; but the **production cell embeds the filter**.
   - *Why it could flip:* a more aggressive RGB filter vetoes true drones → composed recall could drop
     below bare, falsifying the headline RQ1 claim. The whole "filters never raise recall" contract.

3. **"The filter cuts RGB confuser FP 835 → 29 (−96.5%), ~10× stronger and 37–72× faster than the patch CNN."**
   - L170; numbers `rgbconf mlp FP`=29, `filt_patch`=282.
   - *Why it could flip:* any change to `mlp_v5` weights/threshold moves the 29; the "~10× stronger than patch"
     and the headline 30.3% → 1.1% trajectory both ride on it.

4. **"Grayscale filter weights transfer through nothing but the input scaler (656 → 21, −96.8%), where the
   thermal scaler on the same frames manages only 656 → 280."**
   - L170/L645/L772; `grayconf mlp FP`=21.
   - *Why it could flip:* this is the "per-modality standardisation is load-bearing" claim. Swapping the
     grayscale scaler (or retraining the one net) could erase the gap and falsify "load-bearing, not a detail."

5. **Grayscale over-veto / "no operating point preserves drone recall on grayscale Svanstrom (recall ≤ 0.27 even @0.02)."**
   - L772; also `fig:filter_operating` "retains only 46.7% at 0.25."
   - *Why it could flip:* a grayscale filter trained on Svanstrom-scale silhouettes would lift recall and
     falsify "confuser-suppression tool only, gated rather than unconditional." Directly contradicts the
     deployment rule "grayscale detection runs unfiltered."

6. **The airplane hole / "thermal confusers resist; aligned filter absorbs only ~18–20%; fire flat at every threshold."**
   - L173/L649/L802; `irconf composed r8 fire`=0.217, nr 0.237.
   - *Why it could flip:* a native-thermal-airplane-trained IR filter would cut thermal confuser fire and
     falsify both "thermal confusers resist" and "airplanes remain the weakest category for every filter
     generation." Also undermines the future-work framing in §sec:limits.

7. **Composition order: "filt→clf is the better side for the no-reject router (recall), and the two orders give
   IDENTICAL confuser fire (a no-reject router never vetoes a confuser frame)."**
   - L83/L149/L162: `clf->filt[robust8_nr_drop]` = `filt->clf[robust8_nr_drop]` to the digit on confusers
     (`NR rgb_conf fire` == `NR rgb_conf fire filt->clf`).
   - *Why it could flip:* this identity is a structural property of the *no-reject* router + filter. A filter
     that interacts with the router state, or a router that can veto, breaks the identity and the
     "composition order is immaterial on confusers" claim — and re-opens the filt→clf vs clf→filt shipping choice.

8. **Carve-out: "mlp_v5 regresses on rgb_dataset (−11 pp F1) but it is an OOD coverage gap (small drones), not
   structural overlap; recovered by temporal voting on video so the carve-out is retained not patched."**
   - L216/L248/L609/L612; `rgbtest mlp F1`=0.809, per-size `SZ rgbtest <16 filt R`=0.256.
   - *Why it could flip:* a filter with broader small-drone coverage would remove the regression and make the
     entire carve-out / fail-open-not-adopted narrative obsolete. Note the **two different rgb_dataset numbers**
     (tab:ablation_solo 0.809/0.691 vs tab:distill_verifier 0.792/0.664) — a swap must update both consistently.

9. **"The filter is recall-transparent at every detector floor on SelCom (filtered R == bare R); lowering the
   floor converts directly into recall."**
   - L282/L295; `SWEEP selcom filt@0.05`=0.692.
   - *Why it could flip:* recall-transparency is the mechanism that makes the low-conf operating mode safe. A
     filter that vetoes true drones at low floor breaks "nothing true is lost" and the operating-mode recommendation.

10. **"SelCom: filter cuts FP 22 → 7 at ZERO recall cost (P 0.858 → 0.950); the abstention behaviour that
    motivated the feature-reuse design."**
    - L216; `selcom mlp F1`=0.6115 (R unchanged at 0.451).
    - *Why it could flip:* zero recall cost on CCTV-scale crops is the load-bearing "abstain when unsure" property.
      A swapped filter that vetoes a SelCom drone falsifies the abstention claim and the design motivation.

11. **CBAM held-out: "aligned filter cuts novel-confuser FP 48 → 15 (−69%) while preserving thermal drone recall
    (ΔR ≤ −0.007 in-distribution); patch cuts only 7."**
    - L634/L639; FP=15 (regex-pinned in two .tex files + evals.csv canonical).
    - *Why it could flip:* this is the headline IR-filter generalisation result. A swapped IR net changes 15 and
      possibly the ΔR bound — and the audit cross-pins this number across methodology + empirical, so an
      inconsistent swap is caught, but a *consistent-but-wrong* swap silently corrupts the held-out story.

12. **"On the IR test split the aligned filter costs only 0.8 pp recall (recall-safe by design) where the patch
    costs 5.0 pp."**
    - L216; `irtest mlp F1`=0.9578 vs bare 0.961.
    - *Why it could flip:* "recall-safe" is the IR filter's defining property (vs patch). A swap raising the IR
      recall cost falsifies the design claim and the drone-diversity-remine rationale (L645).

13. **"Filter operating points sit deliberately on the high-precision side; RGB recall is recoverable almost for
    free, but IR-thermal/grayscale limits are coverage not threshold."**
    - L649; `fig:filter_operating` (RGB 0.956@0.25, gray 0.467@0.25).
    - *Why it could flip:* the "RGB = threshold tool / IR = coverage limit" dichotomy is the deployer-facing
      reading. Different filter curves would invert which lever helps and falsify the recommendation.

---

## TOTALS

- **Numbers (filter-dependent cells, distinct line×metric entries above):** ~120 individual metric values
  across the tables and prose (counting each TP/FP/FN/P/R/F1 and each inline figure separately).
  - **Of which AUDITED** (covered by a named CHECK in `_audit_headline_numbers.py`): ~38 distinct cells
    (the F1 cells of every paired/DUT/confuser/solo filter row, the NR shipped cells, the filter-operating
    sweep, the conf-sweep SelCom F1, the per-size filt-recall buckets, the CBAM FP regex-pin, clean-split F1).
  - **Of which UN-AUDITED (RISK):** the majority — **all per-cell TP/FP/FN/P/R** in the paired and DUT tables
    (only the F1 is checked), every `tab:temporal_production` mlp filt-only and composed-robust6 window cell,
    every `tab:distill_verifier` cell (ledger-only, different s9/IoU configs), every `tab:ir_aligned` row
    except the CBAM FP, the `tab:lowconf_selcom` 0.10/0.05 P/R, the runtime/latency table, the fail-open
    diagnosis distances, the 0.834/0.733 composed means, the robust8-nr clean-split −3.3pp (0.911), the
    grayscale ≤0.27 sweep, and the thermal-scaler-on-gray 656→280.

- **Figures touching a filter:** 9 — `fig:pipeline_ablation`, `fig:cascade_segment_fig` (predecessor patch),
  `fig:patch_catchbar` (patch), `fig:distill_verifier_bar` (mlp_v5), `fig:failopen_expanded` (mlp_v5),
  `fig:filter_operating` (all 3 filters), `fig:mri_stats`/`fig:mri_activation`/`fig:ir_gray_align` (filter rationale).

- **Tables touching a filter:** 13 — `tab:ablation_svanstrom`, `tab:ablation_antiuav`, `tab:rq3`,
  `tab:ablation_dut`, `tab:ablation_confusers`, `tab:ablation_solo`, `tab:per_size`, `tab:lowconf_selcom`,
  `tab:speed`, `tab:temporal_production`, `tab:robust6_pipeline` (predecessor), `tab:distill_verifier`,
  `tab:ir_aligned`. (Plus `tab:patch_audit` and `tab:classifiers` reference the *patch* predecessor, not mlp_v5.)

- **Qualitative claims a swap could falsify:** 13 (enumerated above).

### Highest-risk swap targets (un-audited + load-bearing)
- All **non-F1 cells** in `tab:ablation_svanstrom`/`antiuav`/`dut` (TP/FP/FN/P/R) — a filter swap changes
  these but no CHECK guards them; the 835/353/330/195/2019 FP figures in the prose ride on them.
- **`tab:ir_aligned`** rows (only CBAM FP=15 is pinned) — every other P/R/F1/FP is ledger-only.
- **`tab:distill_verifier`** — ledger-sourced under s9/IoU configs that don't match the tier1 JSON, so a swap
  to the canonical harness numbers could silently diverge from 0.869/0.792 etc.
- **`tab:temporal_production`** mlp filt-only + composed-robust6 window cells — no CHECK; OOD video where
  the filter over-vetoes, so most sensitive to a swap.
- **Grayscale-scaler claims** (656→21, ≤0.27 over-veto, 46.7%@0.25) — partly audited (`grayconf mlp FP`,
  `FIG gray recall@0.25`) but the over-veto sweep bound is not.
```
