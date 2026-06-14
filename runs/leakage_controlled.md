# Leakage-controlled replays (round-3 N12/N13) — sequence-level clean subsets
2026-06-12 14:57 | Tier-1 cache replay, zero-GPU | patch_thr=0.5
det-clean = eval sequences with ZERO frames in IR_dset_final/train;
cascade-clean = det-clean AND not in the shipped router's reconstructed train groups.

## svanstrom — det-clean: 52/273 sequences

**det-clean (IR detector never trained on these sequences)** (n=767, GT rgb/ir = 295/280)

| arm | clean P | clean R | clean F1 [95% CI] | headline F1 |
|---|---|---|---|---|
| bare | 0.5534 | 0.9374 | 0.6959 [0.6667–0.7261] | 0.7415 |
| filt_mlp | 0.8562 | 0.9113 | 0.8829 [0.8633–0.9033] | 0.9071 |
| clf[robust8] | 0.8325 | 0.9721 | 0.8969 [0.8761–0.9161] | 0.9414 |
| clf->filt[robust8] | 0.9152 | 0.9441 | 0.9294 [0.9131–0.9456] | 0.9485 |
| bare ft4/rgb (own GT) | 0.4288 | 0.9492 | 0.5907 | 0.6067 |
| bare v3b/ir (own GT) | 0.8069 | 0.925 | 0.8619 | 0.9401 |

**cascade-clean (9 seqs; also outside router training)** (n=136, GT rgb/ir = 61/45)

| arm | clean P | clean R | clean F1 [95% CI] | headline F1 |
|---|---|---|---|---|
| bare | 0.5763 | 0.9623 | 0.7208 [0.6525–0.7812] | 0.7415 |
| filt_mlp | 0.8293 | 0.9623 | 0.8908 [0.8404–0.9356] | 0.9071 |
| clf[robust8] | 0.8 | 0.9709 | 0.8772 [0.8265–0.9193] | 0.9414 |
| clf->filt[robust8] | 0.9346 | 0.9709 | 0.9524 [0.9186–0.9798] | 0.9485 |
| bare ft4/rgb (own GT) | 0.5179 | 0.9508 | 0.6705 | 0.6067 |
| bare v3b/ir (own GT) | 0.6769 | 0.9778 | 0.8 | 0.9401 |

## antiuav — det-clean: 60/90 sequences

**det-clean (IR detector never trained on these sequences)** (n=2669, GT rgb/ir = 2537/2647)

| arm | clean P | clean R | clean F1 [95% CI] | headline F1 |
|---|---|---|---|---|
| bare | 0.9784 | 0.9709 | 0.9746 [0.9709–0.9779] | 0.9728 |
| filt_mlp | 0.9786 | 0.9707 | 0.9746 [0.9709–0.9779] | 0.9729 |
| clf[robust8] | 0.9824 | 0.9882 | 0.9853 [0.9824–0.9881] | 0.9845 |
| clf->filt[robust8] | 0.9826 | 0.988 | 0.9853 [0.9823–0.9882] | 0.9844 |
| bare ft4/rgb (own GT) | 0.9877 | 0.9838 | 0.9858 | 0.9853 |
| bare v3b/ir (own GT) | 0.9694 | 0.9584 | 0.9639 | 0.961 |

**cascade-clean (18 seqs; also outside router training)** (n=833, GT rgb/ir = 794/831)

| arm | clean P | clean R | clean F1 [95% CI] | headline F1 |
|---|---|---|---|---|
| bare | 0.9689 | 0.9582 | 0.9635 [0.955–0.9708] | 0.9728 |
| filt_mlp | 0.9689 | 0.9575 | 0.9632 [0.9547–0.9706] | 0.9729 |
| clf[robust8] | 0.973 | 0.9779 | 0.9754 [0.9683–0.9818] | 0.9845 |
| clf->filt[robust8] | 0.973 | 0.9773 | 0.9751 [0.968–0.9815] | 0.9844 |
| bare ft4/rgb (own GT) | 0.9823 | 0.9786 | 0.9804 | 0.9853 |
| bare v3b/ir (own GT) | 0.9559 | 0.9386 | 0.9472 | 0.961 |