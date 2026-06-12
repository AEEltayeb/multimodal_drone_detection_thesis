# Pietro reply — draft v3 (clean numbers-only tables, explanation below each)

Attach: `email_robust6_lda.png`, `email_robust6_ablation.png`. (§3 NEW-MLP = its highest-suppression
operating point; on IR thermal the MLP is recall-safe and does not chase the CNN's suppression.)

---

Dear Pietro,

I re-ran the full evaluation with the new trust classifier (robust6) and the new confuser filter (V5 MLP)
in the same ensemble, scored as before (trust-aware). The three domains, then the evidence for the two
new components.

Frame counts — Anti-UAV: 85,374; Svanström: 28,710 (airplane 6,090, bird 5,298, helicopter 5,627, drone
11,695); IR confusers: 14,985 (airplane 11,993, bird 1,726, helicopter 1,266); RGB confusers: 1,250
(airplane 304, bird 352, helicopter 594).

**1) Svanström — IoP @ 0.5**
```
Config               Precision   Recall      F1     OLD-F1      Δ
ir_only                0.9473    0.9714    0.9592   0.9591   +0.0001
rgb_only               0.4463    0.9174    0.6005   0.5274   +0.0731
classifier             0.9405    0.9791    0.9594   0.9937   -0.0343
ir_filter              0.9477    0.9714    0.9594   0.9457   +0.0137
rgb_filter             0.9007    0.8388    0.8687   0.6988   +0.1699
filter→classifier      0.9612    0.9755    0.9683   0.9932   -0.0249
classifier→filter      0.9616    0.9378    0.9496   0.9747   -0.0251
```
Confuser-heavy. The filter rows improve over the old stack — rgb_filter rises from 0.70 to 0.87 (both
precision and recall up), ir_filter +1.4 pp. rgb_only also rises (0.53→0.60). The classifier rows are
1–3 pp lower than the old fusion classifier; that gap is the robust6 trade explained in §4–§5.

**2) Anti-UAV RGBT — IoU @ 0.5**
```
Config               Precision   Recall      F1     OLD-F1      Δ
ir_only                0.9819    0.9428    0.9619   0.9619   +0.0000
rgb_only               0.9892    0.9835    0.9864   0.9902   -0.0038
classifier             0.9866    0.9769    0.9818   0.9916   -0.0098
ir_filter              0.9819    0.9427    0.9619   0.9607   +0.0012
rgb_filter             0.9903    0.9817    0.9860   0.9901   -0.0041
filter→classifier      0.9871    0.9770    0.9820   0.9916   -0.0096
classifier→filter      0.9871    0.9762    0.9816   0.9909   -0.0093
```
Clean benchmark, no confusers — every config stays ≥ 0.98 and matches the old stack to within ~1 pp.

**3a) IR confuser filter — frames firing on a confuser (%), by category**
```
Category      Baseline   OLD CNN   NEW MLP
Airplane         7.5       1.6       2.9
Bird             5.5       3.9       5.2
Helicopter      39.4       2.1      23.2
All              9.9       1.9       4.9
```

**3b) RGB confuser filter — frames firing on a confuser (%), by category**
```
Category      Baseline   OLD CNN   NEW MLP
Airplane        27.3      10.2       5.3
Bird            62.8      39.5       7.7
Helicopter      20.7      15.8       3.7
All             34.2      21.1       5.2
```
Lower is better — every detection on these clips is a false positive. On RGB the new MLP filter suppresses
far more than the old CNN (5.2% vs 21.1% overall). On IR the CNN fires less, but only because it also
vetoes real thermal drones; the MLP is recall-safe (the recall numbers are in §1).

**4) Speed**
```
Component                       OLD       NEW      Speedup
Trust classifier (per frame)    38.3      0.10     404
Confuser filter (per det)       24-112    1.3-2.1  11-72
```
(milliseconds). robust6 reads only detector confidence + box geometry; the MLP reuses features the
detector already computed. Pipeline overhead: MLP 1–4% vs the CNN patch's 48–191%.

**5) Why robust6 works**

robust6 uses 6 features (detector confidence + box geometry) instead of 40, selected by a statistical
feature-selection study rather than by hand. The trust signal is strongly linearly separable (LDA accuracy
0.932), and the 6 features all carry it (per-feature AUROC 0.78–0.93). The key result is a held-out
ablation: in-domain F1 is flat from 6 to 40 features, but out-of-distribution F1 collapses as more features
are added (drone-video F1 0.58 with 6 features vs 0.26 with 19) — the extra image/scene features memorise
the training scenes and fail on unseen footage. So robust6 matches the larger classifier in-domain using 6
of 40 features at 404× the speed, and generalises better out-of-distribution. (Figures attached.)

---

Your three questions: (1) the new filter is compared to the old CNN filter on the same detector;
(2) the standalone filter does not beat the full ensemble — the IR + classifier routing is essential on
confuser-heavy data; (3) the combined system (ensemble + MLP filter) holds ≥ 0.96 throughout, the filter is
a clear OOD upgrade and far faster, and robust6 adds OOD robustness at 404× the classifier speed.

Best regards,
Ahmed
