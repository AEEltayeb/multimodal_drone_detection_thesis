# Reply to Pietro — baseline clarification + CNN→MLP filter swap (2026-06-02)

**Attach:**
- `mri/results/v5_report_regen/images/v5_prod_per_surface_bars.png` (bare vs CNN/patch-v2 vs MLP filter)
- `mri/results/v5_report_regen/images/v5_prod_latency_per_det.png` (CNN vs MLP latency, 46–62×)

---

Dear Pietro,

Thank you — and you're right that the report didn't make the baseline explicit. Let me clarify.

**1. The baseline.** The `classifier` row in my tables *is* the full ensemble you remember (IR + RGB + the trust meta-model); it still scores >0.97 (Anti-UAV 0.9916, Svanström 0.9937). The `*_filter` rows are that same ensemble with one change — the **old CNN filter replaced by the new MLP filter**. I never showed the old CNN filter as its own row, which is what made the comparison unclear.

**2. "Does the new solution beat the full ensemble?"** The MLP filter isn't meant to replace the ensemble — it replaces the **CNN filter stage inside it**. The trust classifier is what delivers the >0.97, and that is unchanged. So the meaningful question is old-CNN-filter vs new-MLP-filter *within the same ensemble*, which is your next point.

**3. Best of both worlds.** That configuration is already in the tables: `filter→classifier` and `classifier→filter` are the full ensemble running the MLP filter, and they hold >0.97 (Anti-UAV 0.9916 / 0.9909, Svanström 0.9932 / 0.9747). On paired / in-domain data the filter is largely redundant — the classifier already removes most confusers — so CNN vs MLP barely changes the score there.

Where the MLP clearly beats the old CNN filter is on **confusers / out-of-distribution data and on cost**:

- Filter F1 on Svanström: **CNN 0.874 → MLP 0.964**
- Confuser-zoo false positives: **CNN 754 → MLP 121**
- Held-out thermal confusers (CBAM): **CNN 0.66 → MLP 0.82**
- **46–62× faster** (~1.5 ms vs 80–109 ms per detection), with near-zero added pipeline cost — the MLP reuses the detector's own features instead of re-processing each crop with a separate CNN.

One honest caveat: on photo-style still images the MLP slightly trails the CNN filter, but that is not the deployment surface (on video, frame-level recall is unaffected).

So my recommendation matches yours: keep the IR + RGB + trust ensemble and swap the CNN filter for the MLP — equal accuracy on clean/paired data, better on confusers/OOD, and far cheaper to run.

Two figures attached:

- **Per-surface filter comparison** — bare vs CNN (patch v2) vs MLP filter.
- **Per-detection latency** — CNN vs MLP (the 46–62× speed-up).

If helpful, I can add a single side-by-side table of the full ensemble with the CNN filter vs the MLP filter on Anti-UAV and Svanström — just let me know.

Best,
Ahmed
