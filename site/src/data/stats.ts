// Canonical, thesis-faithful numbers. Every value is verbatim from the thesis
// (docs/thesis_working_distilling_overleaf/) and traceable to knowledge/claims.csv.
// Do NOT edit these to "round" - they are the audited figures.
export const stats = {
  // Detectors are already good on ordinary footage (in-distribution sanity floor)
  antiuav_bare_P: 0.989,
  antiuav_bare_R: 0.982,
  inDistHalluc: 0.028, // 2.8% hallucination on the in-distribution RGB test set

  // The narrow-but-stubborn failure: out-of-distribution confusers
  confuserFire: 0.304, // 30.4% of OOD confuser frames trigger a false drone alert
  birdFire: 0.944, // 94.4% on bird-only Svanstrom footage

  // "Just retrain it" backfires
  baselineRecall: 0.961, // baseline Svanstrom drone recall
  retrainRecall: 0.306, // recall after aggressive bird-suppression retrain

  // Resolution gap
  svanMedianPx: 29.8,
  recallBelowFloor: 0.63,

  // Modality reversal (each modality scored vs its own GT; routed is trust-aware)
  svan_rgb: 0.607,
  svan_ir: 0.94,
  svan_routed: 0.946,
  auv_rgb: 0.985,
  auv_ir: 0.961,
  auv_routed: 0.984,

  // Full-pipeline headline (shipped robust8-nr, filt->clf)
  svan_bare_f1: 0.742,
  svan_pipe_f1: 0.946,
  svan_bare_recall: 0.948,
  svan_pipe_recall: 0.991,
  confuser_pipe_fire: 0.014, // 1.4%
} as const
