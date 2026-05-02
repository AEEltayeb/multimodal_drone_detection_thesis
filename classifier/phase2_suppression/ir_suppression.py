"""
ir_suppression.py — Utility to simulate IR modality failure on a dataframe.

Purpose:
  The IR YOLO model was trained on both datasets we use for evaluation, so
  `max_conf_ir` is essentially an oracle. To measure whether the fusion
  classifier CAN learn conditional trust (fall back on RGB when IR fails),
  we synthetically suppress IR features on a controlled subset of rows.

  This simulates scenarios where IR genuinely fails in the field:
    - thermal crossover (drone temp ≈ ambient)
    - IR sensor saturation (sun in frame, hot rooftops)
    - IR occlusion / out-of-range
    - sensor fault

  Suppression = "IR model emitted no detection on this frame".
  The label (drone present or not) is unchanged — a drone is still a drone
  even if the IR sensor didn't see it.
"""

import numpy as np
import pandas as pd


# IR-specific features (hard zero when suppressed)
IR_PRIMARY = ["max_conf_ir", "n_dets_ir", "ir_area_norm", "conf_ir_2nd"]


def suppress_ir_features(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    Return a copy of `df` where IR features are zeroed on rows where mask is True
    and derived features (conf_max/min/mean/delta, both_detected, n_dets_total)
    are recomputed as if IR emitted zero detections on those rows.

    The label column is NOT touched — this is the whole point. We are
    asking: can the classifier still predict the drone is present, using
    RGB features alone, when IR goes silent?

    Formulas used (from build_dataset.py):
        conf_max   = max(max_rgb, max_ir)   -> max_rgb       when IR=0
        conf_min   = min(max_rgb, max_ir)   -> 0             when IR=0
        conf_mean  = (max_rgb + max_ir)/2   -> max_rgb/2     when IR=0
        conf_delta = abs(max_rgb - max_ir)  -> max_rgb       when IR=0
    """
    if not mask.any():
        return df.copy()

    out = df.copy()

    # Zero IR-specific features
    for col in IR_PRIMARY:
        if col in out.columns:
            out.loc[mask, col] = 0.0

    # Recompute derived features assuming IR is silent
    if "max_conf_rgb" in out.columns:
        rgb = out.loc[mask, "max_conf_rgb"].values
        if "conf_max" in out.columns:
            out.loc[mask, "conf_max"] = rgb
        if "conf_min" in out.columns:
            out.loc[mask, "conf_min"] = 0.0
        if "conf_mean" in out.columns:
            out.loc[mask, "conf_mean"] = rgb / 2.0
        if "conf_delta" in out.columns:
            out.loc[mask, "conf_delta"] = rgb

    if "both_detected" in out.columns:
        out.loc[mask, "both_detected"] = 0

    if "n_dets_total" in out.columns and "n_dets_rgb" in out.columns:
        out.loc[mask, "n_dets_total"] = out.loc[mask, "n_dets_rgb"]

    return out


def random_suppression_mask(n_rows: int, rate: float,
                            random_state: int = 42) -> np.ndarray:
    """Return a boolean mask selecting a random `rate` fraction of rows."""
    rng = np.random.default_rng(random_state)
    return rng.random(n_rows) < rate
