"""
Caliper (DCAL) feature helper for the alternative-model experiments.

Extends the frozen 5-feature pipeline with a per-well normalized differential
caliper (``DCAL_NORM``) as an optional sixth input, without mutating
``FEATURE_COLS`` or the published MLP/PINN pipeline. ``preprocess_well`` already
carries the raw ``DCAL`` column through to its output aligned with the surviving
rows, so the normalized caliper is derived directly from there.

Called by: scripts/11_train_xgboost.py, scripts/13_train_lstm.py
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    DCAL_COL,
    FEATURE_COLS,
    TARGET_COL,
    WellScaler,
    preprocess_well,
)

CALIPER_COL = "DCAL_NORM"
FEATURE_COLS_EXT = FEATURE_COLS + [CALIPER_COL]
_WEIGHT_COL = "DCAL_WEIGHT"


def caliper_zscore(df: pd.DataFrame) -> np.ndarray:
    """Per-well z-score of the raw differential caliper, aligned to df rows.

    Computes ``(DCAL - mean) / std`` over the well's valid caliper samples and
    fills any missing value with 0 (neutral, in-gauge). Returns a column of
    zeros when DCAL is absent or effectively constant, so a well without a
    usable caliper contributes a neutral feature rather than noise.

    Args:
        df: Processed well DataFrame that still carries the raw ``DCAL`` column.

    Returns:
        Float32 array of shape (len(df),) with the normalized caliper.
    """
    n = len(df)
    if DCAL_COL not in df.columns:
        return np.zeros(n, dtype=np.float32)
    dcal = df[DCAL_COL].astype(float)
    valid = dcal.dropna()
    if valid.shape[0] < 10 or float(valid.std()) < 1e-6:
        return np.zeros(n, dtype=np.float32)
    mean, std = float(valid.mean()), float(valid.std())
    return ((dcal - mean) / std).fillna(0.0).to_numpy(dtype=np.float32)


def preprocess_well_with_caliper(
    df: pd.DataFrame,
    well_id: str,
    fit: bool = True,
    scaler: WellScaler | None = None,
) -> tuple[pd.DataFrame, WellScaler]:
    """Run the standard per-well preprocessing and append ``DCAL_NORM``.

    Thin wrapper over ``preprocess_well`` that adds the per-well normalized
    caliper as a sixth feature. The published pipeline is untouched: the global
    ``FEATURE_COLS`` and the MLP are not modified.

    Args:
        df: Raw well DataFrame with canonical curve names.
        well_id: Identifier for the well (used by its scaler).
        fit: If True, fit a new scaler on this well; else use ``scaler``.
        scaler: Pre-fitted scaler to use when ``fit`` is False.

    Returns:
        Tuple of (processed DataFrame with ``DCAL_NORM``, fitted/applied scaler).
    """
    proc, scaler = preprocess_well(df, well_id, fit=fit, scaler=scaler)
    proc[CALIPER_COL] = caliper_zscore(proc)
    return proc, scaler


def build_arrays(
    df: pd.DataFrame,
    with_caliper: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble model-ready arrays from a processed (with-caliper) DataFrame.

    Returns the feature matrix plus the auxiliary signals the physics term needs
    (normalized NPHI and GR, and the caliper quality weight), all aligned by row.

    Args:
        df: Processed DataFrame from ``preprocess_well_with_caliper``.
        with_caliper: If True, include ``DCAL_NORM`` as the sixth feature.

    Returns:
        Tuple of (X, y, nphi, gr, w, depth) as float32 numpy arrays:
            X     — features, shape (n, 6) if with_caliper else (n, 5)
            y     — normalized DEN target, shape (n,)
            nphi  — normalized NPHI, shape (n,)
            gr    — normalized GR, shape (n,)
            w     — caliper quality weight in [0, 1], shape (n,)
            depth — depth in feet (or a synthetic 0.5-ft grid if absent), shape (n,)
    """
    cols = FEATURE_COLS_EXT if with_caliper else FEATURE_COLS
    x = df[cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    nphi = df["NPHI"].to_numpy(dtype=np.float32)
    gr = df["GR"].to_numpy(dtype=np.float32)
    if _WEIGHT_COL in df.columns:
        w = df[_WEIGHT_COL].to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)
    if "DEPTH" in df.columns:
        depth = df["DEPTH"].to_numpy(dtype=np.float32)
    else:
        depth = (np.arange(len(df)) * 0.5).astype(np.float32)
    return x, y, nphi, gr, w, depth
