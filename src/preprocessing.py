"""
Preprocessing pipeline for well log DataFrames.

Applies log10 transform to RT (resistivity) and per-well normalization.
Normalization strategy and log-RT decision are set at construction time
so the same scaler can be applied consistently across LOWO folds.

Called by: src/dataset.py, scripts/run_eda.py
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = ["GR", "RT", "RILM", "NPHI", "SP"]
TARGET_COL = "DEN"
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# Feature bounds — values are CLIPPED to the boundary (rows preserved).
# Designed to neutralize sentinel values and tool spikes without losing data.
# Per-well min-max normalization rescales the survived range to [0,1] regardless.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "GR":   (0.0, 400.0),       # GAPI  — covers high-shale spikes; >400 = sentinel
    "RT":   (0.05, 50_000.0),   # Ohm·m — wide; catches sentinels (100,000+) only
    "RILM": (0.05, 50_000.0),   # Ohm·m — same as RT
    "NPHI": (-0.15, 0.80),      # v/v   — generous; catches residual unit issues
    "SP":   (-1000.0, 1000.0),  # mV    — very loose; per-well norm absorbs offsets
}

# Target bounds — rows OUTSIDE this range are DROPPED (cannot train on bad DEN).
# Tight enough to remove washouts and tool errors, loose enough to keep coal/gas.
TARGET_BOUNDS: tuple[float, float] = (1.5, 3.1)  # g/cc

# Legacy alias for backwards-compatible imports.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {**FEATURE_BOUNDS, TARGET_COL: TARGET_BOUNDS}


@dataclass
class WellScaler:
    """Per-well min-max scaler storing fit parameters.

    Args:
        well_id: Identifier for the well.
        mins: Dict mapping column name → minimum value used for scaling.
        ranges: Dict mapping column name → (max - min) used for scaling.
    """

    well_id: str
    mins: dict[str, float] = field(default_factory=dict)
    ranges: dict[str, float] = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling using stored parameters.

        Args:
            df: DataFrame with columns to scale.

        Returns:
            Scaled DataFrame (values in [0, 1] for fitted columns).
        """
        out = df.copy()
        for col in self.mins:
            if col in out.columns:
                rng = self.ranges[col]
                out[col] = (out[col] - self.mins[col]) / (rng if rng > 0 else 1.0)
        return out

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Reverse scaling for the DEN target column.

        Args:
            values: Scaled values to invert.

        Returns:
            Values in original g/cc units.
        """
        rng = self.ranges.get(TARGET_COL, 1.0)
        mn = self.mins.get(TARGET_COL, 0.0)
        return values * rng + mn


def fit_scaler(well_id: str, df: pd.DataFrame, cols: list[str] | None = None) -> WellScaler:
    """Fit a per-well min-max scaler on the given columns.

    Args:
        well_id: Identifier for the well.
        df: DataFrame to fit on.
        cols: Columns to scale. Defaults to ALL_COLS.

    Returns:
        Fitted WellScaler instance.
    """
    cols = cols or ALL_COLS
    scaler = WellScaler(well_id=well_id)
    for col in cols:
        if col not in df.columns:
            continue
        mn = float(df[col].min())
        mx = float(df[col].max())
        scaler.mins[col] = mn
        scaler.ranges[col] = mx - mn
    return scaler


def clip_features_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip feature columns to their physical bounds (in-place per column).

    Values outside FEATURE_BOUNDS are clipped to the boundary value, which:
      - neutralizes sentinel values (e.g., RT=100,000) without losing the row
      - prevents single tool spikes from distorting min-max normalization
      - preserves the row so the (still-valid) target DEN can train the model

    Must be called before apply_log_rt so RT/RILM enter the log transform
    with values already in [0.05, 50000] Ohm·m.

    Args:
        df: DataFrame with canonical feature columns in raw units.

    Returns:
        DataFrame copy with feature values clipped to their physical bounds.
    """
    out = df.copy()
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def filter_invalid_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the DEN target is outside its physical range.

    Unlike features (which we clip), the target cannot be faked: training on
    DEN=1.0 (washout) or DEN=4.0 (sentinel) corrupts the model's loss signal.
    Rows with DEN outside TARGET_BOUNDS are removed entirely.

    Args:
        df: DataFrame with a DEN column in g/cc.

    Returns:
        DataFrame copy with invalid-target rows removed.
    """
    if TARGET_COL not in df.columns:
        return df.copy()
    lo, hi = TARGET_BOUNDS
    mask = (df[TARGET_COL] >= lo) & (df[TARGET_COL] <= hi)
    return df.loc[mask].reset_index(drop=True)


def clip_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper: clip features + filter invalid DEN rows.

    Equivalent to filter_invalid_target_rows(clip_features_to_bounds(df))
    but returned without resetting the index so legacy tests that rely on
    NaN values for out-of-range entries continue to detect them.

    Args:
        df: DataFrame with canonical curve columns in raw units.

    Returns:
        DataFrame with features clipped and DEN outliers set to NaN.
    """
    out = clip_features_to_bounds(df)
    if TARGET_COL in out.columns:
        lo, hi = TARGET_BOUNDS
        out.loc[(out[TARGET_COL] < lo) | (out[TARGET_COL] > hi), TARGET_COL] = np.nan
    return out


def apply_log_rt(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log10 transform to RT and RILM columns in-place.

    Resistivity follows a log-normal distribution; log10 linearizes
    the range and improves gradient flow during training.

    Args:
        df: DataFrame containing RT and/or RILM columns.

    Returns:
        DataFrame with RT and RILM replaced by their log10 values.
    """
    out = df.copy()
    for col in ("RT", "RILM"):
        if col in out.columns:
            # Guard against zero or negative values before log
            out[col] = np.log10(out[col].clip(lower=1e-4))
    return out


def preprocess_well(
    df: pd.DataFrame,
    well_id: str,
    log_rt: bool = True,
    fit: bool = True,
    scaler: WellScaler | None = None,
) -> tuple[pd.DataFrame, WellScaler]:
    """Full preprocessing pipeline for a single well.

    Args:
        df: Raw DataFrame with canonical curve names.
        well_id: Identifier for the well (used in scaler).
        log_rt: Whether to apply log10 to RT and RILM.
        fit: If True, fit a new scaler on this well. If False, use provided scaler.
        scaler: Pre-fitted scaler to use when fit=False.

    Returns:
        Tuple of (processed DataFrame, fitted or provided WellScaler).

    Raises:
        ValueError: If fit=False and no scaler is provided.
    """
    if not fit and scaler is None:
        raise ValueError("scaler must be provided when fit=False")

    out = df.copy()
    rows_initial = len(out)

    # ----------------------------------------
    # Step 1 — Clip features to physical bounds (preserve rows)
    # ----------------------------------------
    out = clip_features_to_bounds(out)

    # ----------------------------------------
    # Step 2 — Drop rows with invalid DEN target
    # ----------------------------------------
    out = filter_invalid_target_rows(out)
    rows_dropped = rows_initial - len(out)
    if rows_dropped > 0:
        logger.debug(
            "Dropped %d / %d rows from %s (DEN outside %s)",
            rows_dropped,
            rows_initial,
            well_id,
            TARGET_BOUNDS,
        )

    # ----------------------------------------
    # Step 3 — Log transform resistivity (RT, RILM)
    # ----------------------------------------
    if log_rt:
        out = apply_log_rt(out)

    # ----------------------------------------
    # Step 4 — Fit or apply per-well scaler
    # ----------------------------------------
    if fit:
        scaler = fit_scaler(well_id, out)

    assert scaler is not None
    out = scaler.transform(out)

    return out, scaler


def preprocess_wells(
    wells: dict[str, pd.DataFrame],
    log_rt: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, WellScaler]]:
    """Preprocess all wells independently (each well gets its own scaler).

    Args:
        wells: Dict mapping well_id to raw DataFrame.
        log_rt: Whether to apply log10 to RT and RILM.

    Returns:
        Tuple of (processed wells dict, scalers dict).
    """
    processed: dict[str, pd.DataFrame] = {}
    scalers: dict[str, WellScaler] = {}

    for well_id, df in wells.items():
        proc, scaler = preprocess_well(df, well_id, log_rt=log_rt, fit=True)
        processed[well_id] = proc
        scalers[well_id] = scaler
        logger.debug("Preprocessed %s — %d rows", well_id, len(proc))

    return processed, scalers
