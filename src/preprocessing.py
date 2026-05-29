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

    # ----------------------------------------
    # Step 1 — Log transform resistivity
    # ----------------------------------------
    if log_rt:
        out = apply_log_rt(out)

    # ----------------------------------------
    # Step 2 — Fit or apply per-well scaler
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
