"""
Preprocessing pipeline for well log DataFrames.

Applies per-well statistical outlier detection (voting consensus: ≥2 of 5 independent
detectors), log10 transform to RT/RILM, and per-well Yeo-Johnson + z-score normalization.

Called by: src/dataset.py, scripts/run_eda.py
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer, StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = ["GR", "RT", "RILM", "NPHI", "SP"]
TARGET_COL = "DEN"
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# Features whose outlier detection is performed on log10 scale.
# RT and RILM are log-normal; linear-scale statistics produce excessive false positives.
LOG_FEATURES: frozenset[str] = frozenset({"RT", "RILM"})

# Normalization strategy per column, chosen from per-well skewness analysis.
# "yeo-johnson" (PowerTransformer) for consistently or severely skewed columns;
# "standard" (StandardScaler / z-score) for approximately symmetric columns.
# RT and RILM receive "standard" because log10 in Step 3 already linearizes them.
SCALER_TYPE: dict[str, str] = {
    "GR":   "yeo-johnson",  # consistently right-skewed across all wells (mean skew 1.53)
    "RT":   "standard",     # log10 pre-transform makes it approximately symmetric
    "RILM": "standard",     # log10 pre-transform makes it approximately symmetric
    "NPHI": "standard",     # near-symmetric (mean skew -0.09)
    "SP":   "yeo-johnson",  # extreme skewness in some wells (up to 6.79)
    "DEN":  "yeo-johnson",  # consistently left-skewed (mean skew -1.64)
}

# Minimum number of independent detectors that must agree to flag a value as an outlier.
_OUTLIER_MIN_VOTES: int = 2

# Physical reference bounds — used for the DEN target filter and EDA/crossplot limits.
# Features are no longer clipped to these bounds; per-well statistical consensus
# detection replaces the fixed-clip approach.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "GR":   (0.0, 400.0),       # GAPI
    "RT":   (0.05, 50_000.0),   # Ohm·m
    "RILM": (0.05, 50_000.0),   # Ohm·m
    "NPHI": (-0.15, 0.80),      # v/v
    "SP":   (-1000.0, 1000.0),  # mV
}

# Target bounds — rows OUTSIDE this range are DROPPED (cannot train on bad DEN).
TARGET_BOUNDS: tuple[float, float] = (1.5, 3.1)  # g/cc

# Legacy alias for backwards-compatible imports.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {**FEATURE_BOUNDS, TARGET_COL: TARGET_BOUNDS}


@dataclass
class WellScaler:
    """Per-well normalizer with per-column scaler selection.

    Each column is normalized with the method defined in SCALER_TYPE:
    Yeo-Johnson + z-score (PowerTransformer) for skewed columns, or plain
    z-score (StandardScaler) for approximately symmetric ones. Output is
    approximately N(0, 1) in both cases.

    Args:
        well_id: Identifier for the well.
        transformers: Dict mapping column name → fitted scaler instance.
    """

    well_id: str
    transformers: dict[str, PowerTransformer | StandardScaler] = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Yeo-Johnson + z-score scaling using stored transformers.

        Args:
            df: DataFrame with columns to scale.

        Returns:
            Scaled DataFrame (approximately N(0, 1) for fitted columns).
        """
        out = df.copy()
        for col, pt in self.transformers.items():
            if col not in out.columns:
                continue
            out[col] = pt.transform(out[col].values.reshape(-1, 1)).ravel()
        return out

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Reverse Yeo-Johnson + z-score transform for the DEN target column.

        Args:
            values: Scaled values to invert.

        Returns:
            Values in original g/cc units.
        """
        pt = self.transformers[TARGET_COL]
        return pt.inverse_transform(values.reshape(-1, 1)).ravel()


def _make_scaler(col: str) -> PowerTransformer | StandardScaler:
    """Instantiate the correct scaler type for a column per SCALER_TYPE."""
    if SCALER_TYPE.get(col, "yeo-johnson") == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    return StandardScaler()


def fit_scaler(well_id: str, df: pd.DataFrame, cols: list[str] | None = None) -> WellScaler:
    """Fit per-well scalers using the method defined in SCALER_TYPE for each column.

    Yeo-Johnson + z-score for skewed columns; plain z-score for symmetric ones.
    For near-constant columns (std < 1e-6 or fewer than 10 samples), a synthetic
    two-point symmetric range is used to avoid singular variance.

    Args:
        well_id: Identifier for the well.
        df: DataFrame to fit on (NaNs are dropped per column before fitting).
        cols: Columns to scale. Defaults to ALL_COLS.

    Returns:
        Fitted WellScaler instance.
    """
    cols = cols or ALL_COLS
    scaler = WellScaler(well_id=well_id)
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col].dropna().values.reshape(-1, 1)
        sk = _make_scaler(col)
        if x.shape[0] < 10 or float(x.std()) < 1e-6:
            # Near-constant: fit on synthetic symmetric range → transform(centre) ≈ 0
            centre = float(x.mean()) if x.size > 0 else 0.0
            sk.fit(np.array([[centre - 0.5], [centre + 0.5]]))
        else:
            sk.fit(x)
        scaler.transformers[col] = sk
    return scaler


# ── Outlier detectors ─────────────────────────────────────────────────────────

def _detect_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Flag values whose modified Z-score (MAD-based) exceeds threshold.

    Uses median and MAD instead of mean/std — robust to the outliers it detects.

    Args:
        series: Input series; existing NaN values are never flagged.
        threshold: Modified Z-score cutoff (Iglewicz & Hoaglin recommend 3.5).

    Returns:
        Boolean Series; True where the value is a suspected outlier.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(False, index=series.index)
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad == 0.0:
        return pd.Series(False, index=series.index)
    scores = 0.6745 * (series - median).abs() / mad
    return (scores > threshold).fillna(False)


def _detect_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Flag values beyond Q1 - k*IQR or Q3 + k*IQR (Tukey's fences).

    Args:
        series: Input series; existing NaN values are never flagged.
        k: Fence multiplier; k=1.5 is mild, k=3.0 targets extreme outliers.

    Returns:
        Boolean Series; True where the value is a suspected outlier.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(False, index=series.index)
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        return pd.Series(False, index=series.index)
    return ((series < q1 - k * iqr) | (series > q3 + k * iqr)).fillna(False)


def _detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag values whose standard Z-score exceeds threshold.

    Less robust than MAD detection but contributes a different signal: values
    so extreme they distort even the mean and standard deviation.

    Args:
        series: Input series; existing NaN values are never flagged.
        threshold: Z-score cutoff (3.0 → ~99.7 % of a normal distribution).

    Returns:
        Boolean Series; True where the value is a suspected outlier.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(False, index=series.index)
    mean = float(clean.mean())
    std = float(clean.std())
    if std == 0.0:
        return pd.Series(False, index=series.index)
    scores = (series - mean).abs() / std
    return (scores > threshold).fillna(False)


def _detect_percentile(series: pd.Series, low: float = 1.5, high: float = 98.5) -> pd.Series:
    """Flag values outside the [low, high] percentile range of the well.

    Args:
        series: Input series; existing NaN values are never flagged.
        low: Lower percentile bound (default 1.5).
        high: Upper percentile bound (default 98.5).

    Returns:
        Boolean Series; True where the value is a suspected outlier.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(False, index=series.index)
    lo = float(clean.quantile(low / 100))
    hi = float(clean.quantile(high / 100))
    return ((series < lo) | (series > hi)).fillna(False)


def _detect_isolation_forest(series: pd.Series, contamination: float = 0.05) -> pd.Series:
    """Flag values classified as anomalies by IsolationForest.

    Provides an algorithm-based perspective complementary to the statistical
    detectors. Skipped for series with fewer than 20 non-null values.

    Args:
        series: Input series; existing NaN values are never flagged.
        contamination: Expected fraction of outliers (default 0.05).

    Returns:
        Boolean Series; True where the value is a suspected outlier.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return pd.Series(False, index=series.index)
    X = clean.values.reshape(-1, 1)
    labels = IsolationForest(contamination=contamination, random_state=42).fit_predict(X)
    result = pd.Series(False, index=series.index)
    result.loc[clean.index] = labels == -1
    return result


_Detector = Callable[[pd.Series], pd.Series]

_DETECTORS: list[_Detector] = [
    _detect_mad,
    _detect_iqr,
    _detect_zscore,
    _detect_percentile,
    _detect_isolation_forest,
]


def flag_outliers_consensus(
    series: pd.Series,
    col: str,
    min_votes: int = _OUTLIER_MIN_VOTES,
) -> pd.Series:
    """Return a boolean mask flagging outliers by multi-detector consensus.

    A value is flagged only when at least min_votes independent detectors agree.
    For RT and RILM (log-normal), detection operates on the log10 scale.

    Args:
        series: Raw feature series; NaN-safe (existing NaN are never flagged).
        col: Column name — used to decide whether to apply log10 scale.
        min_votes: Minimum number of detectors that must agree (default 2).

    Returns:
        Boolean Series; True where ≥ min_votes detectors flagged the value.
    """
    x = np.log10(series.clip(lower=1e-6)) if col in LOG_FEATURES else series
    votes: pd.Series = sum(d(x).astype(int) for d in _DETECTORS)  # type: ignore[assignment]
    return votes >= min_votes


# ── Feature cleaning ──────────────────────────────────────────────────────────

def clip_features_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-well statistical outliers from feature columns.

    Replaces fixed physical-bound clipping with a voting consensus approach:
    a value is flagged only if ≥2 out of 5 independent detectors (MAD, IQR,
    Z-score, percentile, Isolation Forest) identify it as an outlier. This
    adapts to each well's own distribution instead of applying global limits.

    Flagged values are set to NaN, then recovered via linear interpolation
    (limit=5 depth samples). Remaining NaN at well edges are filled with
    forward/backward fill. All rows are preserved so the valid DEN target
    can still be used for training.

    For RT and RILM, detection operates on the log10 scale.

    Args:
        df: DataFrame with canonical feature columns in raw units.

    Returns:
        DataFrame copy with outlier feature values removed and interpolated.
    """
    out = df.copy()
    for col in FEATURE_COLS:
        if col not in out.columns:
            continue
        if out[col].dropna().shape[0] < 10:
            continue
        mask = flag_outliers_consensus(out[col], col)
        out.loc[mask, col] = np.nan
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].interpolate(method="linear", limit=5).ffill().bfill()
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
    """Backwards-compatible wrapper: remove feature outliers + NaN bad DEN rows.

    Calls clip_features_to_bounds (voting consensus) then sets out-of-range DEN
    values to NaN without dropping rows, preserving the original index for
    legacy callers.

    Args:
        df: DataFrame with canonical curve columns in raw units.

    Returns:
        DataFrame with feature outliers removed and DEN outliers set to NaN.
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
    # Step 1 — Remove per-well statistical outliers from features (preserve rows)
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
