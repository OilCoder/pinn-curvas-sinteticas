"""Tests for src/features_caliper.py."""

import numpy as np
import pandas as pd

from src.features_caliper import (
    CALIPER_COL,
    FEATURE_COLS_EXT,
    build_arrays,
    caliper_zscore,
    preprocess_well_with_caliper,
)
from src.preprocessing import FEATURE_COLS


# ----------------------------------------
# Fixtures
# ----------------------------------------


def _make_df(n: int = 200, with_dcal: bool = True) -> pd.DataFrame:
    """Minimal raw well DataFrame with canonical columns (+ optional DCAL)."""
    rng = np.random.default_rng(42)
    data = {
        "DEPTH": np.arange(n, dtype=np.float32) * 0.5,
        "GR": rng.uniform(10, 150, n).astype(np.float32),
        "RT": rng.uniform(0.1, 500, n).astype(np.float32),
        "RILM": rng.uniform(0.1, 500, n).astype(np.float32),
        "NPHI": rng.uniform(0.05, 0.45, n).astype(np.float32),
        "SP": rng.uniform(-80, 20, n).astype(np.float32),
        "DEN": rng.uniform(2.0, 2.9, n).astype(np.float32),
    }
    if with_dcal:
        data["DCAL"] = rng.uniform(8.0, 12.0, n).astype(np.float32)
    return pd.DataFrame(data)


# ----------------------------------------
# Tests — caliper_zscore
# ----------------------------------------


def test_caliper_zscore_is_standardized() -> None:
    """DCAL_NORM has approximately zero mean and unit std per well."""
    df = _make_df()
    z = caliper_zscore(df)
    assert z.shape == (len(df),)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std(ddof=1)) - 1.0) < 1e-2


def test_caliper_zscore_zeros_when_no_dcal() -> None:
    """A well without DCAL yields a neutral zero column."""
    df = _make_df(with_dcal=False)
    z = caliper_zscore(df)
    assert np.allclose(z, 0.0)


def test_caliper_zscore_fills_missing_with_zero() -> None:
    """NaN caliper samples map to 0 (neutral), not NaN."""
    df = _make_df()
    df.loc[5:10, "DCAL"] = np.nan
    z = caliper_zscore(df)
    assert not np.isnan(z).any()
    assert np.allclose(z[5:11], 0.0)


# ----------------------------------------
# Tests — preprocess_well_with_caliper
# ----------------------------------------


def test_preprocess_appends_caliper_aligned() -> None:
    """DCAL_NORM is appended and aligned to the processed row count."""
    df = _make_df()
    proc, _ = preprocess_well_with_caliper(df, "w1", fit=True)
    assert CALIPER_COL in proc.columns
    assert len(proc[CALIPER_COL]) == len(proc)
    assert not proc[CALIPER_COL].isna().any()


# ----------------------------------------
# Tests — build_arrays
# ----------------------------------------


def test_build_arrays_with_caliper_has_six_features() -> None:
    """With caliper, X has 6 columns and matches FEATURE_COLS_EXT order."""
    df = _make_df()
    proc, _ = preprocess_well_with_caliper(df, "w1", fit=True)
    x, y, nphi, gr, w, depth = build_arrays(proc, with_caliper=True)
    assert x.shape[1] == len(FEATURE_COLS_EXT) == 6
    assert y.shape == nphi.shape == gr.shape == w.shape == depth.shape == (len(proc),)
    # GR is the first column, caliper the last.
    assert np.allclose(x[:, 0], gr)
    assert np.allclose(x[:, -1], proc[CALIPER_COL].to_numpy(np.float32))


def test_build_arrays_without_caliper_reproduces_five_features() -> None:
    """with_caliper=False yields exactly the 5 published features."""
    df = _make_df()
    proc, _ = preprocess_well_with_caliper(df, "w1", fit=True)
    x5, *_ = build_arrays(proc, with_caliper=False)
    assert x5.shape[1] == len(FEATURE_COLS) == 5
    assert np.allclose(x5, proc[FEATURE_COLS].to_numpy(np.float32))
