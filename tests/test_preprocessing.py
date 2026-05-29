"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_COLS,
    FEATURE_COLS,
    TARGET_COL,
    WellScaler,
    apply_log_rt,
    fit_scaler,
    preprocess_well,
    preprocess_wells,
)


# ----------------------------------------
# Fixtures
# ----------------------------------------

def _make_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal DataFrame with all canonical columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "GR":   rng.uniform(10, 150, n).astype(np.float32),
        "RT":   rng.uniform(0.1, 500, n).astype(np.float32),
        "RILM": rng.uniform(0.1, 500, n).astype(np.float32),
        "NPHI": rng.uniform(0.05, 0.45, n).astype(np.float32),
        "SP":   rng.uniform(-80, 20, n).astype(np.float32),
        "DEN":  rng.uniform(2.0, 2.9, n).astype(np.float32),
    })


# ----------------------------------------
# Tests — apply_log_rt
# ----------------------------------------

def test_apply_log_rt_transforms_rt_and_rilm():
    df = _make_df()
    result = apply_log_rt(df)
    assert not df["RT"].equals(result["RT"]), "RT should be transformed"
    assert not df["RILM"].equals(result["RILM"]), "RILM should be transformed"


def test_apply_log_rt_does_not_modify_other_cols():
    df = _make_df()
    result = apply_log_rt(df)
    for col in ("GR", "NPHI", "SP", "DEN"):
        pd.testing.assert_series_equal(df[col], result[col], check_names=True)


def test_apply_log_rt_clips_nonpositive():
    df = _make_df()
    df.loc[0, "RT"] = 0.0
    df.loc[1, "RT"] = -5.0
    result = apply_log_rt(df)
    assert np.isfinite(result["RT"].iloc[0])
    assert np.isfinite(result["RT"].iloc[1])


def test_apply_log_rt_returns_copy():
    df = _make_df()
    result = apply_log_rt(df)
    assert result is not df


# ----------------------------------------
# Tests — fit_scaler / WellScaler.transform
# ----------------------------------------

def test_fit_scaler_stores_all_cols():
    df = _make_df()
    scaler = fit_scaler("well_1", df)
    for col in ALL_COLS:
        assert col in scaler.mins, f"Missing min for {col}"
        assert col in scaler.ranges, f"Missing range for {col}"


def test_fit_scaler_transform_in_unit_range():
    df = _make_df(200)
    scaler = fit_scaler("well_1", df)
    scaled = scaler.transform(df)
    for col in ALL_COLS:
        assert scaled[col].min() >= -1e-6, f"{col} min below 0"
        assert scaled[col].max() <= 1.0 + 1e-6, f"{col} max above 1"


def test_fit_scaler_constant_col_no_div_zero():
    df = _make_df()
    df["GR"] = 50.0
    scaler = fit_scaler("well_1", df)
    scaled = scaler.transform(df)
    assert not scaled["GR"].isna().any()


def test_inverse_transform_target_roundtrips():
    df = _make_df(200)
    scaler = fit_scaler("well_1", df)
    scaled = scaler.transform(df)
    recovered = scaler.inverse_transform_target(scaled["DEN"].values)
    np.testing.assert_allclose(recovered, df["DEN"].values, atol=1e-4)


# ----------------------------------------
# Tests — preprocess_well
# ----------------------------------------

def test_preprocess_well_returns_tuple():
    df = _make_df()
    result = preprocess_well(df, "well_1")
    assert isinstance(result, tuple)
    assert len(result) == 2
    out_df, scaler = result
    assert isinstance(out_df, pd.DataFrame)
    assert isinstance(scaler, WellScaler)


def test_preprocess_well_fit_false_requires_scaler():
    df = _make_df()
    with pytest.raises(ValueError):
        preprocess_well(df, "well_1", fit=False, scaler=None)


def test_preprocess_well_fit_false_uses_provided_scaler():
    df = _make_df()
    _, scaler = preprocess_well(df, "well_1", fit=True)
    out_df2, returned_scaler = preprocess_well(df, "well_1", fit=False, scaler=scaler)
    assert returned_scaler is scaler
    assert out_df2 is not None


def test_preprocess_well_no_nans_output():
    df = _make_df()
    out_df, _ = preprocess_well(df, "well_1")
    assert out_df[ALL_COLS].isna().sum().sum() == 0


# ----------------------------------------
# Tests — preprocess_wells
# ----------------------------------------

def test_preprocess_wells_returns_dicts():
    wells = {"w1": _make_df(), "w2": _make_df()}
    processed, scalers = preprocess_wells(wells)
    assert set(processed) == {"w1", "w2"}
    assert set(scalers) == {"w1", "w2"}


def test_preprocess_wells_independent_scalers():
    rng = np.random.default_rng(0)
    wells = {
        "w1": pd.DataFrame({c: rng.uniform(0, 1, 100) for c in ALL_COLS}),
        "w2": pd.DataFrame({c: rng.uniform(10, 20, 100) for c in ALL_COLS}),
    }
    _, scalers = preprocess_wells(wells)
    assert scalers["w1"].mins["GR"] != scalers["w2"].mins["GR"]
