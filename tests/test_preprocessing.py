"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_COLS,
    FEATURE_BOUNDS,
    FEATURE_COLS,
    PHYSICAL_BOUNDS,
    TARGET_BOUNDS,
    TARGET_COL,
    WellScaler,
    apply_log_rt,
    clip_features_to_bounds,
    clip_physical_ranges,
    filter_invalid_target_rows,
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


# ----------------------------------------
# Tests — clip_features_to_bounds
# ----------------------------------------

def test_clip_features_clamps_spikes_to_boundary():
    """Feature outliers must be clipped to the boundary, NOT NaN'd."""
    df = _make_df()
    df.loc[0, "GR"] = 9999.0          # spike well above bound
    df.loc[1, "RT"] = 1e10            # huge sentinel
    df.loc[2, "NPHI"] = 5.0           # residual % issue
    result = clip_features_to_bounds(df)
    assert result.loc[0, "GR"] == FEATURE_BOUNDS["GR"][1]
    assert result.loc[1, "RT"] == FEATURE_BOUNDS["RT"][1]
    assert result.loc[2, "NPHI"] == FEATURE_BOUNDS["NPHI"][1]


def test_clip_features_preserves_row_count():
    df = _make_df(50)
    df.loc[0, "GR"] = 9999.0
    df.loc[5, "RT"] = 1e10
    result = clip_features_to_bounds(df)
    assert len(result) == 50, "clip must not drop rows"


def test_clip_features_does_not_touch_target():
    df = _make_df()
    df.loc[0, "DEN"] = 99.0   # impossible DEN must NOT be clipped here
    result = clip_features_to_bounds(df)
    assert result.loc[0, "DEN"] == 99.0


def test_clip_features_returns_copy():
    df = _make_df()
    df.loc[0, "GR"] = 9999.0
    result = clip_features_to_bounds(df)
    assert result is not df
    assert df.loc[0, "GR"] == 9999.0


# ----------------------------------------
# Tests — filter_invalid_target_rows
# ----------------------------------------

def test_filter_target_drops_below_min():
    df = _make_df(100).astype(float)
    df.loc[0, "DEN"] = 0.5            # below 1.5
    result = filter_invalid_target_rows(df)
    assert len(result) == 99


def test_filter_target_drops_above_max():
    df = _make_df(100).astype(float)
    df.loc[0, "DEN"] = 4.0            # above 3.1
    result = filter_invalid_target_rows(df)
    assert len(result) == 99


def test_filter_target_keeps_boundary_values():
    df = _make_df(3).astype(float)
    df.loc[0, "DEN"] = TARGET_BOUNDS[0]   # exactly lower bound
    df.loc[1, "DEN"] = TARGET_BOUNDS[1]   # exactly upper bound
    df.loc[2, "DEN"] = 2.5                # middle
    result = filter_invalid_target_rows(df)
    assert len(result) == 3


# ----------------------------------------
# Tests — preprocess_well end-to-end
# ----------------------------------------

def test_preprocess_well_drops_only_bad_targets():
    """Feature spikes get clipped (row kept); bad DEN drops the row."""
    df = _make_df(110).astype(float)
    df.loc[0, "DEN"] = 0.1            # invalid target → drop
    df.loc[1, "GR"] = 500.0           # feature spike → clipped, row kept
    df.loc[2, "RT"] = 1e10            # huge sentinel → clipped, row kept
    out_df, _ = preprocess_well(df, "well_1")
    assert len(out_df) == 109, "only the row with DEN=0.1 should drop"


def test_preprocess_well_no_nans_after_pipeline():
    df = _make_df(100)
    out_df, _ = preprocess_well(df, "well_1")
    assert out_df[ALL_COLS].isna().sum().sum() == 0


# ----------------------------------------
# Tests — backwards-compatible clip_physical_ranges
# ----------------------------------------

def test_clip_physical_ranges_compat_clips_features():
    df = _make_df()
    df.loc[0, "GR"] = 9999.0
    result = clip_physical_ranges(df)
    assert result.loc[0, "GR"] == FEATURE_BOUNDS["GR"][1]


def test_clip_physical_ranges_compat_nans_bad_target():
    df = _make_df().astype(float)
    df.loc[0, "DEN"] = 0.1
    result = clip_physical_ranges(df)
    assert np.isnan(result.loc[0, "DEN"])


def test_physical_bounds_covers_all_columns():
    assert set(PHYSICAL_BOUNDS.keys()) == set(ALL_COLS)
    assert set(FEATURE_BOUNDS.keys()) == set(FEATURE_COLS)
    assert TARGET_COL not in FEATURE_BOUNDS
