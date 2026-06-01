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
    _detect_iqr,
    _detect_isolation_forest,
    _detect_mad,
    _detect_percentile,
    _detect_zscore,
    apply_log_rt,
    clip_features_to_bounds,
    clip_physical_ranges,
    filter_invalid_target_rows,
    fit_scaler,
    flag_outliers_consensus,
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


def _make_series_with_spike(n: int = 100, spike_value: float = 1e6, idx: int = 50) -> pd.Series:
    """Uniform series with one injected extreme spike."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(10.0, 100.0, n))
    s.iloc[idx] = spike_value
    return s


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
# Tests — individual outlier detectors
# ----------------------------------------

def test_detect_mad_flags_extreme_spike():
    s = _make_series_with_spike()
    assert _detect_mad(s).iloc[50], "MAD must flag the extreme spike"


def test_detect_mad_does_not_flag_normal_values():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.uniform(10.0, 100.0, 100))
    assert _detect_mad(s).sum() == 0, "no flags expected on clean uniform data"


def test_detect_iqr_flags_extreme_spike():
    s = _make_series_with_spike()
    assert _detect_iqr(s).iloc[50], "IQR must flag the extreme spike"


def test_detect_zscore_flags_extreme_spike():
    s = _make_series_with_spike()
    assert _detect_zscore(s).iloc[50], "Z-score must flag the extreme spike"


def test_detect_percentile_flags_extreme_spike():
    s = _make_series_with_spike()
    assert _detect_percentile(s).iloc[50], "Percentile detector must flag the extreme spike"


def test_detect_isolation_forest_flags_extreme_spike():
    s = _make_series_with_spike(n=100)
    assert _detect_isolation_forest(s).iloc[50], "IsolationForest must flag the extreme spike"


def test_detect_isolation_forest_skips_small_series():
    s = pd.Series([1.0, 2.0, 1e6])  # fewer than 20 samples
    result = _detect_isolation_forest(s)
    assert not result.any(), "IsolationForest must abstain on small series"


def test_detectors_return_false_for_existing_nan():
    s = pd.Series([np.nan, 50.0, 60.0, 70.0, 80.0])
    for fn in (_detect_mad, _detect_iqr, _detect_zscore, _detect_percentile):
        assert not fn(s).iloc[0], f"{fn.__name__} must not flag existing NaN"


# ----------------------------------------
# Tests — flag_outliers_consensus
# ----------------------------------------

def test_consensus_flags_when_at_least_two_agree():
    s = _make_series_with_spike()
    mask = flag_outliers_consensus(s, col="GR")
    assert mask.iloc[50], "consensus must flag a value detected by all detectors"


def test_consensus_uses_log_scale_for_rt():
    """An RT spike that is extreme on log scale should be flagged."""
    rng = np.random.default_rng(2)
    s = pd.Series(rng.uniform(1.0, 10.0, 100))  # normal RT 1–10 Ohm·m
    s.iloc[50] = 1e8                              # extreme sentinel
    mask = flag_outliers_consensus(s, col="RT")
    assert mask.iloc[50], "RT spike must be flagged on log10 scale"


def test_consensus_does_not_flag_mild_deviation():
    # 1.5-sigma in a normal(100, 20) distribution: below z<3, MAD-score<3.5,
    # within IQR fences (k=3), below p98.5, and below the top-5% IsolationForest level.
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(100.0, 20.0, 200))
    s.iloc[50] = 130.0  # 1.5-sigma: clearly within all thresholds
    mask = flag_outliers_consensus(s, col="GR")
    assert not mask.iloc[50], "1.5-sigma deviation must not reach 2-vote threshold"


def test_consensus_preserves_nan_positions():
    s = _make_series_with_spike()
    s.iloc[0] = np.nan
    mask = flag_outliers_consensus(s, col="GR")
    assert not mask.iloc[0], "existing NaN must not be flagged as outlier"


# ----------------------------------------
# Tests — clip_features_to_bounds
# ----------------------------------------

def test_clip_features_extreme_spike_is_removed():
    """An extreme spike must no longer hold its original value after cleaning."""
    df = _make_df(60)
    df = df.astype(float)
    df.loc[30, "GR"] = 9999.0  # middle of the well so interpolation fills both ways
    result = clip_features_to_bounds(df)
    assert result.loc[30, "GR"] != 9999.0, "extreme spike must be removed"
    assert not pd.isna(result.loc[30, "GR"]), "interpolation must fill the NaN"


def test_clip_features_preserves_row_count():
    df = _make_df(50)
    df = df.astype(float)
    df.loc[0, "GR"] = 9999.0
    df.loc[5, "RT"] = 1e10
    result = clip_features_to_bounds(df)
    assert len(result) == 50, "outlier removal must not drop rows"


def test_clip_features_does_not_touch_target():
    df = _make_df()
    df = df.astype(float)
    df.loc[0, "DEN"] = 99.0   # impossible DEN must NOT be touched here
    result = clip_features_to_bounds(df)
    assert result.loc[0, "DEN"] == 99.0


def test_clip_features_returns_copy():
    df = _make_df()
    df = df.astype(float)
    df.loc[25, "GR"] = 9999.0
    result = clip_features_to_bounds(df)
    assert result is not df
    assert df.loc[25, "GR"] == 9999.0


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

def test_clip_physical_ranges_compat_removes_extreme_feature():
    df = _make_df(60).astype(float)
    df.loc[30, "GR"] = 9999.0  # middle of the well so it can be interpolated
    result = clip_physical_ranges(df)
    assert result.loc[30, "GR"] != 9999.0, "extreme spike must be removed by consensus detectors"


def test_clip_physical_ranges_compat_nans_bad_target():
    df = _make_df().astype(float)
    df.loc[0, "DEN"] = 0.1
    result = clip_physical_ranges(df)
    assert np.isnan(result.loc[0, "DEN"])


def test_physical_bounds_covers_all_columns():
    assert set(PHYSICAL_BOUNDS.keys()) == set(ALL_COLS)
    assert set(FEATURE_BOUNDS.keys()) == set(FEATURE_COLS)
    assert TARGET_COL not in FEATURE_BOUNDS
