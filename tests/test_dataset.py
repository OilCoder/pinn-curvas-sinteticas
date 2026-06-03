"""Tests for src/dataset.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import WellDataset
from src.preprocessing import FEATURE_COLS, TARGET_COL


# ----------------------------------------
# Fixtures
# ----------------------------------------

def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "GR":   rng.uniform(0, 1, n).astype(np.float32),
        "RT":   rng.uniform(0, 1, n).astype(np.float32),
        "RILM": rng.uniform(0, 1, n).astype(np.float32),
        "NPHI": rng.uniform(0, 1, n).astype(np.float32),
        "SP":   rng.uniform(0, 1, n).astype(np.float32),
        "DEN":  rng.uniform(0, 1, n).astype(np.float32),
    })


# ----------------------------------------
# Tests — construction
# ----------------------------------------

def test_dataset_from_single_df():
    df = _make_df(50)
    ds = WellDataset(df)
    assert len(ds) == 50


def test_dataset_from_list_of_dfs():
    dfs = [_make_df(30), _make_df(40)]
    ds = WellDataset(dfs)
    assert len(ds) == 70


def test_dataset_missing_column_raises():
    df = _make_df().drop(columns=["DEN"])
    with pytest.raises(ValueError, match="Missing columns"):
        WellDataset(df)


def test_dataset_missing_feature_raises():
    df = _make_df().drop(columns=["GR"])
    with pytest.raises(ValueError, match="Missing columns"):
        WellDataset(df)


# ----------------------------------------
# Tests — shapes and dtypes
# ----------------------------------------

def test_dataset_getitem_shapes():
    df = _make_df(50)
    ds = WellDataset(df)
    x, y, w = ds[0]
    assert x.shape == (5,), f"Expected (5,), got {x.shape}"
    assert y.shape == (1,), f"Expected (1,), got {y.shape}"
    assert w.shape == (), f"Expected scalar, got {w.shape}"


def test_dataset_tensors_are_float32():
    df = _make_df(50)
    ds = WellDataset(df)
    x, y, w = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert w.dtype == torch.float32


def test_dataset_weights_ones_when_no_dcal():
    df = _make_df(50)
    ds = WellDataset(df)
    assert torch.allclose(ds.weights, torch.ones(50))


def test_dataset_weights_from_dcal_weight_col():
    df = _make_df(50)
    df["DCAL_WEIGHT"] = np.linspace(0.5, 1.0, 50).astype(np.float32)
    ds = WellDataset(df)
    assert ds.weights[0].item() == pytest.approx(0.5, abs=1e-5)


def test_dataset_n_features():
    df = _make_df(50)
    ds = WellDataset(df)
    assert ds.n_features == len(FEATURE_COLS)


# ----------------------------------------
# Tests — values
# ----------------------------------------

def test_dataset_values_match_source():
    df = _make_df(20)
    ds = WellDataset(df)
    for i in range(len(ds)):
        x, y, _ = ds[i]
        expected_x = torch.tensor(df[FEATURE_COLS].iloc[i].values, dtype=torch.float32)
        expected_y = torch.tensor([df[TARGET_COL].iloc[i]], dtype=torch.float32)
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(y, expected_y)


def test_dataset_custom_feature_cols():
    df = _make_df(20)
    custom_features = ["GR", "NPHI"]
    ds = WellDataset(df, feature_cols=custom_features)
    x, y, _ = ds[0]
    assert x.shape == (2,)
    assert ds.n_features == 2


def test_dataset_len_consistent_with_x():
    df = _make_df(33)
    ds = WellDataset(df)
    assert len(ds) == ds.X.shape[0]
