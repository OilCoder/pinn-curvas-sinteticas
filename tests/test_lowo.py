"""Tests for src/lowo.py."""

import pandas as pd
import pytest

from src.lowo import field_split, lowo_dataframes, lowo_splits


# ----------------------------------------
# Fixtures
# ----------------------------------------

def _make_wells(n: int = 4) -> dict[str, pd.DataFrame]:
    return {
        f"well_{i}": pd.DataFrame({"x": list(range(i * 10, i * 10 + 10))})
        for i in range(n)
    }


# ----------------------------------------
# Tests — lowo_splits
# ----------------------------------------

def test_lowo_splits_yields_n_folds():
    wells = _make_wells(4)
    folds = list(lowo_splits(wells))
    assert len(folds) == 4


def test_lowo_splits_each_well_is_test_exactly_once():
    wells = _make_wells(5)
    test_ids = [test_id for _, test_id, _ in lowo_splits(wells)]
    assert sorted(test_ids) == sorted(wells.keys())


def test_lowo_splits_test_well_not_in_train():
    wells = _make_wells(5)
    for train_wells, test_id, _ in lowo_splits(wells):
        assert test_id not in train_wells


def test_lowo_splits_train_has_n_minus_one_wells():
    wells = _make_wells(5)
    for train_wells, _, _ in lowo_splits(wells):
        assert len(train_wells) == 4


def test_lowo_splits_test_data_matches_original():
    wells = _make_wells(4)
    for _, test_id, test_data in lowo_splits(wells):
        pd.testing.assert_frame_equal(test_data, wells[test_id])


def test_lowo_splits_single_well_yields_empty_train():
    wells = {"only": pd.DataFrame({"x": [1, 2, 3]})}
    folds = list(lowo_splits(wells))
    assert len(folds) == 1
    train_wells, test_id, _ = folds[0]
    assert test_id == "only"
    assert len(train_wells) == 0


def test_lowo_splits_two_wells():
    wells = _make_wells(2)
    folds = list(lowo_splits(wells))
    assert len(folds) == 2
    for train_wells, test_id, _ in folds:
        assert len(train_wells) == 1
        assert test_id not in train_wells


# ----------------------------------------
# Tests — lowo_dataframes
# ----------------------------------------

def _make_df_wells(n: int = 4) -> dict[str, pd.DataFrame]:
    """Wells with a well_id column to trace origin after concatenation."""
    return {
        f"well_{i}": pd.DataFrame({
            "GR":  [float(i * 10 + j) for j in range(20)],
            "DEN": [float(2.0 + i * 0.1) for _ in range(20)],
            "_src": [f"well_{i}"] * 20,
        })
        for i in range(n)
    }


def test_lowo_dataframes_yields_n_folds():
    wells = _make_df_wells(4)
    folds = list(lowo_dataframes(wells))
    assert len(folds) == 4


def test_lowo_dataframes_test_id_in_output():
    wells = _make_df_wells(4)
    test_ids = [tid for _, _, tid in lowo_dataframes(wells)]
    assert sorted(test_ids) == sorted(wells.keys())


def test_lowo_dataframes_train_contains_all_other_wells():
    wells = _make_df_wells(4)
    for train_df, _, test_id in lowo_dataframes(wells):
        train_sources = set(train_df["_src"].unique())
        assert test_id not in train_sources
        expected_sources = {wid for wid in wells if wid != test_id}
        assert train_sources == expected_sources


def test_lowo_dataframes_train_row_count():
    wells = _make_df_wells(4)
    total = sum(len(df) for df in wells.values())
    for train_df, test_df, test_id in lowo_dataframes(wells):
        test_rows = len(wells[test_id])
        assert len(train_df) == total - test_rows
        assert len(test_df) == test_rows


# ----------------------------------------
# Tests — field_split
# ----------------------------------------

def test_field_split_sizes():
    wells = _make_wells(10)
    train, external = field_split(wells, n_external=3)
    assert len(train) == 7
    assert len(external) == 3


def test_field_split_no_overlap():
    wells = _make_wells(10)
    train, external = field_split(wells, n_external=3)
    assert set(train) & set(external) == set()


def test_field_split_covers_all_wells():
    wells = _make_wells(10)
    train, external = field_split(wells, n_external=3)
    assert set(train) | set(external) == set(wells)


def test_field_split_reproducible():
    wells = _make_wells(25)
    _, ext1 = field_split(wells, n_external=3, seed=42)
    _, ext2 = field_split(wells, n_external=3, seed=42)
    assert set(ext1) == set(ext2)


def test_field_split_different_seeds_differ():
    wells = _make_wells(25)
    _, ext1 = field_split(wells, n_external=3, seed=42)
    _, ext2 = field_split(wells, n_external=3, seed=99)
    assert set(ext1) != set(ext2)


def test_field_split_invalid_n_external_raises():
    wells = _make_wells(5)
    with pytest.raises(ValueError):
        field_split(wells, n_external=5)
    with pytest.raises(ValueError):
        field_split(wells, n_external=0)
