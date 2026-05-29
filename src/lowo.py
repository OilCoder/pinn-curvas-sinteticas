"""
Leave-One-Well-Out (LOWO) split generator and field-level train/external split.

Each LOWO fold uses all wells except one for training and the held-out well
for testing. No depth-level data leaks between wells.

field_split() reserves a fixed external validation set that is never used
during LOWO training or hyperparameter selection.

Called by: scripts/03_train_baseline.py, scripts/04_train_pinn.py
"""

import random
from collections.abc import Iterator
from typing import TypeVar

import pandas as pd

T = TypeVar("T")


def lowo_splits(
    wells: dict[str, T],
) -> Iterator[tuple[dict[str, T], str, T]]:
    """Generate Leave-One-Well-Out splits.

    Args:
        wells: Dict mapping well_id to any per-well object (DataFrame,
               preprocessed data, etc.).

    Yields:
        Tuple of (train_wells, test_well_id, test_well) for each fold.
        train_wells contains all wells except the test well.
    """
    well_ids = list(wells.keys())
    for test_id in well_ids:
        train = {wid: data for wid, data in wells.items() if wid != test_id}
        yield train, test_id, wells[test_id]


def lowo_dataframes(
    wells: dict[str, pd.DataFrame],
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Generate LOWO splits as concatenated train/test DataFrames.

    Convenience wrapper over lowo_splits for use with sklearn-style APIs.

    Args:
        wells: Dict mapping well_id to preprocessed DataFrame.

    Yields:
        Tuple of (train_df, test_df, test_well_id) for each fold.
        train_df is the concatenation of all training wells.
    """
    for train_wells, test_id, test_df in lowo_splits(wells):
        train_df = pd.concat(list(train_wells.values()), ignore_index=True)
        yield train_df, test_df, test_id


def field_split(
    wells: dict[str, T],
    n_external: int = 3,
    seed: int = 42,
) -> tuple[dict[str, T], dict[str, T]]:
    """Randomly split wells into a training pool and an external validation set.

    The external set is held out entirely from LOWO training and
    hyperparameter selection. It is used only for final unbiased evaluation.

    Args:
        wells: Dict mapping well_id to any per-well object.
        n_external: Number of wells to reserve for external validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pool, external_set), both as dicts with the same
        value type as the input.

    Raises:
        ValueError: If n_external >= len(wells) or n_external < 1.
    """
    if n_external < 1 or n_external >= len(wells):
        raise ValueError(
            f"n_external must be in [1, {len(wells) - 1}], got {n_external}"
        )

    rng = random.Random(seed)
    all_ids = sorted(wells.keys())
    external_ids = set(rng.sample(all_ids, n_external))

    train_pool = {wid: data for wid, data in wells.items() if wid not in external_ids}
    external_set = {wid: data for wid, data in wells.items() if wid in external_ids}

    return train_pool, external_set
