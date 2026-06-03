"""
PyTorch Dataset for well log data.

WellDataset takes a preprocessed DataFrame (or list of DataFrames) and
returns (features, target, weight) tensor triples suitable for DataLoader.

Input features:  [GR, RT, RILM, NPHI, SP]
Target:          [DEN]
Weight:          DCAL_WEIGHT if present, else ones (used to weight physics loss)

Called by: src/train.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import FEATURE_COLS, TARGET_COL

_WEIGHT_COL = "DCAL_WEIGHT"


class WellDataset(Dataset):
    """Depth-point dataset from one or more preprocessed well DataFrames.

    Args:
        data: Single DataFrame or list of DataFrames with canonical columns.
              All DataFrames must be already preprocessed (normalized, log-RT).
        feature_cols: Input feature column names. Defaults to FEATURE_COLS.
        target_col: Target column name. Defaults to TARGET_COL.
    """

    def __init__(
        self,
        data: pd.DataFrame | list[pd.DataFrame],
        feature_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
    ) -> None:
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col = target_col

        if isinstance(data, list):
            df = pd.concat(data, ignore_index=True)
        else:
            df = data.reset_index(drop=True)

        missing = [c for c in self.feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        x = df[self.feature_cols].values.astype(np.float32)
        y = df[[target_col]].values.astype(np.float32)

        if _WEIGHT_COL in df.columns:
            w = df[_WEIGHT_COL].values.astype(np.float32)
        else:
            w = np.ones(len(df), dtype=np.float32)

        self.X = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.weights = torch.from_numpy(w)

    def __len__(self) -> int:
        """Return number of depth points."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (features, target, weight) tensors for a single depth point.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Tuple of (features tensor [n_features], target tensor [1],
                      caliper quality weight scalar []).
        """
        return self.X[idx], self.y[idx], self.weights[idx]

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.X.shape[1]
