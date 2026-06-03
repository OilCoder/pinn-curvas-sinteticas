"""
External-validation inference helpers shared across scripts.

Centralizes the two inference protocols used to evaluate the held-out external
wells, so the logic is not duplicated across scripts/08, scripts/09, scripts/10:

  - Single final model: train one MLP on the whole train pool, predict a well.
  - LOWO ensemble: average the predictions of the 27 per-fold checkpoints.

Both protocols preprocess every well with its OWN per-well scaler (no cross-well
leakage), matching the LOWO benchmark.

Called by: scripts/08_eval_external.py, scripts/09_plot_diagnostics.py,
scripts/10_external_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import WellDataset
from src.lowo import field_split
from src.model import MLP
from src.preprocessing import preprocess_well
from src.train import TrainConfig, predict, set_seed, train_model

# Inference TrainConfig: only device/batch_size matter for predict().
_INFER_CFG = TrainConfig(epochs=1)


def predict_well(
    model: MLP,
    raw_df: pd.DataFrame,
    well_id: str,
    cfg: TrainConfig = _INFER_CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict DEN on a single well with a trained model, in g/cc.

    The well is preprocessed with its own per-well scaler and the prediction is
    inverse-transformed back to physical units.

    Args:
        model: Trained MLP.
        raw_df: Raw well DataFrame (canonical columns, pre-preprocessing).
        well_id: Identifier for the well (used by its scaler).
        cfg: Training/inference config (device, batch size).

    Returns:
        Tuple of (depth, den_true_gcc, den_pred_gcc), each shape (n,).
    """
    df_proc, scaler = preprocess_well(raw_df.copy(), well_id, fit=True)
    dataset = WellDataset(df_proc)
    pred = scaler.inverse_transform_target(predict(model, dataset, cfg))
    true = scaler.inverse_transform_target(df_proc["DEN"].values)
    depth = (
        raw_df["DEPTH"].values[: len(true)]
        if "DEPTH" in raw_df.columns
        else np.arange(len(true)) * 0.5
    )
    return depth, true, pred


def ensemble_predict(
    raw_df: pd.DataFrame,
    well_id: str,
    checkpoint_dir: Path,
    valid_ids: set[str],
    cfg: TrainConfig = _INFER_CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict DEN on a well by averaging all valid LOWO checkpoints, in g/cc.

    Loads every ``*_best.pt`` in ``checkpoint_dir`` whose stem corresponds to a
    train-pool well, predicts with each, and averages the predictions before the
    inverse transform.

    Args:
        raw_df: Raw well DataFrame (canonical columns).
        well_id: Identifier for the well (used by its scaler).
        checkpoint_dir: Directory containing ``*_best.pt`` LOWO checkpoints.
        valid_ids: Set of train-pool well ids; checkpoints outside this set are skipped.
        cfg: Training/inference config (device, batch size).

    Returns:
        Tuple of (depth, den_true_gcc, den_pred_gcc), each shape (n,).

    Raises:
        FileNotFoundError: If no valid checkpoint is found in checkpoint_dir.
    """
    df_proc, scaler = preprocess_well(raw_df.copy(), well_id, fit=True)
    dataset = WellDataset(df_proc)

    ckpts = [
        c for c in sorted(checkpoint_dir.glob("*_best.pt"))
        if c.stem.replace("_best", "") in valid_ids
    ]
    if not ckpts:
        raise FileNotFoundError(f"No valid checkpoints in {checkpoint_dir}")

    preds_norm = []
    for ckpt in ckpts:
        model = MLP()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        preds_norm.append(predict(model, dataset, cfg))
    ensemble_norm = np.mean(preds_norm, axis=0)

    pred = scaler.inverse_transform_target(ensemble_norm)
    true = scaler.inverse_transform_target(df_proc["DEN"].values)
    depth = (
        raw_df["DEPTH"].values[: len(true)]
        if "DEPTH" in raw_df.columns
        else np.arange(len(true)) * 0.5
    )
    return depth, true, pred


def train_final_model(
    train_pool: dict[str, pd.DataFrame],
    cfg: TrainConfig,
    well_id: str = "final",
) -> MLP:
    """Train a single final MLP on the entire train pool.

    Each well is preprocessed with its own per-well scaler and all wells are
    concatenated into one dataset. This is the "single final model" protocol
    (vs the LOWO ensemble) for evaluating external wells.

    Args:
        train_pool: Dict mapping well_id to raw DataFrame (all training wells).
        cfg: Training configuration (epochs, lambda_phys, checkpoint_dir, etc.).
        well_id: Label used for the saved checkpoint filename.

    Returns:
        The trained MLP (best weights restored).
    """
    # Substep 1 — Preprocess every training well independently
    train_dfs = [preprocess_well(df.copy(), wid, fit=True)[0] for wid, df in train_pool.items()]
    dataset = WellDataset(train_dfs)

    # Substep 2 — Train one model on the concatenated pool
    set_seed(cfg.seed)
    model = MLP()
    train_model(model, dataset, cfg, well_id=well_id)
    return model


def train_pool_ids(field_dir_wells: dict[str, pd.DataFrame], n_external: int = 3, seed: int = 42) -> set[str]:
    """Return the set of train-pool well ids for checkpoint filtering.

    Args:
        field_dir_wells: All loaded wells (train pool + external).
        n_external: Number of external wells reserved.
        seed: Split seed.

    Returns:
        Set of well ids in the train pool.
    """
    pool, _ = field_split(field_dir_wells, n_external=n_external, seed=seed)
    return set(pool.keys())
