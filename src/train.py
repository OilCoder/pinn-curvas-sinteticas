"""
Training loop for MLP and PINN models.

Implements Adam optimization with MSE loss, early stopping on held-out
validation loss, and optional physics regularization (lambda_phys=0.0
reproduces the pure supervised MLP baseline exactly).

The full per-fold dataset (~3 MB) is moved to the device once and batches
are indexed in-place on the GPU — no per-batch host→device copies and no
DataLoader overhead. This keeps the GPU fed for the tiny MLP, where the
per-batch transfer cost otherwise dominates wall-clock time.

Called by: scripts/03_train_baseline.py, scripts/04_train_pinn.py
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataset import WellDataset
from src.model import MLP
from src.physics import physics_loss as _physics_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters and training settings.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Samples per gradient update.
        lr: Adam learning rate.
        patience: Early stopping — epochs without sufficient improvement.
        min_delta: Minimum val-loss drop to reset patience counter.
        val_fraction: Fraction of training data reserved for early-stopping validation.
        lambda_phys: Physics loss weight (0 = pure supervised MLP).
        seed: Random seed for weights, data split, and per-epoch batch shuffling.
        device: PyTorch device string ('cuda' or 'cpu').
        checkpoint_dir: Directory to save per-fold best-model checkpoints.
    """

    epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 30
    min_delta: float = 1e-5
    val_fraction: float = 0.15
    lambda_phys: float = 0.0
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed value applied to Python random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: MLP,
    dataset: WellDataset,
    cfg: TrainConfig,
    well_id: str = "well",
) -> dict[str, list[float]]:
    """Train model with Adam + MSE and early stopping on an internal val split.

    The internal val split is used only for early stopping — it is a random
    subset of the training folds and is not the LOWO test well.

    Args:
        model: MLP instance (modified in-place; best weights restored on exit).
        dataset: Full dataset for this fold (train + internal val split here).
        cfg: Training configuration.
        well_id: Label used for checkpoint filename.

    Returns:
        Dict with "train_loss" and "val_loss" (per-epoch float lists).
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    # ----------------------------------------
    # Step 1 — Move full fold dataset to device once (GPU-resident)
    # ----------------------------------------
    x_all = dataset.X.to(device)
    y_all = dataset.y.to(device)
    w_all = dataset.weights.to(device)
    n_total = x_all.shape[0]

    # ----------------------------------------
    # Step 2 — Internal train/val split for early stopping (seeded, reproducible)
    # ----------------------------------------
    n_val = max(1, int(n_total * cfg.val_fraction))
    n_train = n_total - n_val
    split_perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(cfg.seed)
    ).to(device)
    train_idx = split_perm[:n_train]
    val_idx = split_perm[n_train:]

    # ----------------------------------------
    # Step 3 — Optimizer and loss criterion
    # ----------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    # ----------------------------------------
    # Step 4 — Training loop with early stopping
    # ----------------------------------------
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / f"{well_id}_best.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    train_history: list[float] = []
    val_history: list[float] = []
    bs = cfg.batch_size
    vbs = cfg.batch_size * 4

    for epoch in range(cfg.epochs):
        # — Train pass: shuffle indices on-device, slice batches in-place
        model.train()
        epoch_idx = train_idx[torch.randperm(n_train, device=device)]
        train_loss_sum = torch.zeros((), device=device)
        for start in range(0, n_train, bs):
            bidx = epoch_idx[start : start + bs]
            xb, yb, wb = x_all[bidx], y_all[bidx], w_all[bidx]
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = mse(y_pred, yb)
            if cfg.lambda_phys > 0.0:
                nphi_batch = xb[:, 3]  # NPHI is feature index 3
                gr_batch = xb[:, 0]    # GR is feature index 0
                loss = loss + cfg.lambda_phys * _physics_loss(y_pred, nphi_batch, gr_batch, wb)
            loss.backward()
            optimizer.step()
            train_loss_sum = train_loss_sum + loss.detach() * xb.shape[0]
        train_history.append((train_loss_sum / n_train).item())

        # — Validation pass
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        with torch.no_grad():
            for start in range(0, n_val, vbs):
                bidx = val_idx[start : start + vbs]
                xb, yb = x_all[bidx], y_all[bidx]
                val_loss_sum = val_loss_sum + mse(model(xb), yb) * xb.shape[0]
        val_loss = (val_loss_sum / n_val).item()
        val_history.append(val_loss)

        # — Early stopping check
        if best_val_loss - val_loss > cfg.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.debug("Early stop at epoch %d for %s", epoch + 1, well_id)
                break

    # ----------------------------------------
    # Step 4 — Restore best checkpoint
    # ----------------------------------------
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    return {"train_loss": train_history, "val_loss": val_history}


def predict(
    model: MLP,
    dataset: WellDataset,
    cfg: TrainConfig,
) -> np.ndarray:
    """Run inference on a dataset and return predictions as a numpy array.

    Args:
        model: Trained MLP (should have best weights already loaded).
        dataset: Dataset to predict on.
        cfg: Training configuration (used for device and batch_size).

    Returns:
        Numpy array of shape (n_samples,) with scalar predictions.
    """
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    x_all = dataset.X.to(device)
    n = x_all.shape[0]
    bs = cfg.batch_size * 4
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, bs):
            parts.append(model(x_all[start : start + bs]).cpu().numpy())
    return np.concatenate(parts, axis=0).squeeze(axis=1)
