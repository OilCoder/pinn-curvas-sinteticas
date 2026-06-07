"""
Physics-informed LSTM for DEN prediction over depth windows.

A small sequence model (comparable in scale to the 5→64→64→32→1 MLP) that reads
a fixed-length window of depth samples and predicts DEN at the window's last
sample. Training mirrors the MLP's PINN loss: MSE on the data plus an optional
caliper-weighted physics penalty (``lambda_phys=0`` is pure data-driven).

Windows are built per well and never cross well boundaries, so each sequence is
geologically contiguous. The caliper (DCAL_NORM) is included as a sixth feature.

Called by: scripts/13_train_lstm.py, scripts/14_eval_external_lstm.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.physics import physics_loss as _physics_loss
from src.train import TrainConfig, set_seed

WINDOW: int = 32


class LSTMRegressor(nn.Module):
    """Single-layer LSTM with a linear head predicting DEN at the last step.

    Args:
        input_dim: Number of input features per depth sample.
        hidden: LSTM hidden size.
        num_layers: Number of stacked LSTM layers.
        output_dim: Output size (1 for scalar DEN).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden: int = 64,
        num_layers: int = 1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict DEN from a batch of depth windows.

        Args:
            x: Input tensor of shape (batch, window, input_dim).

        Returns:
            Tensor of shape (batch, output_dim) — DEN at each window's last step.
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_sequences(
    x: np.ndarray,
    y: np.ndarray,
    nphi: np.ndarray,
    gr: np.ndarray,
    w: np.ndarray,
    depth: np.ndarray,
    window: int = WINDOW,
) -> dict[str, np.ndarray]:
    """Build sliding depth windows for a single well (no boundary crossing).

    Each window spans ``window`` consecutive depth samples; the supervised
    target and the physics signals (NPHI, GR, caliper weight) are taken at the
    window's last sample. A well with fewer than ``window`` samples yields no
    windows.

    Args:
        x: Feature matrix for one well, shape (n, n_features).
        y: Normalized DEN target, shape (n,).
        nphi: Normalized NPHI, shape (n,).
        gr: Normalized GR, shape (n,).
        w: Caliper quality weight, shape (n,).
        depth: Depth in feet, shape (n,).
        window: Window length in samples.

    Returns:
        Dict with arrays: xs (m, window, n_features) and y/nphi/gr/w/depth (m,),
        where m = max(0, n - window + 1).
    """
    n = x.shape[0]
    if n < window:
        f = x.shape[1]
        return {
            "xs": np.empty((0, window, f), dtype=np.float32),
            "y": np.empty(0, dtype=np.float32),
            "nphi": np.empty(0, dtype=np.float32),
            "gr": np.empty(0, dtype=np.float32),
            "w": np.empty(0, dtype=np.float32),
            "depth": np.empty(0, dtype=np.float32),
        }
    sw = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    xs = np.ascontiguousarray(sw.transpose(0, 2, 1), dtype=np.float32)
    last = slice(window - 1, n)
    return {
        "xs": xs,
        "y": y[last].astype(np.float32),
        "nphi": nphi[last].astype(np.float32),
        "gr": gr[last].astype(np.float32),
        "w": w[last].astype(np.float32),
        "depth": depth[last].astype(np.float32),
    }


def train_lstm(
    model: LSTMRegressor,
    seqs: dict[str, np.ndarray],
    cfg: TrainConfig,
    well_id: str = "lstm",
) -> dict[str, list[float]]:
    """Train the LSTM with Adam + MSE and an optional physics penalty.

    GPU-resident batching with a seeded internal validation split for early
    stopping, mirroring src.train.train_model. ``cfg.lambda_phys > 0`` adds the
    caliper-weighted physics loss on the target-sample NPHI/GR.

    Args:
        model: LSTMRegressor (best weights restored on exit).
        seqs: Concatenated training windows from ``build_sequences``.
        cfg: Training configuration.
        well_id: Label used for the checkpoint filename.

    Returns:
        Dict with per-epoch "train_loss" and "val_loss" lists.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    x_all = torch.from_numpy(seqs["xs"]).to(device)
    y_all = torch.from_numpy(seqs["y"]).to(device).unsqueeze(1)
    nphi_all = torch.from_numpy(seqs["nphi"]).to(device)
    gr_all = torch.from_numpy(seqs["gr"]).to(device)
    w_all = torch.from_numpy(seqs["w"]).to(device)
    n_total = x_all.shape[0]

    n_val = max(1, int(n_total * cfg.val_fraction))
    n_train = n_total - n_val
    perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(cfg.seed)
    ).to(device)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / f"{well_id}_best.pt"

    best_val = float("inf")
    patience = 0
    hist_tr: list[float] = []
    hist_val: list[float] = []
    bs, vbs = cfg.batch_size, cfg.batch_size * 4

    for _ in range(cfg.epochs):
        model.train()
        epoch_idx = train_idx[torch.randperm(n_train, device=device)]
        tr_sum = torch.zeros((), device=device)
        for start in range(0, n_train, bs):
            bidx = epoch_idx[start : start + bs]
            optimizer.zero_grad()
            pred = model(x_all[bidx])
            loss = mse(pred, y_all[bidx])
            if cfg.lambda_phys > 0.0:
                loss = loss + cfg.lambda_phys * _physics_loss(
                    pred, nphi_all[bidx], gr_all[bidx], w_all[bidx]
                )
            loss.backward()
            optimizer.step()
            tr_sum = tr_sum + loss.detach() * bidx.shape[0]
        hist_tr.append((tr_sum / n_train).item())

        model.eval()
        val_sum = torch.zeros((), device=device)
        with torch.no_grad():
            for start in range(0, n_val, vbs):
                bidx = val_idx[start : start + vbs]
                val_sum = val_sum + mse(model(x_all[bidx]), y_all[bidx]) * bidx.shape[0]
        val_loss = (val_sum / n_val).item()
        hist_val.append(val_loss)

        if best_val - val_loss > cfg.min_delta:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if ckpt_path.exists():
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
    return {"train_loss": hist_tr, "val_loss": hist_val}


def predict_lstm(model: LSTMRegressor, xs: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    """Run windowed inference and return normalized DEN predictions.

    Args:
        model: Trained LSTMRegressor.
        xs: Window tensor, shape (m, window, n_features).
        cfg: Config (device, batch size).

    Returns:
        Numpy array of shape (m,) with normalized DEN predictions.
    """
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    x_all = torch.from_numpy(xs).to(device)
    bs = cfg.batch_size * 4
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_all.shape[0], bs):
            parts.append(model(x_all[start : start + bs]).cpu().numpy())
    return np.concatenate(parts, axis=0).squeeze(axis=1)
