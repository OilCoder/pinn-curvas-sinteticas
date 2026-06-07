"""Tests for src/model_lstm.py."""

import numpy as np
import torch

from src.model_lstm import (
    LSTMRegressor,
    build_sequences,
    predict_lstm,
    train_lstm,
)
from src.train import TrainConfig


# ----------------------------------------
# Fixtures
# ----------------------------------------


def _well_arrays(n: int = 100, f: int = 6) -> dict[str, np.ndarray]:
    """Per-well arrays as build_arrays would return them."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, f)).astype(np.float32)
    return {
        "x": x,
        "y": rng.standard_normal(n).astype(np.float32),
        "nphi": x[:, 3],
        "gr": x[:, 0],
        "w": np.ones(n, dtype=np.float32),
        "depth": (np.arange(n) * 0.5).astype(np.float32),
    }


# ----------------------------------------
# Tests — build_sequences
# ----------------------------------------


def test_sequences_shapes_and_count() -> None:
    """A well of n samples yields n-window+1 windows of the right shape."""
    a = _well_arrays(n=100, f=6)
    window = 32
    seqs = build_sequences(
        a["x"], a["y"], a["nphi"], a["gr"], a["w"], a["depth"], window
    )
    m = 100 - window + 1
    assert seqs["xs"].shape == (m, window, 6)
    assert seqs["y"].shape == seqs["nphi"].shape == seqs["w"].shape == (m,)


def test_sequence_content_matches_slice() -> None:
    """Window i is exactly the slice x[i:i+window]; target is its last sample."""
    a = _well_arrays(n=50, f=6)
    window = 16
    seqs = build_sequences(
        a["x"], a["y"], a["nphi"], a["gr"], a["w"], a["depth"], window
    )
    assert np.allclose(seqs["xs"][0], a["x"][0:window])
    assert np.allclose(seqs["xs"][5], a["x"][5 : 5 + window])
    # Target aligns to the last sample of each window.
    assert np.allclose(seqs["y"], a["y"][window - 1 :])


def test_short_well_yields_no_windows() -> None:
    """A well shorter than the window produces zero windows, not an error."""
    a = _well_arrays(n=10, f=6)
    seqs = build_sequences(
        a["x"], a["y"], a["nphi"], a["gr"], a["w"], a["depth"], window=32
    )
    assert seqs["xs"].shape == (0, 32, 6)
    assert seqs["y"].shape == (0,)


def test_no_boundary_crossing_via_per_well_build() -> None:
    """Concatenating per-well windows never mixes samples from two wells."""
    a = _well_arrays(n=40, f=6)
    b = _well_arrays(n=40, f=6)
    window = 16
    sa = build_sequences(a["x"], a["y"], a["nphi"], a["gr"], a["w"], a["depth"], window)
    sb = build_sequences(b["x"], b["y"], b["nphi"], b["gr"], b["w"], b["depth"], window)
    # Every window in sa comes only from well a's rows; sb only from b's.
    for i in range(sa["xs"].shape[0]):
        assert np.allclose(sa["xs"][i], a["x"][i : i + window])
    for i in range(sb["xs"].shape[0]):
        assert np.allclose(sb["xs"][i], b["x"][i : i + window])


# ----------------------------------------
# Tests — LSTMRegressor + training
# ----------------------------------------


def test_forward_shape() -> None:
    """Forward maps (B, W, F) to (B, 1)."""
    model = LSTMRegressor(input_dim=6, hidden=16)
    out = model(torch.randn(8, 32, 6))
    assert out.shape == (8, 1)


def test_train_and_predict_smoke(tmp_path) -> None:
    """Training runs, returns finite losses, and predict returns one value/window."""
    a = _well_arrays(n=200, f=6)
    seqs = build_sequences(
        a["x"], a["y"], a["nphi"], a["gr"], a["w"], a["depth"], window=16
    )
    cfg = TrainConfig(
        epochs=3, batch_size=32, lambda_phys=0.5, device="cpu", checkpoint_dir=tmp_path
    )
    model = LSTMRegressor(input_dim=6, hidden=16)
    hist = train_lstm(model, seqs, cfg, well_id="t")
    assert len(hist["train_loss"]) >= 1
    assert all(np.isfinite(hist["train_loss"]))
    preds = predict_lstm(model, seqs["xs"], cfg)
    assert preds.shape == (seqs["xs"].shape[0],)
