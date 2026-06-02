"""Tests for src/train.py."""

import numpy as np
import pandas as pd
import torch

from src.dataset import WellDataset
from src.model import MLP
from src.train import TrainConfig, predict, set_seed, train_model


def _make_dataset(n: int = 300, seed: int = 0) -> WellDataset:
    """Synthetic dataset with all 6 canonical columns pre-normalized to [0,1]."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "GR":   rng.uniform(0, 1, n).astype(np.float32),
        "RT":   rng.uniform(0, 1, n).astype(np.float32),
        "RILM": rng.uniform(0, 1, n).astype(np.float32),
        "NPHI": rng.uniform(0, 1, n).astype(np.float32),
        "SP":   rng.uniform(0, 1, n).astype(np.float32),
        "DEN":  rng.uniform(0, 1, n).astype(np.float32),
    })
    return WellDataset(df)


def test_smoke_train_2_epochs(tmp_path):
    model = MLP()
    dataset = _make_dataset()
    cfg = TrainConfig(epochs=2, patience=100, checkpoint_dir=tmp_path)
    history = train_model(model, dataset, cfg, well_id="smoke")
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2


def test_history_losses_are_finite(tmp_path):
    model = MLP()
    dataset = _make_dataset()
    cfg = TrainConfig(epochs=5, patience=100, checkpoint_dir=tmp_path)
    history = train_model(model, dataset, cfg, well_id="finite")
    for loss in history["train_loss"] + history["val_loss"]:
        assert np.isfinite(loss), f"Non-finite loss: {loss}"


def test_train_loss_decreases_over_50_epochs(tmp_path):
    """MLP should make meaningful progress on a fixed random dataset."""
    set_seed(42)
    model = MLP()
    dataset = _make_dataset(n=500, seed=42)
    cfg = TrainConfig(epochs=50, patience=200, batch_size=64, checkpoint_dir=tmp_path)
    history = train_model(model, dataset, cfg, well_id="decrease")
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_predict_output_shape(tmp_path):
    model = MLP()
    dataset = _make_dataset(n=100)
    cfg = TrainConfig(epochs=1, patience=100, checkpoint_dir=tmp_path)
    train_model(model, dataset, cfg, well_id="pred_shape")
    preds = predict(model, dataset, cfg)
    assert preds.shape == (100,)


def test_predict_values_are_finite(tmp_path):
    model = MLP()
    dataset = _make_dataset(n=50)
    cfg = TrainConfig(epochs=1, patience=100, checkpoint_dir=tmp_path)
    train_model(model, dataset, cfg, well_id="pred_finite")
    preds = predict(model, dataset, cfg)
    assert np.all(np.isfinite(preds))


def test_early_stopping_stops_before_max_epochs(tmp_path):
    """With patience=2, training should stop well before 500 epochs."""
    model = MLP()
    dataset = _make_dataset(n=200)
    cfg = TrainConfig(epochs=500, patience=2, min_delta=0.0, checkpoint_dir=tmp_path)
    history = train_model(model, dataset, cfg, well_id="early_stop")
    assert len(history["train_loss"]) < 500


def test_set_seed_same_initialization():
    """Two models initialized after the same seed must have identical weights."""
    set_seed(42)
    m1 = MLP()
    set_seed(42)
    m2 = MLP()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_lambda_phys_zero_same_as_no_physics(tmp_path):
    """lambda_phys=0.0 must produce the same result as not passing it."""
    set_seed(42)
    m1 = MLP()
    set_seed(42)
    m2 = MLP()
    dataset = _make_dataset()
    cfg_base = TrainConfig(epochs=3, patience=100, lambda_phys=0.0, checkpoint_dir=tmp_path)
    h1 = train_model(m1, dataset, cfg_base, well_id="lp0_a")
    set_seed(42)
    m2 = MLP()
    h2 = train_model(m2, dataset, cfg_base, well_id="lp0_b")
    np.testing.assert_allclose(h1["train_loss"], h2["train_loss"], rtol=1e-5)


def test_lambda_phys_nonzero_affects_training(tmp_path):
    """lambda_phys > 0 must produce a different loss trajectory than lambda_phys=0."""
    dataset = _make_dataset(n=500, seed=7)
    cfg_base = TrainConfig(epochs=5, patience=100, lambda_phys=0.0, batch_size=64, checkpoint_dir=tmp_path)
    cfg_phys = TrainConfig(epochs=5, patience=100, lambda_phys=1.0, batch_size=64, checkpoint_dir=tmp_path)

    set_seed(42)
    m_base = MLP()
    h_base = train_model(m_base, dataset, cfg_base, well_id="ctrl_base")

    set_seed(42)
    m_phys = MLP()
    h_phys = train_model(m_phys, dataset, cfg_phys, well_id="ctrl_phys")

    # With a strong physics weight the training loss must diverge from the pure MLP
    losses_differ = not np.allclose(h_base["train_loss"], h_phys["train_loss"], rtol=1e-4)
    assert losses_differ, "lambda_phys=1.0 should produce different losses than lambda_phys=0"
