"""Tests for src/external_eval.py."""

import numpy as np
import pandas as pd
import torch

from src.external_eval import ensemble_predict, predict_well, train_final_model
from src.model import MLP
from src.train import TrainConfig


def _make_raw_well(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw well with canonical columns and physically valid DEN."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "DEPTH": np.arange(n, dtype=float) * 0.5,
        "GR":    rng.uniform(20, 120, n),
        "RT":    rng.uniform(1.0, 50.0, n),
        "RILM":  rng.uniform(1.0, 50.0, n),
        "NPHI":  rng.uniform(0.05, 0.40, n),
        "SP":    rng.uniform(-80, 10, n),
        "DEN":   rng.uniform(2.0, 2.8, n),
    })


def _make_pool(n_wells: int = 3) -> dict[str, pd.DataFrame]:
    return {f"w{i}": _make_raw_well(seed=i) for i in range(n_wells)}


# ----------------------------------------
# train_final_model
# ----------------------------------------

def test_train_final_model_returns_trained_mlp(tmp_path):
    cfg = TrainConfig(epochs=2, patience=100, batch_size=64, checkpoint_dir=tmp_path)
    model = train_final_model(_make_pool(), cfg, well_id="final")
    assert isinstance(model, MLP)


def test_train_final_model_deterministic(tmp_path):
    """Same seed → identical final weights."""
    cfg = TrainConfig(epochs=2, patience=100, batch_size=64, checkpoint_dir=tmp_path)
    m1 = train_final_model(_make_pool(), cfg, well_id="a")
    m2 = train_final_model(_make_pool(), cfg, well_id="b")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


# ----------------------------------------
# predict_well
# ----------------------------------------

def test_predict_well_shapes_and_finite(tmp_path):
    cfg = TrainConfig(epochs=2, patience=100, batch_size=64, checkpoint_dir=tmp_path)
    model = train_final_model(_make_pool(), cfg)
    depth, true, pred = predict_well(model, _make_raw_well(seed=9), "test_well", cfg)
    assert depth.shape == true.shape == pred.shape
    assert len(pred) > 0
    assert np.all(np.isfinite(pred))
    # Predictions in g/cc must land in a physical-ish range
    assert pred.min() > 0.5 and pred.max() < 4.0


# ----------------------------------------
# ensemble_predict
# ----------------------------------------

def test_ensemble_predict_averages(tmp_path):
    """Ensemble of two identical checkpoints equals a single-model prediction."""
    cfg = TrainConfig(epochs=2, patience=100, batch_size=64, checkpoint_dir=tmp_path)
    model = train_final_model(_make_pool(), cfg)

    # Save the same model state under two valid well-named checkpoints
    (tmp_path / "w0_best.pt").write_bytes(b"")  # placeholder to be overwritten
    torch.save(model.state_dict(), tmp_path / "w0_best.pt")
    torch.save(model.state_dict(), tmp_path / "w1_best.pt")

    raw = _make_raw_well(seed=5)
    _, _, pred_single = predict_well(model, raw, "ext", cfg)
    _, _, pred_ens = ensemble_predict(raw, "ext", tmp_path, valid_ids={"w0", "w1"}, cfg=cfg)

    np.testing.assert_allclose(pred_ens, pred_single, rtol=1e-5)


def test_ensemble_predict_skips_invalid_checkpoints(tmp_path):
    """Checkpoints whose stem is not in valid_ids are ignored."""
    cfg = TrainConfig(epochs=2, patience=100, batch_size=64, checkpoint_dir=tmp_path)
    model = train_final_model(_make_pool(), cfg)
    torch.save(model.state_dict(), tmp_path / "w0_best.pt")
    torch.save(model.state_dict(), tmp_path / "stale_best.pt")

    _, _, pred = ensemble_predict(_make_raw_well(seed=3), "ext", tmp_path, valid_ids={"w0"}, cfg=cfg)
    assert np.all(np.isfinite(pred))
