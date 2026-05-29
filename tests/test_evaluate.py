"""Tests for src/evaluate.py."""

import numpy as np
import pytest

from src.evaluate import evaluate


def test_perfect_prediction_all_metrics():
    y = np.array([2.1, 2.3, 2.5, 2.7])
    m = evaluate(y, y)
    assert m["mae"] == pytest.approx(0.0, abs=1e-10)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert m["r2"] == pytest.approx(1.0, abs=1e-10)
    assert m["pe_90"] == pytest.approx(0.0, abs=1e-10)


def test_mae_known_constant_error():
    y_true = np.array([2.0, 2.0, 2.0, 2.0])
    y_pred = np.array([2.1, 2.1, 2.1, 2.1])
    m = evaluate(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.1)
    assert m["rmse"] == pytest.approx(0.1)


def test_r2_zero_for_mean_prediction():
    y_true = np.array([2.0, 2.2, 2.4, 2.6])
    y_pred = np.full(4, float(np.mean(y_true)))
    m = evaluate(y_true, y_pred)
    assert m["r2"] == pytest.approx(0.0, abs=1e-8)


def test_r2_negative_when_worse_than_mean():
    y_true = np.array([2.0, 2.5, 3.0])
    y_pred = np.array([3.0, 2.0, 1.0])  # inverted
    m = evaluate(y_true, y_pred)
    assert m["r2"] < 0.0


def test_pe_90_is_90th_percentile():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(2.0, 2.8, 100)
    errors = np.linspace(0.01, 1.00, 100)  # known sorted errors
    y_pred = y_true + errors
    m = evaluate(y_true, y_pred)
    assert m["pe_90"] == pytest.approx(float(np.percentile(errors, 90)), rel=1e-5)


def test_returns_all_keys():
    m = evaluate(np.array([2.0]), np.array([2.0]))
    assert set(m.keys()) == {"mae", "rmse", "r2", "pe_90"}


def test_all_values_are_floats():
    m = evaluate(np.array([2.0, 2.5, 3.0]), np.array([2.1, 2.4, 2.9]))
    for key, val in m.items():
        assert isinstance(val, float), f"{key} is {type(val)}, expected float"


def test_constant_target_r2_is_zero():
    """When ss_tot == 0 (all targets equal), r2 is defined as 0."""
    y_true = np.array([2.5, 2.5, 2.5])
    y_pred = np.array([2.5, 2.5, 2.5])
    m = evaluate(y_true, y_pred)
    assert m["r2"] == 0.0


def test_1d_and_2d_inputs_equivalent():
    y_true_1d = np.array([2.0, 2.3, 2.6])
    y_pred_1d = np.array([2.1, 2.2, 2.7])
    m1 = evaluate(y_true_1d, y_pred_1d)
    m2 = evaluate(y_true_1d.reshape(-1, 1), y_pred_1d.reshape(-1, 1))
    assert m1 == pytest.approx(m2, rel=1e-8)
