"""
Evaluation metrics for well log regression.

All inputs must be in physical units (g/cc) after inverse transform.
Returns a plain dict so results are JSON-serializable.

Called by: scripts/03_train_baseline.py, scripts/04_train_pinn.py
"""

import numpy as np


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics between ground truth and predictions.

    Args:
        y_true: Ground truth DEN values in g/cc, shape (n,).
        y_pred: Predicted DEN values in g/cc, shape (n,).

    Returns:
        Dict with keys:
            mae   — mean absolute error (g/cc)
            rmse  — root mean squared error (g/cc)
            r2    — coefficient of determination
            pe_90 — 90th-percentile absolute error (g/cc)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    abs_err = np.abs(y_true - y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "mae":   float(np.mean(abs_err)),
        "rmse":  float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2":    1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0,
        "pe_90": float(np.percentile(abs_err, 90)),
    }
