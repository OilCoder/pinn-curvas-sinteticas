"""
Physics constraint for the PINN loss term.

Encodes the empirical linear relation DEN_norm = A * NPHI_norm + B,
calibrated from 27 train-pool wells in normalized space (post preprocess_well).

Calibration results (debug/dbg_calibrate_physics.py, 2026-05-31):
  DEN_norm = -0.2939 * NPHI_norm + 0.7608   R²=0.125  RMSE=0.205

Called by: src/train.py (physics_loss term, weighted by lambda_phys)
"""

import torch
from torch import Tensor

# Coefficients calibrated in normalized space over the Kraft Prusa train pool.
# The relation is intentionally weak (R²=0.125) — lambda_phys acts as soft
# regularization, not a hard constraint.
A_PHYS: float = -0.293883
B_PHYS: float = 0.760774


def den_from_nphi(nphi: Tensor, a: float = A_PHYS, b: float = B_PHYS) -> Tensor:
    """Compute expected DEN from NPHI using the linear physics relation.

    Both inputs and outputs are in normalized space ([0, 1] per-well min-max).

    Args:
        nphi: NPHI tensor of any shape, normalized.
        a: Slope of the linear relation (default: calibrated value).
        b: Intercept of the linear relation (default: calibrated value).

    Returns:
        Tensor of same shape as nphi with expected DEN values.
    """
    return a * nphi + b


def physics_loss(
    den_pred: Tensor,
    nphi_obs: Tensor,
    a: float = A_PHYS,
    b: float = B_PHYS,
) -> Tensor:
    """Compute the physics regularization loss.

    Penalizes deviation of the model's DEN prediction from the expected DEN
    derived from the observed NPHI via the calibrated linear relation.

    Loss = mean( (den_pred - den_from_nphi(nphi_obs))² )

    Both tensors must be in normalized space. The caller multiplies this loss
    by lambda_phys before adding it to the data loss.

    Args:
        den_pred: Model-predicted DEN, shape (N,) or (N, 1), normalized.
        nphi_obs: Observed NPHI features, shape (N,) or (N, 1), normalized.
        a: Slope of the physics relation (default: calibrated value).
        b: Intercept of the physics relation (default: calibrated value).

    Returns:
        Scalar tensor with the mean squared physics residual.
    """
    den_expected = den_from_nphi(nphi_obs, a, b)
    return torch.mean((den_pred.squeeze() - den_expected.squeeze()) ** 2)
