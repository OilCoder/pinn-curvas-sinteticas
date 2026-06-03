"""
Physics constraint for the PINN loss term.

Encodes the empirical bivariate relation:
  DEN_norm = A * NPHI_norm + D * (NPHI_norm * GR_norm)

calibrated from 27 train-pool wells in Yeo-Johnson + z-score normalized space.

Calibration results (debug/dbg_calibrate_physics.py, 2026-06-02):
  DEN_norm = -0.5563 * NPHI_norm + 0.0864 * (NPHI_norm * GR_norm)
  R² = 0.338  (vs 0.330 univariate NPHI-only)

Rationale for the interaction term:
  In shaly zones (high GR), the NPHI-DEN slope weakens because clay minerals
  have high apparent neutron porosity but intermediate bulk density.
  The product NPHI*GR captures this lithology-dependent attenuation.

B=0 exactly by construction: Yeo-Johnson+standardize centers all variables
at zero, so the intercept of any regression through the origin is zero.

Called by: src/train.py (physics_loss term, weighted by lambda_phys)
"""

import torch
from torch import Tensor

# Coefficients calibrated in Yeo-Johnson + z-score space over the Kraft Prusa train pool.
# DEN_norm = A * NPHI_norm + D * (NPHI_norm * GR_norm)
A_PHYS: float = -0.556300
D_PHYS: float = 0.086400


def den_from_nphi(nphi: Tensor, gr: Tensor, a: float = A_PHYS, d: float = D_PHYS) -> Tensor:
    """Compute expected DEN from NPHI and GR using the bivariate physics relation.

    DEN_expected = A * NPHI + D * (NPHI * GR)

    The interaction term D*(NPHI*GR) captures lithology-dependent attenuation:
    in shaly zones (high GR), the NPHI-DEN slope is reduced.

    All inputs and outputs are in Yeo-Johnson + z-score normalized space.

    Args:
        nphi: NPHI tensor of any shape, normalized.
        gr:   GR tensor of same shape as nphi, normalized.
        a:    Slope coefficient for NPHI (default: calibrated value).
        d:    Interaction coefficient for NPHI*GR (default: calibrated value).

    Returns:
        Tensor of same shape as nphi with expected DEN values.
    """
    return a * nphi + d * (nphi * gr)


def physics_loss(
    den_pred: Tensor,
    nphi_obs: Tensor,
    gr_obs: Tensor,
    weights: Tensor | None = None,
    a: float = A_PHYS,
    d: float = D_PHYS,
) -> Tensor:
    """Compute the physics regularization loss, optionally weighted by caliper quality.

    Penalizes deviation of the model's DEN prediction from the expected DEN
    derived from observed NPHI and GR via the calibrated bivariate relation.
    In borehole washout zones (low DCAL_WEIGHT), the penalty is reduced because
    the physical relation is less reliable there.

    All tensors must be in Yeo-Johnson + z-score normalized space.

    Args:
        den_pred: Model-predicted DEN, shape (N,) or (N, 1), normalized.
        nphi_obs: Observed NPHI features, shape (N,) or (N, 1), normalized.
        gr_obs:   Observed GR features, shape (N,) or (N, 1), normalized.
        weights:  Caliper quality weights in [0, 1], shape (N,). None → uniform.
        a:        Slope coefficient for NPHI (default: calibrated value).
        d:        Interaction coefficient for NPHI*GR (default: calibrated value).

    Returns:
        Scalar tensor with the (weighted) mean squared physics residual.
    """
    den_expected = den_from_nphi(nphi_obs, gr_obs, a, d)
    residuals = (den_pred.squeeze() - den_expected.squeeze()) ** 2
    if weights is not None:
        w = weights.squeeze()
        return (w * residuals).sum() / w.sum().clamp(min=1e-6)
    return torch.mean(residuals)
