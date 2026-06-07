"""
Physics-informed gradient boosting (XGBoost) for DEN prediction.

Trains an XGBoost regressor with a custom objective that adds a caliper-weighted
physics penalty to the data term, mirroring the PINN loss of the MLP:

    Loss = (1/2)(y_pred - y)^2  +  lambda * (1/2) * w * (y_pred - f_phys)^2

with ``f_phys = A * NPHI + D * (NPHI * GR)`` (coefficients from src.physics, all
in normalized space). ``lambda_phys = 0`` reproduces plain squared-error
regression exactly — the same baseline/PINN invariant the project uses for the MLP.

Called by: scripts/11_train_xgboost.py, scripts/12_eval_external_xgboost.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.physics import A_PHYS, D_PHYS

# Modest defaults — comparable robustness to the 5→64→64→32→1 MLP, not a huge model.
DEFAULT_PARAMS: dict[str, object] = {
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "tree_method": "hist",
    "seed": 42,
}
N_ROUNDS: int = 400


def physics_expected(nphi: np.ndarray, gr: np.ndarray) -> np.ndarray:
    """Expected normalized DEN from NPHI and GR via the bivariate relation.

    Numpy counterpart of ``src.physics.den_from_nphi`` (kept separate so the
    custom objective stays in pure numpy).

    Args:
        nphi: Normalized NPHI, shape (n,).
        gr: Normalized GR, shape (n,).

    Returns:
        Expected normalized DEN, shape (n,).
    """
    return A_PHYS * nphi + D_PHYS * (nphi * gr)


_Objective = Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]


def _make_physics_objective(
    f_phys: np.ndarray,
    weights: np.ndarray,
    lambda_phys: float,
) -> _Objective:
    """Build an XGBoost custom objective for the caliper-weighted physics loss.

    The data term uses the half-squared-error convention (grad = pred - y,
    hess = 1), identical to ``reg:squarederror``; the physics term adds
    ``lambda * w * (pred - f_phys)``. With ``lambda_phys = 0`` the objective is
    exactly standard squared-error regression.

    Args:
        f_phys: Expected DEN per training row (normalized), shape (n,).
        weights: Caliper quality weights in [0, 1], shape (n,).
        lambda_phys: Physics loss weight (0 → pure data regression).

    Returns:
        Callable ``obj(preds, dtrain) -> (grad, hess)`` for ``xgb.train``.
    """

    def obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        grad = preds - y
        hess = np.ones_like(preds)
        if lambda_phys > 0.0:
            grad = grad + lambda_phys * weights * (preds - f_phys)
            hess = hess + lambda_phys * weights
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


class PhysicsXGB:
    """XGBoost regressor with an optional physics-informed training objective.

    Args:
        lambda_phys: Physics loss weight (0 reproduces plain regression).
        params: Booster params merged over ``DEFAULT_PARAMS``.
        n_rounds: Number of boosting rounds.
    """

    def __init__(
        self,
        lambda_phys: float = 0.0,
        params: dict[str, object] | None = None,
        n_rounds: int = N_ROUNDS,
    ) -> None:
        self.lambda_phys = float(lambda_phys)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.n_rounds = n_rounds
        self.booster: xgb.Booster | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nphi: np.ndarray,
        gr: np.ndarray,
        weights: np.ndarray,
    ) -> PhysicsXGB:
        """Train the booster on a fold's concatenated wells.

        Args:
            x: Feature matrix, shape (n, n_features).
            y: Normalized DEN target, shape (n,).
            nphi: Normalized NPHI, shape (n,) — for the physics term.
            gr: Normalized GR, shape (n,) — for the physics term.
            weights: Caliper quality weights in [0, 1], shape (n,).

        Returns:
            Self, with a trained ``booster``.
        """
        dtrain = xgb.DMatrix(x, label=y)
        params = {**self.params, "base_score": float(np.mean(y))}
        obj = _make_physics_objective(
            physics_expected(nphi, gr), weights, self.lambda_phys
        )
        self.booster = xgb.train(params, dtrain, num_boost_round=self.n_rounds, obj=obj)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict normalized DEN for a feature matrix.

        Args:
            x: Feature matrix, shape (n, n_features).

        Returns:
            Predicted normalized DEN, shape (n,).

        Raises:
            RuntimeError: If called before ``fit``/``load``.
        """
        if self.booster is None:
            raise RuntimeError("PhysicsXGB is not trained; call fit() or load() first")
        return self.booster.predict(xgb.DMatrix(x))

    def save(self, path: Path) -> None:
        """Persist the booster to ``path`` (UBJSON format).

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If called before ``fit``/``load``.
        """
        if self.booster is None:
            raise RuntimeError("PhysicsXGB is not trained; nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    def load(self, path: Path) -> PhysicsXGB:
        """Load a previously saved booster from ``path``.

        Args:
            path: Source file path.

        Returns:
            Self, with the loaded ``booster``.
        """
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        return self
