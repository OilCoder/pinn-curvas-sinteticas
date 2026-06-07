"""Tests for src/model_trees.py."""

import numpy as np
import xgboost as xgb

from src.model_trees import DEFAULT_PARAMS, PhysicsXGB, physics_expected


# ----------------------------------------
# Fixtures
# ----------------------------------------


def _make_data(n: int = 800, n_features: int = 6) -> dict[str, np.ndarray]:
    """Synthetic normalized features/target with a usable physics signal."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, n_features)).astype(np.float32)
    nphi = x[:, 3]
    gr = x[:, 0]
    f_phys = physics_expected(nphi, gr)
    other = rng.standard_normal(n).astype(np.float32)
    # Target mixes the physics signal with an unrelated component + noise,
    # so a physics penalty can measurably pull predictions toward f_phys.
    y = (0.5 * f_phys + 0.5 * other + 0.1 * rng.standard_normal(n)).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    return {"x": x, "y": y, "nphi": nphi, "gr": gr, "w": w, "f_phys": f_phys}


# ----------------------------------------
# Tests
# ----------------------------------------


def test_fit_predict_shapes() -> None:
    """fit then predict returns one value per row."""
    d = _make_data()
    model = PhysicsXGB(lambda_phys=0.0, n_rounds=30).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    preds = model.predict(d["x"])
    assert preds.shape == (len(d["y"]),)
    assert np.isfinite(preds).all()


def test_lambda_zero_matches_standard_regression() -> None:
    """lambda_phys=0 reproduces reg:squarederror (the project invariant)."""
    d = _make_data()
    rounds = 40
    custom = PhysicsXGB(lambda_phys=0.0, n_rounds=rounds).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    dtrain = xgb.DMatrix(d["x"], label=d["y"])
    params = {
        **DEFAULT_PARAMS,
        "base_score": float(np.mean(d["y"])),
        "objective": "reg:squarederror",
    }
    standard = xgb.train(params, dtrain, num_boost_round=rounds)
    assert np.allclose(custom.predict(d["x"]), standard.predict(dtrain), atol=1e-4)


def test_physics_objective_lowers_physics_residual() -> None:
    """A positive lambda pulls predictions toward the physics expectation."""
    d = _make_data()
    base = PhysicsXGB(lambda_phys=0.0, n_rounds=60).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    phys = PhysicsXGB(lambda_phys=2.0, n_rounds=60).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    res_base = float(np.mean((base.predict(d["x"]) - d["f_phys"]) ** 2))
    res_phys = float(np.mean((phys.predict(d["x"]) - d["f_phys"]) ** 2))
    assert res_phys < res_base


def test_determinism_same_seed() -> None:
    """Two fits with the same seed give identical predictions."""
    d = _make_data()
    a = PhysicsXGB(lambda_phys=0.5, n_rounds=30).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    b = PhysicsXGB(lambda_phys=0.5, n_rounds=30).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    assert np.array_equal(a.predict(d["x"]), b.predict(d["x"]))


def test_save_load_roundtrip(tmp_path) -> None:
    """A loaded booster predicts identically to the saved one."""
    d = _make_data()
    model = PhysicsXGB(lambda_phys=0.5, n_rounds=30).fit(
        d["x"], d["y"], d["nphi"], d["gr"], d["w"]
    )
    path = tmp_path / "booster.ubj"
    model.save(path)
    reloaded = PhysicsXGB().load(path)
    assert np.array_equal(model.predict(d["x"]), reloaded.predict(d["x"]))
