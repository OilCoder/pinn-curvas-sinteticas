"""
LOWO training for the physics-informed XGBoost model (with caliper input).

Mirrors scripts/03_train_baseline.py but swaps the MLP for PhysicsXGB and adds
the per-well normalized caliper (DCAL_NORM) as a sixth feature. For each lambda
in the sweep it runs Leave-One-Well-Out on the Kraft Prusa train pool (27 wells
after field_split with n_external=3, seed=42). lambda=0 is the pure data-driven
gradient boosting; lambda>0 adds the caliper-weighted physics penalty.

Per fold:
  1. Preprocess train wells independently (own WellScaler) + caliper feature
  2. Preprocess test well independently (own WellScaler — no leakage)
  3. Fit PhysicsXGB on the 26 train wells; save the fold booster
  4. Predict the held-out well; inverse-transform preds and targets to g/cc
  5. Compute MAE, RMSE, R², PE_90; save predictions parquet

Saves (per lambda):
  outputs/experiments/xgboost/lambda_{λ}/metrics.json
  outputs/experiments/xgboost/lambda_{λ}/predictions/*.parquet
  outputs/experiments/xgboost/lambda_{λ}/checkpoints/*.ubj  (27 fold boosters)

CLI args:
  --lambdas "0,0.1,0.5"  Comma-separated physics weights to sweep.
  --folds N              Run only the first N folds (smoke test).
  --rounds N             Override boosting rounds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.evaluate import evaluate
from src.features_caliper import build_arrays, preprocess_well_with_caliper
from src.lowo import field_split
from src.model_trees import N_ROUNDS, PhysicsXGB

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_ROOT = Path("outputs/experiments/xgboost")
N_EXTERNAL = 3
SPLIT_SEED = 42


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides."""
    p = argparse.ArgumentParser(description="LOWO physics-informed XGBoost training")
    p.add_argument(
        "--lambdas",
        type=str,
        default="0,0.1,0.5",
        help="Comma-separated physics weights",
    )
    p.add_argument("--folds", type=int, default=None, help="Run only first N folds")
    p.add_argument("--rounds", type=int, default=N_ROUNDS, help="Boosting rounds")
    return p.parse_args()


def _build_cache(train_pool: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Preprocess each well once (own scaler) and cache its model-ready arrays.

    Per-well preprocessing is independent of the LOWO fold and of lambda, so it
    is computed a single time and reused — identical results, far less compute.

    Args:
        train_pool: Dict mapping well_id to raw DataFrame.

    Returns:
        Dict mapping well_id to {x, y, nphi, gr, w, depth, scaler}.
    """
    cache: dict[str, dict] = {}
    for wid, df in tqdm(train_pool.items(), desc="Preprocess", unit="well"):
        proc, scaler = preprocess_well_with_caliper(df.copy(), wid, fit=True)
        x, y, nphi, gr, w, depth = build_arrays(proc, with_caliper=True)
        cache[wid] = {
            "x": x,
            "y": y,
            "nphi": nphi,
            "gr": gr,
            "w": w,
            "depth": depth,
            "scaler": scaler,
        }
    return cache


def _run_lambda(
    lam: float,
    cache: dict[str, dict],
    well_ids: list[str],
    rounds: int,
) -> dict[str, dict[str, float]]:
    """Run LOWO for one lambda and write metrics, predictions, and boosters.

    Args:
        lam: Physics loss weight for this sweep value.
        cache: Per-well preprocessed arrays from ``_build_cache``.
        well_ids: Ordered list of train-pool well ids (one held out per fold).
        rounds: Number of boosting rounds.

    Returns:
        Dict mapping test well id to its metric dict.
    """
    lam_dir = OUT_ROOT / f"lambda_{lam}"
    pred_dir = lam_dir / "predictions"
    ckpt_dir = lam_dir / "checkpoints"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: dict[str, dict[str, float]] = {}

    for test_id in tqdm(well_ids, desc=f"XGB λ={lam}", unit="fold"):
        train_ids = [w for w in well_ids if w != test_id]

        # Substep 1 — Assemble train arrays from cache (26 wells)
        x_tr = np.concatenate([cache[w]["x"] for w in train_ids])
        y_tr = np.concatenate([cache[w]["y"] for w in train_ids])
        nphi_tr = np.concatenate([cache[w]["nphi"] for w in train_ids])
        gr_tr = np.concatenate([cache[w]["gr"] for w in train_ids])
        w_tr = np.concatenate([cache[w]["w"] for w in train_ids])

        # Substep 2 — Fit physics-informed booster; save fold checkpoint
        model = PhysicsXGB(lambda_phys=lam, n_rounds=rounds).fit(
            x_tr, y_tr, nphi_tr, gr_tr, w_tr
        )
        model.save(ckpt_dir / f"{test_id}.ubj")

        # Substep 3 — Predict held-out well, inverse-transform to g/cc
        te = cache[test_id]
        preds_gcc = te["scaler"].inverse_transform_target(model.predict(te["x"]))
        true_gcc = te["scaler"].inverse_transform_target(te["y"])

        # Substep 4 — Evaluate and save predictions
        fold_metrics[test_id] = evaluate(true_gcc, preds_gcc)
        pd.DataFrame(
            {"DEPTH": te["depth"], "DEN_true": true_gcc, "DEN_pred": preds_gcc}
        ).to_parquet(pred_dir / f"{test_id}.parquet", index=False)

    # Aggregate + save
    aggregate: dict[str, float] = {}
    for key in ["mae", "rmse", "r2", "pe_90"]:
        vals = [fold_metrics[wid][key] for wid in fold_metrics]
        aggregate[f"{key}_mean"] = float(np.mean(vals))
        aggregate[f"{key}_std"] = float(np.std(vals))
    with open(lam_dir / "metrics.json", "w") as f:
        json.dump({"folds": fold_metrics, "aggregate": aggregate}, f, indent=2)

    print(
        f"  λ={lam}: MAE {aggregate['mae_mean']:.4f} ± {aggregate['mae_std']:.4f} | "
        f"R² {aggregate['r2_mean']:.4f} ± {aggregate['r2_std']:.4f}"
    )
    return fold_metrics


def main() -> None:
    """Run the LOWO XGBoost sweep over the requested lambda values."""
    args = _parse_args()
    lambdas = [float(s) for s in args.lambdas.split(",")]

    print("Loading LAS files...")
    wells_raw = load_field(FIELD_DIR)
    train_pool, external_set = field_split(
        wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED
    )
    print(
        f"  Train pool: {len(train_pool)} wells  |  External set: {sorted(external_set)}"
    )

    well_ids = sorted(train_pool)
    if args.folds is not None:
        well_ids = well_ids[: args.folds]
        print(f"  Smoke test: {args.folds} folds")

    cache = _build_cache({w: train_pool[w] for w in well_ids})

    print(f"\nRunning LOWO XGBoost for λ ∈ {lambdas} ({len(well_ids)} folds each)...")
    for lam in lambdas:
        _run_lambda(lam, cache, well_ids, args.rounds)

    print(f"\nResults saved under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
