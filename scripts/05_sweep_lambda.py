"""
Lambda sweep for the PINN model.

Runs LOWO cross-validation for each lambda_phys in a predefined grid,
calling the same training loop as 04_train_pinn.py for each value.
Per-lambda metrics are saved to individual directories; aggregate results
across the sweep are written to a single summary JSON.

Lambda grid: {0.0, 0.01, 0.05, 0.1, 0.5, 1.0}
  lambda=0.0 serves as the paired control — must reproduce the baseline.

Saves:
  outputs/pinn/lambda_{λ}/metrics.json       — per-lambda per-fold metrics
  outputs/pinn/lambda_sweep.json             — sweep summary (all lambdas)

CLI args (all optional):
  --folds N     Run only the first N folds per lambda (smoke test / CI).
  --epochs N    Override max epochs for all lambdas.
  --patience N  Override early-stopping patience for all lambdas.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.dataset import WellDataset
from src.evaluate import evaluate
from src.lowo import field_split, lowo_splits
from src.model import MLP
from src.preprocessing import preprocess_well
from src.train import TrainConfig, predict, set_seed, train_model

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------
# Configuration
# ----------------------------------------
FIELD_DIR = Path("data/raw/Kraft Prusa")
SWEEP_OUT = Path("outputs/pinn")
N_EXTERNAL = 3
SPLIT_SEED = 42

LAMBDA_GRID: list[float] = [0.0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]

_EPOCHS: int = 500
_BATCH_SIZE: int = 512
_LR: float = 1e-3
_PATIENCE: int = 20
_MIN_DELTA: float = 1e-5
_VAL_FRACTION: float = 0.15
_SEED: int = 42
_CHECKPOINT_DIR_BASE: Path = Path("outputs/checkpoints")


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides for quick iteration."""
    p = argparse.ArgumentParser(description="Lambda sweep for PINN")
    p.add_argument("--folds",   type=int, default=None, help="Run only first N folds per lambda")
    p.add_argument("--epochs",  type=int, default=None, help="Override max epochs")
    p.add_argument("--patience",type=int, default=None, help="Override early-stopping patience")
    return p.parse_args()


def _run_lowo(
    train_pool: dict[str, pd.DataFrame],
    cfg: TrainConfig,
    out_dir: Path,
    n_folds: int | None,
) -> dict[str, dict[str, float]]:
    """Run one full LOWO pass and return per-fold metrics dict."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: dict[str, dict[str, float]] = {}
    folds = list(lowo_splits(train_pool))
    if n_folds is not None:
        folds = folds[:n_folds]

    for train_wells_raw, test_id, test_df_raw in folds:
        train_dfs: list[pd.DataFrame] = []
        for wid, df in train_wells_raw.items():
            proc_df, _ = preprocess_well(df.copy(), wid, fit=True)
            train_dfs.append(proc_df)

        test_df_proc, test_scaler = preprocess_well(test_df_raw.copy(), test_id, fit=True)
        train_dataset = WellDataset(train_dfs)
        test_dataset = WellDataset(test_df_proc)

        set_seed(cfg.seed)
        model = MLP()
        train_model(model, train_dataset, cfg, well_id=test_id)

        preds_norm = predict(model, test_dataset, cfg)
        preds_gcc = test_scaler.inverse_transform_target(preds_norm)
        true_gcc = test_scaler.inverse_transform_target(test_df_proc["DEN"].values)

        fold_metrics[test_id] = evaluate(true_gcc, preds_gcc)

        depth_col = (
            test_df_raw["DEPTH"].values[: len(true_gcc)]
            if "DEPTH" in test_df_raw.columns
            else np.arange(len(true_gcc)) * 0.5
        )
        pd.DataFrame({
            "DEPTH":    depth_col,
            "DEN_true": true_gcc,
            "DEN_pred": preds_gcc,
        }).to_parquet(pred_dir / f"{test_id}.parquet", index=False)

    return fold_metrics


def _aggregate(fold_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute mean and std for each metric across folds."""
    keys = ["mae", "rmse", "r2", "pe_90"]
    agg: dict[str, float] = {}
    for key in keys:
        vals = [fold_metrics[wid][key] for wid in fold_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    return agg


def main() -> None:
    args = _parse_args()

    # ----------------------------------------
    # Step 1 — Load data once (shared across all lambda runs)
    # ----------------------------------------
    print("Loading LAS files...")
    wells_raw = load_field(FIELD_DIR)
    print(f"  Loaded {len(wells_raw)} wells ({sum(len(d) for d in wells_raw.values()):,} rows)")

    train_pool, external_set = field_split(wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    print(f"  Train pool: {len(train_pool)} wells  |  External set: {sorted(external_set)}")

    n_folds_total = len(list(lowo_splits(train_pool)))
    n_folds_run = args.folds if args.folds is not None else n_folds_total
    print(f"\nSweep over lambda grid: {LAMBDA_GRID}")
    print(f"Folds per lambda: {n_folds_run} of {n_folds_total}\n")

    # ----------------------------------------
    # Step 2 — Sweep over lambda grid
    # ----------------------------------------
    sweep_results: dict[str, dict] = {}

    for lam in tqdm(LAMBDA_GRID, desc="Lambda sweep", unit="λ"):
        ckpt_dir = _CHECKPOINT_DIR_BASE / f"lambda_{lam}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg = TrainConfig(
            epochs=args.epochs if args.epochs is not None else _EPOCHS,
            batch_size=_BATCH_SIZE,
            lr=_LR,
            patience=args.patience if args.patience is not None else _PATIENCE,
            min_delta=_MIN_DELTA,
            val_fraction=_VAL_FRACTION,
            lambda_phys=lam,
            seed=_SEED,
            checkpoint_dir=ckpt_dir,
        )
        out_dir = SWEEP_OUT / f"lambda_{lam}"
        out_dir.mkdir(parents=True, exist_ok=True)

        fold_metrics = _run_lowo(train_pool, cfg, out_dir, args.folds)
        aggregate = _aggregate(fold_metrics)

        # Save per-lambda metrics
        per_lambda = {"lambda_phys": lam, "folds": fold_metrics, "aggregate": aggregate}
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(per_lambda, f, indent=2)

        sweep_results[str(lam)] = {"lambda_phys": lam, "aggregate": aggregate}
        print(
            f"  λ={lam:5.2f}  MAE={aggregate['mae_mean']:.4f}±{aggregate['mae_std']:.4f}"
            f"  R²={aggregate['r2_mean']:.4f}±{aggregate['r2_std']:.4f}"
        )

    # ----------------------------------------
    # Step 3 — Save sweep summary
    # ----------------------------------------
    sweep_path = SWEEP_OUT / "lambda_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    # ----------------------------------------
    # Step 4 — Print summary table
    # ----------------------------------------
    print("\n=== Lambda Sweep Summary ===")
    print(f"{'λ':>6} {'MAE mean':>10} {'MAE std':>9} {'R² mean':>9} {'R² std':>8}")
    print("-" * 50)
    for lam in LAMBDA_GRID:
        a = sweep_results[str(lam)]["aggregate"]
        print(
            f"{lam:>6.2f} {a['mae_mean']:>10.4f} {a['mae_std']:>9.4f}"
            f" {a['r2_mean']:>9.4f} {a['r2_std']:>8.4f}"
        )
    print(f"\nSweep summary saved to {sweep_path}")


if __name__ == "__main__":
    main()
