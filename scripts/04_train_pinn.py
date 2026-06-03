"""
LOWO PINN training with a fixed physics loss weight (lambda_phys).

Runs Leave-One-Well-Out cross-validation on the Kraft Prusa train pool
(27 wells after field_split with n_external=3, seed=42). Identical to
the baseline training loop except that lambda_phys > 0 adds a physics
regularization term to the MSE loss:

  Loss = MSE(DEN_pred, DEN_true) + lambda_phys * physics_loss(DEN_pred, NPHI)

For lambda_phys=0 this reproduces the baseline exactly.

Saves:
  outputs/pinn/lambda_{λ}/metrics.json         — per-fold + aggregate metrics
  outputs/pinn/lambda_{λ}/predictions/*.parquet — [DEPTH, DEN_true, DEN_pred]

CLI args (all optional — defaults run a full experiment with lambda_phys=0.1):
  --lambda_phys L  Physics loss weight (default: 0.1).
  --folds N        Run only the first N folds (smoke test / CI).
  --epochs N       Override TrainConfig.epochs.
  --patience N     Override TrainConfig.patience.
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
N_EXTERNAL = 3
SPLIT_SEED = 42

_EPOCHS: int = 500
_BATCH_SIZE: int = 512
_LR: float = 1e-3
_PATIENCE: int = 20
_MIN_DELTA: float = 1e-5
_VAL_FRACTION: float = 0.15
_SEED: int = 42
_CHECKPOINT_DIR_BASE: Path = Path("outputs/checkpoints")


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides."""
    p = argparse.ArgumentParser(description="LOWO PINN training with fixed lambda")
    p.add_argument("--lambda_phys", type=float, default=0.1, help="Physics loss weight")
    p.add_argument("--folds",        type=int,   default=None, help="Run only first N folds")
    p.add_argument("--epochs",       type=int,   default=None, help="Override max epochs")
    p.add_argument("--patience",     type=int,   default=None, help="Override early-stopping patience")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    lam = args.lambda_phys
    out_dir = Path("outputs/pinn") / f"lambda_{lam}"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

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

    # ----------------------------------------
    # Step 1 — Load raw wells and split off external set
    # ----------------------------------------
    print(f"PINN training  |  lambda_phys={lam}")
    print("Loading LAS files...")
    wells_raw = load_field(FIELD_DIR)
    print(f"  Loaded {len(wells_raw)} wells ({sum(len(d) for d in wells_raw.values()):,} rows)")

    train_pool, external_set = field_split(wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    print(f"  Train pool: {len(train_pool)} wells  |  External set: {sorted(external_set)}")

    # ----------------------------------------
    # Step 2 — LOWO cross-validation
    # ----------------------------------------
    fold_metrics: dict[str, dict[str, float]] = {}

    folds = list(lowo_splits(train_pool))
    if args.folds is not None:
        folds = folds[: args.folds]
        print(f"\nRunning LOWO ({args.folds} of {len(list(lowo_splits(train_pool)))} folds — smoke test)...")
    else:
        print(f"\nRunning LOWO ({len(folds)} folds)...")

    for train_wells_raw, test_id, test_df_raw in tqdm(folds, desc="LOWO", unit="fold"):

        # Substep 2.1 — Preprocess train wells (each with own scaler)
        train_dfs: list[pd.DataFrame] = []
        for wid, df in train_wells_raw.items():
            proc_df, _ = preprocess_well(df.copy(), wid, fit=True)
            train_dfs.append(proc_df)

        # Substep 2.2 — Preprocess test well with its own scaler (no leakage)
        test_df_proc, test_scaler = preprocess_well(test_df_raw.copy(), test_id, fit=True)

        # Substep 2.3 — Build datasets
        train_dataset = WellDataset(train_dfs)
        test_dataset = WellDataset(test_df_proc)

        # Substep 2.4 — Initialize and train model
        set_seed(cfg.seed)
        model = MLP()
        train_model(model, train_dataset, cfg, well_id=test_id)

        # Substep 2.5 — Predict and inverse-transform to g/cc
        preds_norm = predict(model, test_dataset, cfg)
        preds_gcc = test_scaler.inverse_transform_target(preds_norm)
        true_gcc = test_scaler.inverse_transform_target(test_df_proc["DEN"].values)

        # Substep 2.6 — Evaluate
        metrics = evaluate(true_gcc, preds_gcc)
        fold_metrics[test_id] = metrics

        # Substep 2.7 — Save predictions parquet
        depth_col = (
            test_df_raw["DEPTH"].values[: len(true_gcc)]
            if "DEPTH" in test_df_raw.columns
            else np.arange(len(true_gcc)) * 0.5
        )
        pred_df = pd.DataFrame({
            "DEPTH":    depth_col,
            "DEN_true": true_gcc,
            "DEN_pred": preds_gcc,
        })
        pred_df.to_parquet(pred_dir / f"{test_id}.parquet", index=False)

    # ----------------------------------------
    # Step 3 — Aggregate metrics across folds
    # ----------------------------------------
    metric_keys = ["mae", "rmse", "r2", "pe_90"]
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        vals = [fold_metrics[wid][key] for wid in fold_metrics]
        aggregate[f"{key}_mean"] = float(np.mean(vals))
        aggregate[f"{key}_std"] = float(np.std(vals))

    # ----------------------------------------
    # Step 4 — Save metrics JSON
    # ----------------------------------------
    output = {"lambda_phys": lam, "folds": fold_metrics, "aggregate": aggregate}
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    # ----------------------------------------
    # Step 5 — Print summary
    # ----------------------------------------
    print(f"\n=== PINN Results  (lambda_phys={lam}) ===")
    print(f"{'Well':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'PE_90':>8}")
    print("-" * 66)
    for wid in sorted(fold_metrics):
        m = fold_metrics[wid]
        print(f"{wid:<30} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['pe_90']:>8.4f}")
    print("-" * 66)
    print(
        f"{'Mean':<30} {aggregate['mae_mean']:>8.4f} {aggregate['rmse_mean']:>8.4f} "
        f"{aggregate['r2_mean']:>8.4f} {aggregate['pe_90_mean']:>8.4f}"
    )
    print(
        f"{'Std':<30} {aggregate['mae_std']:>8.4f} {aggregate['rmse_std']:>8.4f} "
        f"{aggregate['r2_std']:>8.4f} {aggregate['pe_90_std']:>8.4f}"
    )
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Predictions saved to {pred_dir}/")


if __name__ == "__main__":
    main()
