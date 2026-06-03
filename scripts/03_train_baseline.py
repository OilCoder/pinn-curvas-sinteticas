"""
LOWO baseline training for the MLP model.

Runs Leave-One-Well-Out cross-validation on the Kraft Prusa train pool
(27 wells after field_split with n_external=3, seed=42). For each fold:
  1. Preprocess train wells independently (each with its own WellScaler)
  2. Preprocess test well independently (its own WellScaler — no train leakage)
  3. Train MLP (seed=42, early stopping on internal 15% val split)
  4. Predict on test well; inverse-transform both preds and targets to g/cc
  5. Compute MAE, RMSE, R², PE_90; save predictions parquet

Saves:
  outputs/baseline/metrics.json         — per-fold + aggregate metrics
  outputs/baseline/predictions/*.parquet — [DEPTH, DEN_true, DEN_pred] per well

CLI args (all optional — defaults run the full experiment):
  --folds N     Run only the first N folds (smoke test / CI).
  --epochs N    Override TrainConfig.epochs.
  --patience N  Override TrainConfig.patience.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from project root without installing the package
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
OUT_DIR = Path("outputs/baseline")
PRED_DIR = OUT_DIR / "predictions"

CFG = TrainConfig(
    epochs=500,
    batch_size=512,
    lr=1e-3,
    patience=20,
    min_delta=1e-5,
    val_fraction=0.15,
    lambda_phys=0.0,
    seed=42,
    checkpoint_dir=Path("outputs/checkpoints/baseline"),
)

N_EXTERNAL = 3
SPLIT_SEED = 42


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides for quick iteration."""
    p = argparse.ArgumentParser(description="LOWO baseline training")
    p.add_argument("--folds",   type=int, default=None, help="Run only first N folds")
    p.add_argument("--epochs",  type=int, default=None, help="Override max epochs")
    p.add_argument("--patience",type=int, default=None, help="Override early-stopping patience")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = CFG
    if args.epochs is not None:
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            patience=args.patience if args.patience is not None else cfg.patience,
            min_delta=cfg.min_delta,
            val_fraction=cfg.val_fraction,
            lambda_phys=cfg.lambda_phys,
            seed=cfg.seed,
            checkpoint_dir=cfg.checkpoint_dir,
        )
    elif args.patience is not None:
        cfg = TrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            patience=args.patience,
            min_delta=cfg.min_delta,
            val_fraction=cfg.val_fraction,
            lambda_phys=cfg.lambda_phys,
            seed=cfg.seed,
            checkpoint_dir=cfg.checkpoint_dir,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Step 1 — Load raw wells and split off external set
    # ----------------------------------------
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
        depth_col = test_df_raw["DEPTH"].values[: len(true_gcc)] if "DEPTH" in test_df_raw.columns else np.arange(len(true_gcc)) * 0.5
        pred_df = pd.DataFrame({
            "DEPTH":    depth_col,
            "DEN_true": true_gcc,
            "DEN_pred": preds_gcc,
        })
        pred_df.to_parquet(PRED_DIR / f"{test_id}.parquet", index=False)

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
    output = {"folds": fold_metrics, "aggregate": aggregate}
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    # ----------------------------------------
    # Step 5 — Print summary
    # ----------------------------------------
    print("\n=== LOWO Baseline Results ===")
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
    print(f"Predictions saved to {PRED_DIR}/")


if __name__ == "__main__":
    main()
