"""
Evaluate trained LOWO models on the held-out external validation set.

Loads the 27 checkpoint files saved during LOWO training and uses them as
an ensemble to predict DEN on {Arensman_2, Burmeister_1, Rous_'F'_2}.
Each model was trained on 26 of the 27 train-pool wells; none has seen
the external wells at any stage.

Ensemble strategy: mean of 27 model predictions (one per LOWO fold).

NOTE: checkpoint files correspond to the lambda value of the most recent
training run that wrote to outputs/checkpoints/. Verify which lambda was
used before interpreting absolute metric values.

Saves:
  outputs/external/metrics.json          — per-well + aggregate metrics
  outputs/external/predictions/*.parquet — [DEPTH, DEN_true, DEN_pred]

CLI args:
  --checkpoint_dir DIR   Directory with *_best.pt files (default: outputs/checkpoints)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.dataset import WellDataset
from src.evaluate import evaluate
from src.lowo import field_split
from src.model import MLP
from src.preprocessing import preprocess_well
from src.train import TrainConfig, predict

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_DIR = Path("outputs/external")
N_EXTERNAL = 3
SPLIT_SEED = 42

_DUMMY_CFG = TrainConfig(epochs=1, checkpoint_dir=Path("outputs/checkpoints"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate external set with LOWO ensemble")
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory with *_best.pt checkpoint files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_dir: Path = args.checkpoint_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "predictions").mkdir(exist_ok=True)

    # ----------------------------------------
    # Step 1 — Load wells and identify external set
    # ----------------------------------------
    print("Loading LAS files...")
    wells_raw = load_field(FIELD_DIR)
    _, external_set = field_split(wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    print(f"External wells: {sorted(external_set)}")

    # ----------------------------------------
    # Step 2 — Load all checkpoint files
    # ----------------------------------------
    checkpoints = sorted(checkpoint_dir.glob("*_best.pt"))
    # Keep only checkpoints that correspond to train-pool wells (exclude stale files)
    train_pool, _ = field_split(wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    valid_ids = set(train_pool.keys())
    checkpoints = [c for c in checkpoints if c.stem.replace("_best", "") in valid_ids]

    if not checkpoints:
        print(f"ERROR: no valid checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    print(f"Ensemble: {len(checkpoints)} models from {checkpoint_dir}")

    # ----------------------------------------
    # Step 3 — Predict each external well
    # ----------------------------------------
    import torch

    fold_metrics: dict[str, dict[str, float]] = {}

    for well_id, df_raw in sorted(external_set.items()):
        print(f"\n  → {well_id}")

        # Preprocess external well with its own scaler (no leakage)
        df_proc, scaler = preprocess_well(df_raw.copy(), well_id, fit=True)
        dataset = WellDataset(df_proc)

        # Collect predictions from each checkpoint model
        all_preds_norm: list[np.ndarray] = []
        for ckpt_path in checkpoints:
            model = MLP()
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            model.eval()
            preds = predict(model, dataset, _DUMMY_CFG)
            all_preds_norm.append(preds)

        # Ensemble: mean over all models
        ensemble_norm = np.mean(all_preds_norm, axis=0)

        # Inverse-transform to g/cc
        preds_gcc = scaler.inverse_transform_target(ensemble_norm)
        true_gcc = scaler.inverse_transform_target(df_proc["DEN"].values)

        metrics = evaluate(true_gcc, preds_gcc)
        fold_metrics[well_id] = metrics
        print(f"     MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
              f"R²={metrics['r2']:.4f}  PE_90={metrics['pe_90']:.4f}")

        # Save predictions
        depth_col = (
            df_raw["DEPTH"].values[: len(true_gcc)]
            if "DEPTH" in df_raw.columns
            else np.arange(len(true_gcc)) * 0.5
        )
        pd.DataFrame({
            "DEPTH":    depth_col,
            "DEN_true": true_gcc,
            "DEN_pred": preds_gcc,
        }).to_parquet(OUT_DIR / "predictions" / f"{well_id}.parquet", index=False)

    # ----------------------------------------
    # Step 4 — Aggregate and save
    # ----------------------------------------
    metric_keys = ["mae", "rmse", "r2", "pe_90"]
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        vals = [fold_metrics[w][key] for w in fold_metrics]
        aggregate[f"{key}_mean"] = float(np.mean(vals))
        aggregate[f"{key}_std"] = float(np.std(vals))

    output = {
        "checkpoint_dir": str(checkpoint_dir),
        "n_models_ensemble": len(checkpoints),
        "folds": fold_metrics,
        "aggregate": aggregate,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== External Validation Results ===")
    print(f"{'Well':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'PE_90':>8}")
    print("-" * 66)
    for w, m in sorted(fold_metrics.items()):
        print(f"{w:<30} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['pe_90']:>8.4f}")
    print("-" * 66)
    print(f"{'Mean':<30} {aggregate['mae_mean']:>8.4f} {aggregate['rmse_mean']:>8.4f} "
          f"{aggregate['r2_mean']:>8.4f} {aggregate['pe_90_mean']:>8.4f}")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
