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
from src.evaluate import evaluate
from src.external_eval import ensemble_predict, train_pool_ids
from src.lowo import field_split

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_DIR = Path("outputs/external")
N_EXTERNAL = 3
SPLIT_SEED = 42


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
    # Step 2 — Identify valid LOWO checkpoints (exclude stale files)
    # ----------------------------------------
    valid_ids = train_pool_ids(wells_raw, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    n_models = len([
        c for c in checkpoint_dir.glob("*_best.pt")
        if c.stem.replace("_best", "") in valid_ids
    ])
    if n_models == 0:
        print(f"ERROR: no valid checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    print(f"Ensemble: {n_models} models from {checkpoint_dir}")

    # ----------------------------------------
    # Step 3 — Predict each external well with the LOWO ensemble
    # ----------------------------------------
    fold_metrics: dict[str, dict[str, float]] = {}

    for well_id, df_raw in sorted(external_set.items()):
        print(f"\n  → {well_id}")
        depth_col, true_gcc, preds_gcc = ensemble_predict(
            df_raw, well_id, checkpoint_dir, valid_ids)

        metrics = evaluate(true_gcc, preds_gcc)
        fold_metrics[well_id] = metrics
        print(f"     MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
              f"R²={metrics['r2']:.4f}  PE_90={metrics['pe_90']:.4f}")

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
        "n_models_ensemble": n_models,
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
