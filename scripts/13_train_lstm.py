"""
LOWO training for the physics-informed LSTM model (with caliper input).

Mirrors scripts/11 but uses a small sequence model: per-well depth windows of
length WINDOW (no boundary crossing), DEN predicted at the window's last sample.
For each lambda it runs Leave-One-Well-Out on the Kraft Prusa train pool (27
wells, field_split n_external=3 seed=42). lambda=0 is pure data-driven; lambda>0
adds the caliper-weighted physics penalty on the target-sample NPHI/GR.

Per-well sequences are cached once (independent of fold and lambda).

Saves (per lambda):
  outputs/experiments/lstm/lambda_{λ}/metrics.json
  outputs/experiments/lstm/lambda_{λ}/predictions/*.parquet
  outputs/experiments/lstm/lambda_{λ}/checkpoints/{well}_best.pt  (27 fold models)

CLI args:
  --lambdas "0,0.5"   Comma-separated physics weights to sweep.
  --folds N           Run only the first N folds (smoke test).
  --epochs N          Max epochs (default 150).
  --window N          Window length (default WINDOW=32).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.evaluate import evaluate
from src.features_caliper import build_arrays, preprocess_well_with_caliper
from src.lowo import field_split
from src.model_lstm import (
    WINDOW,
    LSTMRegressor,
    build_sequences,
    predict_lstm,
    train_lstm,
)
from src.train import TrainConfig, set_seed

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_ROOT = Path("outputs/experiments/lstm")
N_EXTERNAL = 3
SPLIT_SEED = 42
N_FEATURES = 6


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides."""
    p = argparse.ArgumentParser(description="LOWO physics-informed LSTM training")
    p.add_argument(
        "--lambdas", type=str, default="0,0.5", help="Comma-separated physics weights"
    )
    p.add_argument("--folds", type=int, default=None, help="Run only first N folds")
    p.add_argument("--epochs", type=int, default=150, help="Max epochs")
    p.add_argument("--window", type=int, default=WINDOW, help="Depth window length")
    return p.parse_args()


def _build_cache(train_pool: dict[str, pd.DataFrame], window: int) -> dict[str, dict]:
    """Preprocess each well once and cache its depth-window sequences + scaler."""
    cache: dict[str, dict] = {}
    for wid, df in tqdm(train_pool.items(), desc="Sequences", unit="well"):
        proc, scaler = preprocess_well_with_caliper(df.copy(), wid, fit=True)
        x, y, nphi, gr, w, depth = build_arrays(proc, with_caliper=True)
        seqs = build_sequences(x, y, nphi, gr, w, depth, window)
        cache[wid] = {"seqs": seqs, "scaler": scaler}
    return cache


def _concat(seqs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-well sequence dicts into one training set."""
    return {k: np.concatenate([s[k] for s in seqs_list]) for k in seqs_list[0]}


def _run_lambda(
    lam: float,
    cache: dict[str, dict],
    well_ids: list[str],
    cfg: TrainConfig,
) -> None:
    """Run LOWO for one lambda; write metrics, predictions, and checkpoints."""
    lam_dir = OUT_ROOT / f"lambda_{lam}"
    pred_dir = lam_dir / "predictions"
    ckpt_dir = lam_dir / "checkpoints"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: dict[str, dict[str, float]] = {}
    fold_cfg = TrainConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        val_fraction=cfg.val_fraction,
        lambda_phys=lam,
        seed=cfg.seed,
        checkpoint_dir=ckpt_dir,
    )

    for test_id in tqdm(well_ids, desc=f"LSTM λ={lam}", unit="fold"):
        train_seqs = _concat([cache[w]["seqs"] for w in well_ids if w != test_id])

        set_seed(cfg.seed)
        model = LSTMRegressor(input_dim=N_FEATURES)
        train_lstm(model, train_seqs, fold_cfg, well_id=test_id)

        te = cache[test_id]
        scaler = te["scaler"]
        preds_gcc = scaler.inverse_transform_target(
            predict_lstm(model, te["seqs"]["xs"], fold_cfg)
        )
        true_gcc = scaler.inverse_transform_target(te["seqs"]["y"])

        fold_metrics[test_id] = evaluate(true_gcc, preds_gcc)
        pd.DataFrame(
            {"DEPTH": te["seqs"]["depth"], "DEN_true": true_gcc, "DEN_pred": preds_gcc}
        ).to_parquet(pred_dir / f"{test_id}.parquet", index=False)

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


def main() -> None:
    """Run the LOWO LSTM sweep over the requested lambda values."""
    args = _parse_args()
    lambdas = [float(s) for s in args.lambdas.split(",")]
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=1024, lr=1e-3, patience=12, seed=42
    )

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

    cache = _build_cache({w: train_pool[w] for w in well_ids}, args.window)

    print(
        f"\nRunning LOWO LSTM for λ ∈ {lambdas} (window={args.window}, {len(well_ids)} folds)..."
    )
    for lam in lambdas:
        _run_lambda(lam, cache, well_ids, cfg)

    print(f"\nResults saved under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
