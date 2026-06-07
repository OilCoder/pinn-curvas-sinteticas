"""
External validation for the physics-informed LSTM model, two protocols.

Evaluates the 3 held-out external wells (field_split n_external=3, seed=42) under
the same two protocols as the MLP/XGBoost studies:

  A) Ensemble — average the 27 per-fold LOWO checkpoints saved by scripts/13
     (outputs/experiments/lstm/lambda_{λ}/checkpoints/*_best.pt).
  B) Single   — one LSTM trained on all 27 train-pool wells' sequences.

Run for λ=0 (data-driven) and λ=0.5 (physics-informed).

Saves:
  outputs/experiments/lstm/external.json — per-well + aggregate, both protocols
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
_INFER = TrainConfig(epochs=1)


def _aggregate(folds: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across the external wells (keys suffixed `_mean`)."""
    keys = ["mae", "rmse", "r2", "pe_90"]
    return {f"{k}_mean": float(np.mean([folds[w][k] for w in folds])) for k in keys}


def _well_sequences(raw_df, well_id: str, window: int):
    """Preprocess a well and return (sequences dict, scaler)."""
    proc, scaler = preprocess_well_with_caliper(raw_df.copy(), well_id, fit=True)
    x, y, nphi, gr, w, depth = build_arrays(proc, with_caliper=True)
    return build_sequences(x, y, nphi, gr, w, depth, window), scaler


def _eval_ensemble(wells, ext_ids, ckpt_dir: Path, window: int) -> dict:
    """Average all fold checkpoints to predict each external well."""
    ckpts = sorted(ckpt_dir.glob("*_best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    models = []
    for c in ckpts:
        m = LSTMRegressor(input_dim=N_FEATURES)
        m.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
        models.append(m)
    folds = {}
    for wid in ext_ids:
        seqs, scaler = _well_sequences(wells[wid], wid, window)
        ens = np.mean([predict_lstm(m, seqs["xs"], _INFER) for m in models], axis=0)
        folds[wid] = evaluate(
            scaler.inverse_transform_target(seqs["y"]),
            scaler.inverse_transform_target(ens),
        )
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _train_single(
    train_pool, lam: float, window: int, cfg: TrainConfig
) -> LSTMRegressor:
    """Train one LSTM on all train-pool wells' concatenated sequences."""
    seqs_list = [_well_sequences(df, wid, window)[0] for wid, df in train_pool.items()]
    train_seqs = {k: np.concatenate([s[k] for s in seqs_list]) for k in seqs_list[0]}
    fold_cfg = TrainConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        patience=cfg.patience,
        lambda_phys=lam,
        seed=cfg.seed,
        checkpoint_dir=OUT_ROOT / "final",
    )
    set_seed(cfg.seed)
    model = LSTMRegressor(input_dim=N_FEATURES)
    train_lstm(model, train_seqs, fold_cfg, well_id=f"final_lambda_{lam}")
    return model


def _eval_single(model, wells, ext_ids, window: int) -> dict:
    """Evaluate external wells with a single trained LSTM."""
    folds = {}
    for wid in ext_ids:
        seqs, scaler = _well_sequences(wells[wid], wid, window)
        folds[wid] = evaluate(
            scaler.inverse_transform_target(seqs["y"]),
            scaler.inverse_transform_target(predict_lstm(model, seqs["xs"], _INFER)),
        )
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides."""
    p = argparse.ArgumentParser(
        description="External validation for physics-informed LSTM"
    )
    p.add_argument(
        "--lambdas", type=str, default="0,0.5", help="Physics weights to evaluate"
    )
    p.add_argument("--epochs", type=int, default=150, help="Max epochs (single model)")
    p.add_argument("--window", type=int, default=WINDOW, help="Depth window length")
    return p.parse_args()


def main() -> None:
    """Run both external-validation protocols for each requested lambda."""
    args = _parse_args()
    lambdas = [float(s) for s in args.lambdas.split(",")]
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=1024, lr=1e-3, patience=12, seed=42
    )

    print("Loading wells...")
    wells = load_field(FIELD_DIR)
    train_pool, external = field_split(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    ext_ids = sorted(external)
    print(f"  External wells: {ext_ids}")

    summary: dict[str, dict] = {"ensemble": {}, "single": {}}
    for lam in lambdas:
        key = f"lambda_{lam}"
        print(f"\n[A] Ensemble — λ={lam} (27 LOWO checkpoints)...")
        summary["ensemble"][key] = _eval_ensemble(
            wells, ext_ids, OUT_ROOT / key / "checkpoints", args.window
        )
        print(f"[B] Single — λ={lam} (train one LSTM on 27 wells)...")
        model = _train_single(train_pool, lam, args.window, cfg)
        summary["single"][key] = _eval_single(model, wells, ext_ids, args.window)

    (OUT_ROOT / "external.json").write_text(json.dumps(summary, indent=2))

    print("\n=== LSTM externo: media sobre 3 pozos ciegos ===")
    print(f"{'Protocolo':<14}{'λ':>6}{'MAE':>10}{'R²':>10}")
    for proto in ("ensemble", "single"):
        for lam in lambdas:
            agg = summary[proto][f"lambda_{lam}"]["aggregate"]
            print(f"{proto:<14}{lam:>6}{agg['mae_mean']:>10.4f}{agg['r2_mean']:>10.4f}")
    print(f"\nSaved to {OUT_ROOT}/external.json")


if __name__ == "__main__":
    main()
