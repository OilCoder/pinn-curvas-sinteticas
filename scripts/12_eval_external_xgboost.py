"""
External validation for the physics-informed XGBoost model, two protocols.

Evaluates the 3 held-out external wells (field_split n_external=3, seed=42) under
the same two inference protocols as the MLP study (scripts/10):

  A) Ensemble — average the 27 per-fold LOWO boosters saved by scripts/11
     (outputs/experiments/xgboost/lambda_{λ}/checkpoints/*.ubj).
  B) Single   — one booster trained on all 27 train-pool wells.

Run for λ=0 (data-driven) and λ=0.5 (physics-informed) so the physics delta can
be read within each protocol, mirroring the published baseline→PINN comparison.

Saves:
  outputs/experiments/xgboost/external.json — per-well + aggregate, both protocols
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def _aggregate(folds: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across the external wells (keys suffixed `_mean`)."""
    keys = ["mae", "rmse", "r2", "pe_90"]
    return {f"{k}_mean": float(np.mean([folds[w][k] for w in folds])) for k in keys}


def _eval_ensemble(wells: dict, ext_ids: list[str], ckpt_dir: Path) -> dict:
    """Evaluate external wells by averaging all fold boosters in a checkpoint dir."""
    boosters = [PhysicsXGB().load(c) for c in sorted(ckpt_dir.glob("*.ubj"))]
    if not boosters:
        raise FileNotFoundError(f"No boosters in {ckpt_dir}")
    folds = {}
    for wid in ext_ids:
        proc, scaler = preprocess_well_with_caliper(wells[wid].copy(), wid, fit=True)
        x, y, _, _, _, _ = build_arrays(proc, with_caliper=True)
        ens = np.mean([m.predict(x) for m in boosters], axis=0)
        folds[wid] = evaluate(
            scaler.inverse_transform_target(y), scaler.inverse_transform_target(ens)
        )
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _train_single(train_pool: dict, lam: float, rounds: int) -> PhysicsXGB:
    """Train one booster on the whole train pool (single-model protocol)."""
    xs, ys, nphis, grs, ws = [], [], [], [], []
    for wid, df in train_pool.items():
        proc, _ = preprocess_well_with_caliper(df.copy(), wid, fit=True)
        x, y, nphi, gr, w, _ = build_arrays(proc, with_caliper=True)
        xs.append(x), ys.append(y), nphis.append(nphi), grs.append(gr), ws.append(w)
    return PhysicsXGB(lambda_phys=lam, n_rounds=rounds).fit(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(nphis),
        np.concatenate(grs),
        np.concatenate(ws),
    )


def _eval_single(model: PhysicsXGB, wells: dict, ext_ids: list[str]) -> dict:
    """Evaluate external wells with a single trained booster."""
    folds = {}
    for wid in ext_ids:
        proc, scaler = preprocess_well_with_caliper(wells[wid].copy(), wid, fit=True)
        x, y, _, _, _, _ = build_arrays(proc, with_caliper=True)
        folds[wid] = evaluate(
            scaler.inverse_transform_target(y),
            scaler.inverse_transform_target(model.predict(x)),
        )
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides."""
    p = argparse.ArgumentParser(
        description="External validation for physics-informed XGBoost"
    )
    p.add_argument(
        "--lambdas", type=str, default="0.0,0.5", help="Physics weights to evaluate"
    )
    p.add_argument(
        "--rounds", type=int, default=N_ROUNDS, help="Boosting rounds (single model)"
    )
    return p.parse_args()


def main() -> None:
    """Run both external-validation protocols for each requested lambda."""
    args = _parse_args()
    lambdas = [float(s) for s in args.lambdas.split(",")]

    print("Loading wells...")
    wells = load_field(FIELD_DIR)
    train_pool, external = field_split(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    ext_ids = sorted(external)
    print(f"  External wells: {ext_ids}")

    summary: dict[str, dict] = {"ensemble": {}, "single": {}}
    for lam in lambdas:
        key = f"lambda_{lam}"
        print(f"\n[A] Ensemble — λ={lam} (27 LOWO boosters)...")
        summary["ensemble"][key] = _eval_ensemble(
            wells, ext_ids, OUT_ROOT / key / "checkpoints"
        )
        print(f"[B] Single — λ={lam} (train one booster on 27 wells)...")
        summary["single"][key] = _eval_single(
            _train_single(train_pool, lam, args.rounds), wells, ext_ids
        )

    (OUT_ROOT / "external.json").write_text(json.dumps(summary, indent=2))

    print("\n=== XGBoost externo: media sobre 3 pozos ciegos ===")
    print(f"{'Protocolo':<14}{'λ':>6}{'MAE':>10}{'R²':>10}")
    for proto in ("ensemble", "single"):
        for lam in lambdas:
            agg = summary[proto][f"lambda_{lam}"]["aggregate"]
            print(f"{proto:<14}{lam:>6}{agg['mae_mean']:>10.4f}{agg['r2_mean']:>10.4f}")
    print(f"\nSaved to {OUT_ROOT}/external.json")


if __name__ == "__main__":
    main()
