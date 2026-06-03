"""
External validation under two inference protocols.

Evaluates the 3 held-out external wells with baseline (λ=0) and PINN (λ=0.5)
under both protocols, as a robustness check for the PINN advantage:

  A) Ensemble  — average the 27 per-fold LOWO checkpoints (reuses existing
     outputs/checkpoints/{baseline,lambda_0.5}/).
  B) Single    — one final model trained on all 27 train-pool wells
     (saved to outputs/checkpoints/final/).

The PINN advantage is validated by the baseline→PINN delta within EACH protocol.

Saves:
  outputs/external/validation_comparison.json — per-well + aggregate, both protocols
  outputs/figures/external_comparison.png      — grouped MAE/R² bars
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.evaluate import evaluate
from src.external_eval import ensemble_predict, predict_well, train_final_model, train_pool_ids
from src.lowo import field_split
from src.train import TrainConfig

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_DIR = Path("outputs/external")
FIG_DIR = Path("outputs/figures")
N_EXTERNAL = 3
SPLIT_SEED = 42
RECOMMENDED_LAMBDA = 0.5

# Final-model training config (matches the sweep: epochs=500, patience=20, batch=512)
_EPOCHS, _BATCH, _LR, _PATIENCE = 500, 512, 1e-3, 20

BLUE, ORANGE = "#2B6CB0", "#C05621"


def _final_cfg(lam: float) -> TrainConfig:
    """TrainConfig for a single final model with the given physics weight."""
    return TrainConfig(
        epochs=_EPOCHS, batch_size=_BATCH, lr=_LR, patience=_PATIENCE,
        lambda_phys=lam, seed=42,
        checkpoint_dir=Path("outputs/checkpoints/final"),
    )


def _aggregate(folds: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across the external wells (keys suffixed `_mean`)."""
    keys = ["mae", "rmse", "r2", "pe_90"]
    return {f"{k}_mean": float(np.mean([folds[w][k] for w in folds])) for k in keys}


def _eval_ensemble(wells: dict, ext_ids: list[str], valid: set[str], ckpt_dir: Path) -> dict:
    """Evaluate all external wells with the LOWO ensemble in a checkpoint dir."""
    folds = {}
    for wid in ext_ids:
        _, true, pred = ensemble_predict(wells[wid], wid, ckpt_dir, valid)
        folds[wid] = evaluate(true, pred)
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _eval_single(wells: dict, train_pool: dict, ext_ids: list[str], lam: float) -> dict:
    """Train one final model (given λ) and evaluate all external wells."""
    model = train_final_model(train_pool, _final_cfg(lam), well_id=f"final_lambda_{lam}")
    folds = {}
    for wid in ext_ids:
        _, true, pred = predict_well(model, wells[wid], wid, _final_cfg(lam))
        folds[wid] = evaluate(true, pred)
    return {"folds": folds, "aggregate": _aggregate(folds)}


def _plot_comparison(summary: dict, out: Path) -> None:
    """Grouped MAE and R² bars: baseline vs PINN under each protocol."""
    groups = ["Ensemble\n(27 modelos LOWO)", "Modelo único\n(entrenado en 27)"]
    base_mae = [summary["ensemble"]["baseline"]["aggregate"]["mae_mean"],
                summary["single"]["baseline"]["aggregate"]["mae_mean"]]
    pinn_mae = [summary["ensemble"]["pinn"]["aggregate"]["mae_mean"],
                summary["single"]["pinn"]["aggregate"]["mae_mean"]]
    base_r2 = [summary["ensemble"]["baseline"]["aggregate"]["r2_mean"],
               summary["single"]["baseline"]["aggregate"]["r2_mean"]]
    pinn_r2 = [summary["ensemble"]["pinn"]["aggregate"]["r2_mean"],
               summary["single"]["pinn"]["aggregate"]["r2_mean"]]

    x = np.arange(2)
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, base, pinn, ylabel, title in [
        (ax1, base_mae, pinn_mae, "MAE (g/cc)", "MAE en pozos externos  (menor es mejor)"),
        (ax2, base_r2, pinn_r2, "R²", "R² en pozos externos  (mayor es mejor)"),
    ]:
        ax.bar(x - w / 2, base, w, color=BLUE, alpha=0.85, label="Baseline (λ=0)")
        ax.bar(x + w / 2, pinn, w, color=ORANGE, alpha=0.85, label="PINN (λ=0.5)")
        for i in range(2):
            ax.text(x[i] - w / 2, base[i], f"{base[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(x[i] + w / 2, pinn[i], f"{pinn[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Validación externa — el PINN mejora bajo los dos protocolos de inferencia",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load and split
    print("Loading wells...")
    wells = load_field(FIELD_DIR)
    train_pool, external = field_split(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    ext_ids = sorted(external)
    valid = train_pool_ids(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    print(f"  External wells: {ext_ids}")

    # Step 2 — Protocol A: LOWO ensemble (existing checkpoints)
    print("\n[A] Ensemble (27 LOWO models)...")
    summary = {"ensemble": {}, "single": {}}
    summary["ensemble"]["baseline"] = _eval_ensemble(wells, ext_ids, valid, Path("outputs/checkpoints/baseline"))
    summary["ensemble"]["pinn"] = _eval_ensemble(
        wells, ext_ids, valid, Path(f"outputs/checkpoints/lambda_{RECOMMENDED_LAMBDA}")
    )

    # Step 3 — Protocol B: single final model (train fresh on all 27)
    print("[B] Single final model — training baseline (λ=0)...")
    summary["single"]["baseline"] = _eval_single(wells, train_pool, ext_ids, 0.0)
    print("[B] Single final model — training PINN (λ=0.5)...")
    summary["single"]["pinn"] = _eval_single(wells, train_pool, ext_ids, RECOMMENDED_LAMBDA)

    # Step 4 — Save metrics
    (OUT_DIR / "validation_comparison.json").write_text(json.dumps(summary, indent=2))

    # Step 5 — Figure + console table
    _plot_comparison(summary, FIG_DIR / "external_comparison.png")

    print("\n=== Validación externa: media sobre 3 pozos ciegos ===")
    print(f"{'Protocolo':<18}{'MAE base':>10}{'MAE PINN':>10}{'R2 base':>10}{'R2 PINN':>10}")
    for proto, label in [("ensemble", "Ensemble"), ("single", "Modelo único")]:
        b = summary[proto]["baseline"]["aggregate"]
        p = summary[proto]["pinn"]["aggregate"]
        print(f"{label:<18}{b['mae_mean']:>10.4f}{p['mae_mean']:>10.4f}"
              f"{b['r2_mean']:>10.4f}{p['r2_mean']:>10.4f}")
    print(f"\nSaved to {OUT_DIR}/validation_comparison.json")


if __name__ == "__main__":
    main()
