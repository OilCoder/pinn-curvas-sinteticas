"""
Diagnostic depth-profile figures for the PINN portfolio.

Produces two figures that answer "is the limitation the model or generalization?":

1. external_profiles.png — depth profiles of ALL 3 external blind wells
   (Arensman_2, Burmeister_1, Rous_'F'_2), baseline vs PINN λ=0.5, using the
   27-model LOWO ensemble. These wells were never seen at any training stage.

2. insample_vs_oos.png — the SAME model (a LOWO fold) predicting on a well it
   trained on (in-sample) vs the held-out test well (out-of-sample). Shows the
   model fits training data well (high correlation) — isolating the gap as
   generalization, not model capacity.

Saves to outputs/figures/ and copies to documentation/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_field
from src.external_eval import ensemble_predict, predict_well, train_pool_ids
from src.lowo import field_split
from src.model import MLP

FIELD_DIR = Path("data/raw/Kraft Prusa")
OUT_DIR = Path("outputs/figures")
N_EXTERNAL = 3
SPLIT_SEED = 42
BEST_LAMBDA = 0.5

BLUE, ORANGE, INK = "#2B6CB0", "#C05621", "#1A202C"


def _corr_stats(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    """Return (correlation, std ratio pred/true, MAE) on the finite overlap."""
    m = ~np.isnan(pred) & ~np.isnan(true)
    p, t = pred[m], true[m]
    corr = float(np.corrcoef(p, t)[0, 1])
    return corr, float(p.std() / t.std()), float(np.abs(p - t).mean())


def plot_external_profiles(wells: dict, out: Path) -> None:
    """Depth profiles of all 3 external blind wells: baseline vs PINN λ=0.5 ensemble."""
    _, external = field_split(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    ext_ids = sorted(external)
    valid = train_pool_ids(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)

    fig, axes = plt.subplots(1, len(ext_ids), figsize=(4.2 * len(ext_ids), 9), sharey=False)
    for ax, wid in zip(axes, ext_ids):
        depth, true, pred_base = ensemble_predict(
            wells[wid], wid, Path("outputs/checkpoints/baseline"), valid)
        _, _, pred_pinn = ensemble_predict(
            wells[wid], wid, Path(f"outputs/checkpoints/lambda_{BEST_LAMBDA}"), valid)
        corr_b, _, mae_b = _corr_stats(pred_base, true)
        corr_p, _, mae_p = _corr_stats(pred_pinn, true)

        ax.plot(true, depth, color=INK, lw=1.1, label="DEN real")
        ax.plot(pred_base, depth, color=BLUE, lw=0.8, alpha=0.75, label="Baseline (λ=0)")
        ax.plot(pred_pinn, depth, color=ORANGE, lw=0.8, alpha=0.85, label=f"PINN (λ={BEST_LAMBDA})")
        ax.invert_yaxis()
        ax.set_xlabel("DEN (g/cc)")
        ax.set_title(f"{wid}\nMAE {mae_b:.3f}→{mae_p:.3f}  ·  corr {corr_b:.2f}→{corr_p:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Profundidad (ft)")
    fig.suptitle("Validación externa — 3 pozos ciegos (ensemble LOWO de 27 modelos)",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_external_profiles_single(wells: dict, out: Path) -> None:
    """Depth profiles of the 3 external blind wells with the SINGLE final model.

    Uses the two models trained on all 27 wells (baseline λ=0 and PINN λ=0.5),
    saved by scripts/10_external_validation.py to outputs/checkpoints/final/.
    """
    _, external = field_split(wells, n_external=N_EXTERNAL, seed=SPLIT_SEED)
    ext_ids = sorted(external)

    base = MLP()
    base.load_state_dict(torch.load(
        "outputs/checkpoints/final/final_lambda_0.0_best.pt", map_location="cpu", weights_only=True))
    pinn = MLP()
    pinn.load_state_dict(torch.load(
        f"outputs/checkpoints/final/final_lambda_{BEST_LAMBDA}_best.pt", map_location="cpu", weights_only=True))

    fig, axes = plt.subplots(1, len(ext_ids), figsize=(4.2 * len(ext_ids), 13), sharey=False)
    for ax, wid in zip(axes, ext_ids):
        depth, true, pred_base = predict_well(base, wells[wid], wid)
        _, _, pred_pinn = predict_well(pinn, wells[wid], wid)
        corr_b, _, mae_b = _corr_stats(pred_base, true)
        corr_p, _, mae_p = _corr_stats(pred_pinn, true)

        ax.plot(true, depth, color=INK, lw=1.1, label="DEN real")
        ax.plot(pred_base, depth, color=BLUE, lw=0.8, alpha=0.75, label="Baseline (λ=0)")
        ax.plot(pred_pinn, depth, color=ORANGE, lw=0.8, alpha=0.85, label=f"PINN (λ={BEST_LAMBDA})")
        ax.invert_yaxis()
        ax.set_xlabel("DEN (g/cc)")
        ax.set_title(f"{wid}\nMAE {mae_b:.3f}→{mae_p:.3f}  ·  corr {corr_b:.2f}→{corr_p:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Profundidad (ft)")
    fig.suptitle("Validación externa — 3 pozos ciegos (modelo único entrenado en 27)",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_insample_vs_oos(wells: dict, out: Path) -> None:
    """Same model: in-sample fit (well it trained on) vs out-of-sample (test well)."""
    # Model from the fold where Soeken_12 was the test well (trained on 26 wells incl. Oeser)
    fold_test = "Soeken_12"
    in_sample = "Oeser,_R__1"
    ckpt = Path("outputs/checkpoints/baseline") / f"{fold_test}_best.pt"
    model = MLP()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 9))
    for ax, wid, tag in [(ax1, in_sample, "IN-SAMPLE (lo vio al entrenar)"),
                          (ax2, fold_test, "OUT-OF-SAMPLE (pozo de prueba)")]:
        depth, true, pred = predict_well(model, wells[wid], wid)
        corr, sr, mae = _corr_stats(pred, true)
        ax.plot(true, depth, color=INK, lw=1.1, label="DEN real")
        ax.plot(pred, depth, color=ORANGE, lw=0.9, alpha=0.85, label="Predicción")
        ax.invert_yaxis()
        ax.set_xlabel("DEN (g/cc)")
        ax.set_title(f"{wid}\n{tag}\ncorr={corr:.2f}  MAE={mae:.3f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("Profundidad (ft)")
    fig.suptitle("Diagnóstico — el modelo ajusta lo que vio; el reto es generalizar",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wells = load_field(FIELD_DIR)
    print("Generating diagnostic figures...")
    plot_external_profiles(wells, OUT_DIR / "external_profiles.png")
    plot_external_profiles_single(wells, OUT_DIR / "external_profiles_single.png")
    plot_insample_vs_oos(wells, OUT_DIR / "insample_vs_oos.png")
    print("Done.")


if __name__ == "__main__":
    main()
