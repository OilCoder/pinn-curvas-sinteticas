"""
Result plots for the PINN vs baseline comparison.

Produces three figure types saved to outputs/figures/:

1. lambda_vs_error.png  — MAE and R² vs lambda (sweep curve)
2. depth_profiles.png   — DEN_true / DEN_base / DEN_pinn for a sample well
                          per best lambda (sorted by baseline MAE descending,
                          picks the worst-performing well as the most informative)
3. crossplot.png        — DEN_pred vs DEN_true scatter for best lambda,
                          colored by well, with the physics line overlaid

Reads:
  outputs/baseline/metrics.json
  outputs/pinn/lambda_sweep.json
  outputs/pinn/lambda_{best_λ}/predictions/*.parquet
  outputs/baseline/predictions/*.parquet

Saves:
  outputs/figures/lambda_vs_error.png
  outputs/figures/depth_profiles.png
  outputs/figures/crossplot.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


BASELINE_PATH = Path("outputs/baseline/metrics.json")
SWEEP_PATH = Path("outputs/pinn/lambda_sweep.json")
PINN_DIR = Path("outputs/pinn")
BASE_PRED_DIR = Path("outputs/baseline/predictions")
OUT_DIR = Path("outputs/figures")
LAMBDA_GRID: list[float] = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _plot_lambda_vs_error(sweep: dict, baseline_agg: dict, out: Path) -> None:
    """Plot MAE and R² mean ± std vs lambda."""
    lambdas, maes, mae_stds, r2s, r2_stds = [], [], [], [], []
    for lam in LAMBDA_GRID:
        s = sweep.get(str(lam))
        if s is None:
            continue
        lambdas.append(lam)
        a = s["aggregate"]
        maes.append(a["mae_mean"])
        mae_stds.append(a["mae_std"])
        r2s.append(a["r2_mean"])
        r2_stds.append(a["r2_std"])

    base_mae = baseline_agg["mae_mean"]
    base_r2 = baseline_agg["r2_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # MAE
    ax1.errorbar(lambdas, maes, yerr=mae_stds, marker="o", color="steelblue", label="PINN")
    ax1.axhline(base_mae, color="gray", linestyle="--", label="Baseline")
    ax1.set_xlabel("λ (physics weight)")
    ax1.set_ylabel("MAE (g/cc)")
    ax1.set_title("MAE vs λ")
    ax1.legend()
    ax1.set_xscale("symlog", linthresh=0.005)

    # R²
    ax2.errorbar(lambdas, r2s, yerr=r2_stds, marker="o", color="darkorange", label="PINN")
    ax2.axhline(base_r2, color="gray", linestyle="--", label="Baseline")
    ax2.set_xlabel("λ (physics weight)")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs λ")
    ax2.legend()
    ax2.set_xscale("symlog", linthresh=0.005)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_depth_profiles(well: str, best_lam: float, out: Path) -> None:
    """Plot depth profile for baseline and best-lambda PINN for one well."""
    base_path = BASE_PRED_DIR / f"{well}.parquet"
    pinn_path = PINN_DIR / f"lambda_{best_lam}" / "predictions" / f"{well}.parquet"

    if not base_path.exists() or not pinn_path.exists():
        print(f"  SKIP depth profile — predictions not found for {well}")
        return

    base_df = pd.read_parquet(base_path)
    pinn_df = pd.read_parquet(pinn_path)

    fig, ax = plt.subplots(figsize=(4, 10))
    ax.plot(base_df["DEN_true"], base_df["DEPTH"], color="black", lw=1.0, label="True DEN")
    ax.plot(base_df["DEN_pred"], base_df["DEPTH"], color="steelblue", lw=0.8,
            alpha=0.8, label="Baseline MLP")
    ax.plot(pinn_df["DEN_pred"], pinn_df["DEPTH"], color="darkorange", lw=0.8,
            alpha=0.8, label=f"PINN λ={best_lam}")
    ax.invert_yaxis()
    ax.set_xlabel("DEN (g/cc)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title(f"Depth profile — {well}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_crossplot(best_lam: float, out: Path) -> None:
    """Scatter DEN_pred vs DEN_true for all available wells at best_lam."""
    pred_dir = PINN_DIR / f"lambda_{best_lam}" / "predictions"
    if not pred_dir.exists():
        print(f"  SKIP crossplot — no predictions for lambda={best_lam}")
        return

    all_true, all_pred = [], []
    for p in sorted(pred_dir.glob("*.parquet")):
        df = pd.read_parquet(p).dropna()
        all_true.append(df["DEN_true"].values)
        all_pred.append(df["DEN_pred"].values)

    if not all_true:
        return

    true_arr = np.concatenate(all_true)
    pred_arr = np.concatenate(all_pred)

    # Physics line in g/cc — approximate from linear fit on data
    lim_lo = max(true_arr.min(), 1.4)
    lim_hi = min(true_arr.max(), 3.2)
    ref = np.linspace(lim_lo, lim_hi, 100)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_arr, pred_arr, s=1, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot(ref, ref, "k--", lw=1.0, label="Perfect prediction")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("DEN true (g/cc)")
    ax.set_ylabel("DEN predicted (g/cc)")
    ax.set_title(f"DEN crossplot — PINN λ={best_lam}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Step 1 — Load data
    # ----------------------------------------
    if not BASELINE_PATH.exists():
        print(f"ERROR: {BASELINE_PATH} not found.")
        sys.exit(1)
    if not SWEEP_PATH.exists():
        print(f"ERROR: {SWEEP_PATH} not found.")
        sys.exit(1)

    baseline = _load_json(BASELINE_PATH)
    sweep = _load_json(SWEEP_PATH)

    # ----------------------------------------
    # Step 2 — Determine best lambda (lowest MAE mean)
    # ----------------------------------------
    best_lam = min(
        (lam for lam in LAMBDA_GRID if str(lam) in sweep),
        key=lambda lv: sweep[str(lv)]["aggregate"]["mae_mean"],
    )
    print(f"Best lambda by MAE: {best_lam}")

    # Worst-performing baseline well (most room for PINN to help)
    valid_folds = {
        w: m for w, m in baseline["folds"].items()
        if m["mae"] == m["mae"]  # exclude NaN
    }
    sample_well = max(valid_folds, key=lambda w: valid_folds[w]["mae"])
    print(f"Sample well for depth profile: {sample_well} (MAE={valid_folds[sample_well]['mae']:.3f})")

    # ----------------------------------------
    # Step 3 — Generate plots
    # ----------------------------------------
    print("\nGenerating figures...")
    _plot_lambda_vs_error(sweep, baseline["aggregate"], OUT_DIR / "lambda_vs_error.png")
    _plot_depth_profiles(sample_well, best_lam, OUT_DIR / "depth_profiles.png")
    _plot_crossplot(best_lam, OUT_DIR / "crossplot.png")

    print("\nAll figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
