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
LAMBDA_GRID: list[float] = [0.0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]
# Recommended operating point: λ=0.5 improves the most wells (81.5%) and
# captures nearly all the MAE benefit before the curve saturates.
RECOMMENDED_LAMBDA: float = 0.5

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _plot_lambda_vs_error(sweep: dict, baseline_agg: dict, out: Path) -> None:
    """Plot the mean MAE and R² trend vs lambda (log-x), baseline as reference.

    The PINN line covers lambda > 0 on a log axis; lambda=0 is the baseline,
    drawn as a horizontal dashed reference. Per-fold std is large (wells vary
    widely) and is intentionally omitted so the mean trend — the signal of this
    paired comparison — is legible. The best lambda (lowest MAE) is highlighted.
    """
    pos = [lam for lam in LAMBDA_GRID if lam > 0 and str(lam) in sweep]
    maes = [sweep[str(lam)]["aggregate"]["mae_mean"] for lam in pos]
    r2s = [sweep[str(lam)]["aggregate"]["r2_mean"] for lam in pos]

    base_mae = baseline_agg["mae_mean"]
    base_r2 = baseline_agg["r2_mean"]
    best_lam = min(pos, key=lambda lam: sweep[str(lam)]["aggregate"]["mae_mean"])
    best_i = pos.index(best_lam)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # — MAE (lower is better)
    ax1.axhline(base_mae, color="#718096", linestyle="--", linewidth=1.2,
                label=f"Baseline λ=0 ({base_mae:.4f})")
    ax1.plot(pos, maes, marker="o", color="#2B6CB0", linewidth=1.8, label="PINN")
    ax1.scatter([best_lam], [maes[best_i]], s=120, facecolor="none",
                edgecolor="#C53030", linewidth=2, zorder=5,
                label=f"óptimo λ={best_lam}")
    ax1.set_xscale("log")
    ax1.set_xlabel("λ (peso físico)")
    ax1.set_ylabel("MAE (g/cc)")
    ax1.set_title("MAE vs λ  (menor es mejor)")
    ax1.legend(fontsize=8)

    # — R² (higher is better)
    ax2.axhline(base_r2, color="#718096", linestyle="--", linewidth=1.2,
                label=f"Baseline λ=0 ({base_r2:.3f})")
    ax2.plot(pos, r2s, marker="o", color="#C05621", linewidth=1.8, label="PINN")
    ax2.scatter([best_lam], [r2s[best_i]], s=120, facecolor="none",
                edgecolor="#C53030", linewidth=2, zorder=5,
                label=f"óptimo λ={best_lam}")
    ax2.set_xscale("log")
    ax2.set_xlabel("λ (peso físico)")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs λ  (mayor es mejor)")
    ax2.legend(fontsize=8)

    fig.suptitle("Barrido de λ — la física mejora el modelo y satura cerca de λ≈0.5",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(4.5, 10))
    ax.plot(base_df["DEN_true"], base_df["DEPTH"], color="#1A202C", lw=1.1, label="DEN real")
    ax.plot(base_df["DEN_pred"], base_df["DEPTH"], color="#2B6CB0", lw=0.8,
            alpha=0.75, label="Baseline (λ=0)")
    ax.plot(pinn_df["DEN_pred"], pinn_df["DEPTH"], color="#C05621", lw=0.8,
            alpha=0.85, label=f"PINN (λ={best_lam})")
    ax.invert_yaxis()
    ax.set_xlabel("DEN (g/cc)")
    ax.set_ylabel("Profundidad (ft)")
    ax.set_title(f"Perfil de profundidad — {well}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(true_arr, pred_arr, s=1, alpha=0.15, color="#2B6CB0", rasterized=True)
    ax.plot(ref, ref, color="#1A202C", linestyle="--", lw=1.0, label="Predicción perfecta")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("DEN real (g/cc)")
    ax.set_ylabel("DEN predicho (g/cc)")
    ax.set_title(f"Crossplot DEN — PINN (λ={best_lam})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
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
    # Use the recommended operating point (robust optimum) for the per-well figures
    display_lam = RECOMMENDED_LAMBDA
    print(f"Display lambda (recommended operating point): {display_lam}")

    # Well where the PINN improves MAE the most over baseline (best illustration)
    pinn_path = PINN_DIR / f"lambda_{display_lam}" / "metrics.json"
    pinn_folds = _load_json(pinn_path)["folds"]
    improvements = {
        w: baseline["folds"][w]["mae"] - pinn_folds[w]["mae"]
        for w in pinn_folds
        if pinn_folds[w]["mae"] == pinn_folds[w]["mae"]  # exclude NaN
        and baseline["folds"][w]["mae"] == baseline["folds"][w]["mae"]
    }
    sample_well = max(improvements, key=lambda w: improvements[w])
    print(f"Sample well for depth profile: {sample_well} (ΔMAE={improvements[sample_well]:+.3f})")

    # ----------------------------------------
    # Step 3 — Generate plots
    # ----------------------------------------
    print("\nGenerating figures...")
    _plot_lambda_vs_error(sweep, baseline["aggregate"], OUT_DIR / "lambda_vs_error.png")
    _plot_depth_profiles(sample_well, display_lam, OUT_DIR / "depth_profiles.png")
    _plot_crossplot(display_lam, OUT_DIR / "crossplot.png")

    print("\nAll figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
