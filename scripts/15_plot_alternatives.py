"""
Figures for the "alternative models" website tab.

Reads the LOWO sweeps and external-validation JSONs for the three architectures
(MLP/PINN, physics-informed XGBoost, physics-informed LSTM) and renders the
comparison figures used in docs/baselines.html:

  alt_lambda_curves.png   — LOWO MAE vs lambda per architecture (different optima)
  alt_lowo_external.png    — best-lambda LOWO + external MAE/R² bars per architecture
  alt_xgb_importance.png   — XGBoost gain importance, caliper (DCAL_NORM) highlighted

Also prints the exact numbers used in the tab's tables.

Reads (gitignored outputs):
  outputs/baseline, outputs/pinn/lambda_*, outputs/experiments/{xgboost,lstm}/*
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_caliper import FEATURE_COLS_EXT
from src.plot_style import BLUE, GREEN, ORANGE, apply_style

FIG_DIR = Path("docs/assets/figures")
INK = "#1A202C"

# Per-architecture LOWO sweep sources and the lambda grid each one was run on.
MLP_LAMBDAS = [0.0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]
XGB_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0]
LSTM_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]

ARCH_COLOR = {"MLP / PINN": GREEN, "XGBoost": ORANGE, "LSTM": BLUE}

# Operational optimum per architecture. The MLP is fixed to its published value
# (lambda=0.5, chosen by per-well robustness in the main report); XGBoost and
# LSTM use the lowest-mean-MAE lambda from their own sweep (None = argmin).
ARCH_OPTIMUM: dict[str, float | None] = {
    "MLP / PINN": 0.5,
    "XGBoost": None,
    "LSTM": None,
}


def _agg(path: Path) -> dict[str, float]:
    """Load the 'aggregate' block of a metrics.json."""
    return json.loads(path.read_text())["aggregate"]


def _mlp_metrics(lam: float) -> dict[str, float]:
    """MLP aggregate at a given lambda (lambda=0 lives under outputs/pinn too)."""
    return _agg(Path(f"outputs/pinn/lambda_{lam}/metrics.json"))


def _exp_metrics(model: str, lam: float) -> dict[str, float]:
    """Aggregate for an experiment model (xgboost/lstm) at a given lambda."""
    return _agg(Path(f"outputs/experiments/{model}/lambda_{lam}/metrics.json"))


def _curve(getter, lambdas: list[float]) -> tuple[list[float], list[float]]:
    """Return (lambdas, MAE_mean) for the lambdas that have results on disk."""
    xs, ys = [], []
    for lam in lambdas:
        try:
            ys.append(getter(lam)["mae_mean"])
            xs.append(lam)
        except FileNotFoundError:
            continue
    return xs, ys


def _best_lambda(getter, lambdas: list[float], fixed: float | None) -> float:
    """Operational lambda: ``fixed`` if pinned, else lowest-mean-MAE on disk."""
    if fixed is not None:
        return fixed
    xs, ys = _curve(getter, lambdas)
    return xs[int(np.argmin(ys))]


def plot_lambda_curves() -> None:
    """LOWO MAE vs lambda for the three architectures, optimum marked."""
    curves = {
        "MLP / PINN": _curve(_mlp_metrics, MLP_LAMBDAS),
        "XGBoost": _curve(lambda v: _exp_metrics("xgboost", v), XGB_LAMBDAS),
        "LSTM": _curve(lambda v: _exp_metrics("lstm", v), LSTM_LAMBDAS),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, (xs, ys) in curves.items():
        color = ARCH_COLOR[name]
        ax.plot(xs, ys, "-o", color=color, label=name, lw=2, ms=5)
        fixed = ARCH_OPTIMUM[name]
        best_i = (
            xs.index(fixed) if fixed is not None and fixed in xs else int(np.argmin(ys))
        )
        ax.plot(xs[best_i], ys[best_i], "*", color=color, ms=16, zorder=5)
        ax.annotate(
            f"λ*={xs[best_i]:g}",
            (xs[best_i], ys[best_i]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=9,
            color=color,
        )
    ax.set_xlabel("Peso de la física  λ")
    ax.set_ylabel("MAE en LOWO (g/cc)  ·  menor es mejor")
    ax.set_title(
        "El λ óptimo depende de la arquitectura", fontsize=12, fontweight="semibold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alt_lambda_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved alt_lambda_curves.png")


def _ext(model: str, proto: str, lam: float, metric: str) -> float:
    """External metric for an experiment model under a protocol/lambda."""
    data = json.loads(Path(f"outputs/experiments/{model}/external.json").read_text())
    return data[proto][f"lambda_{lam}"]["aggregate"][f"{metric}_mean"]


def _ext_mlp(proto: str, model_key: str, metric: str) -> float:
    """External metric for the published MLP comparison (baseline/pinn)."""
    data = json.loads(Path("outputs/external/validation_comparison.json").read_text())
    return data[proto][model_key]["aggregate"][f"{metric}_mean"]


def plot_lowo_external(xgb_best: float, lstm_best: float) -> None:
    """Grouped bars: LOWO and external MAE per architecture, lambda=0 vs best."""
    mlp_best = ARCH_OPTIMUM["MLP / PINN"]
    # (label, lowo_l0, lowo_best, ext_ens_l0, ext_ens_best)
    rows = [
        (
            f"MLP / PINN\n(λ*={mlp_best:g})",
            _mlp_metrics(0.0)["mae_mean"],
            _mlp_metrics(mlp_best)["mae_mean"],
            _ext_mlp("ensemble", "baseline", "mae"),
            _ext_mlp("ensemble", "pinn", "mae"),
        ),
        (
            f"XGBoost\n(λ*={xgb_best:g})",
            _exp_metrics("xgboost", 0.0)["mae_mean"],
            _exp_metrics("xgboost", xgb_best)["mae_mean"],
            _ext("xgboost", "ensemble", 0.0, "mae"),
            _ext("xgboost", "ensemble", xgb_best, "mae"),
        ),
        (
            f"LSTM\n(λ*={lstm_best:g})",
            _exp_metrics("lstm", 0.0)["mae_mean"],
            _exp_metrics("lstm", lstm_best)["mae_mean"],
            _ext("lstm", "ensemble", 0.0, "mae"),
            _ext("lstm", "ensemble", lstm_best, "mae"),
        ),
    ]
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, i0, i1, title in [
        (ax1, 1, 2, "LOWO  ·  27 pozos de entrenamiento"),
        (ax2, 3, 4, "Externo (ensemble)  ·  3 pozos ciegos"),
    ]:
        d0 = [r[i0] for r in rows]
        d1 = [r[i1] for r in rows]
        ax.bar(x - w / 2, d0, w, color="#9CA3AF", label="sin física (λ=0)")
        ax.bar(x + w / 2, d1, w, color=GREEN, label="con su mejor física (λ*)")
        for i in range(len(rows)):
            ax.text(
                x[i] - w / 2,
                d0[i],
                f"{d0[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x[i] + w / 2,
                d1[i],
                f"{d1[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("MAE (g/cc)  ·  menor es mejor")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Cada arquitectura con su propio λ óptimo", fontsize=12, fontweight="semibold"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alt_lowo_external.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved alt_lowo_external.png")


def plot_xgb_importance() -> None:
    """Average gain importance of the 27 XGBoost LOWO boosters (lambda=0)."""
    ckpts = sorted(
        Path("outputs/experiments/xgboost/lambda_0.0/checkpoints").glob("*.ubj")
    )
    totals = np.zeros(len(FEATURE_COLS_EXT))
    for c in ckpts:
        b = xgb.Booster()
        b.load_model(str(c))
        score = b.get_score(importance_type="gain")
        for k, v in score.items():
            totals[int(k[1:])] += v
    totals /= max(len(ckpts), 1)
    order = np.argsort(totals)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [BLUE if FEATURE_COLS_EXT[i] != "DCAL_NORM" else ORANGE for i in order]
    ax.barh([FEATURE_COLS_EXT[i] for i in order], totals[order], color=colors)
    ax.set_xlabel("Importancia media (gain)  ·  27 modelos LOWO")
    ax.set_title("Qué mira XGBoost  ·  el caliper (DCAL_NORM) destacado", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alt_xgb_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved alt_xgb_importance.png")


def _corr_mae(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Return (correlation, MAE) on the finite overlap of two curves."""
    m = ~np.isnan(true) & ~np.isnan(pred)
    return float(np.corrcoef(pred[m], true[m])[0, 1]), float(
        np.abs(pred[m] - true[m]).mean()
    )


def plot_external_profiles(model: str, lam: float, color: str, title: str) -> None:
    """Depth profiles of the 3 blind wells: real DEN vs single-model best-λ prediction."""
    pred_dir = Path(f"outputs/experiments/{model}/external_predictions")
    files = sorted(pred_dir.glob(f"*_lambda_{lam}.parquet"))
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 9), sharey=False)
    for ax, path in zip(axes, files):
        wid = path.name.replace(f"_lambda_{lam}.parquet", "")
        df = pd.read_parquet(path)
        true, pred, depth = (
            df["DEN_true"].values,
            df["DEN_pred"].values,
            df["DEPTH"].values,
        )
        corr, mae = _corr_mae(true, pred)
        ax.plot(true, depth, color=INK, lw=1.1, label="DEN real")
        ax.plot(
            pred, depth, color=color, lw=0.8, alpha=0.85, label=f"{title} (λ={lam:g})"
        )
        ax.invert_yaxis()
        ax.set_xlabel("DEN (g/cc)")
        ax.set_title(f"{wid}\nMAE {mae:.3f}  ·  corr {corr:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Profundidad (ft)")
    fig.suptitle(
        f"{title} en los 3 pozos ciegos  ·  modelo único, λ={lam:g}",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    out = FIG_DIR / f"alt_profiles_{model}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def print_tables(xgb_best: float, lstm_best: float) -> None:
    """Print the exact numbers used in the tab's tables."""
    mlp_best = ARCH_OPTIMUM["MLP / PINN"]
    print("\n=== LOWO MAE / R² (λ=0 → mejor λ) ===")
    print(
        f"  MLP    : {_mlp_metrics(0.0)['mae_mean']:.4f}/{_mlp_metrics(0.0)['r2_mean']:.3f}"
        f" → {_mlp_metrics(mlp_best)['mae_mean']:.4f}/{_mlp_metrics(mlp_best)['r2_mean']:.3f}  (λ*={mlp_best:g})"
    )
    print(
        f"  XGBoost: {_exp_metrics('xgboost', 0.0)['mae_mean']:.4f}/{_exp_metrics('xgboost', 0.0)['r2_mean']:.3f}"
        f" → {_exp_metrics('xgboost', xgb_best)['mae_mean']:.4f}/{_exp_metrics('xgboost', xgb_best)['r2_mean']:.3f}  (λ*={xgb_best:g})"
    )
    print(
        f"  LSTM   : {_exp_metrics('lstm', 0.0)['mae_mean']:.4f}/{_exp_metrics('lstm', 0.0)['r2_mean']:.3f}"
        f" → {_exp_metrics('lstm', lstm_best)['mae_mean']:.4f}/{_exp_metrics('lstm', lstm_best)['r2_mean']:.3f}  (λ*={lstm_best:g})"
    )
    print("\n=== Externo ensemble / single (MAE, R²) ===")
    for model, best in [("xgboost", xgb_best), ("lstm", lstm_best)]:
        for proto in ("ensemble", "single"):
            print(
                f"  {model:<8} {proto:<9} λ0: {_ext(model, proto, 0.0, 'mae'):.4f}/{_ext(model, proto, 0.0, 'r2'):.3f}"
                f"  λ*: {_ext(model, proto, best, 'mae'):.4f}/{_ext(model, proto, best, 'r2'):.3f}"
            )


def main() -> None:
    """Render all figures and print the numbers for the tab."""
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Derive each architecture's operational lambda once (argmin MAE, or pinned).
    xgb_best = _best_lambda(lambda v: _exp_metrics("xgboost", v), XGB_LAMBDAS, None)
    lstm_best = _best_lambda(lambda v: _exp_metrics("lstm", v), LSTM_LAMBDAS, None)
    plot_lambda_curves()
    plot_lowo_external(xgb_best, lstm_best)
    plot_xgb_importance()
    plot_external_profiles("xgboost", xgb_best, ORANGE, "XGBoost")
    plot_external_profiles("lstm", lstm_best, BLUE, "LSTM")
    print_tables(xgb_best, lstm_best)
    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
