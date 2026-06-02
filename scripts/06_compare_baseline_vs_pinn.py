"""
Paired comparison of baseline MLP vs PINN across all lambda values.

For each well and each lambda, computes:
  delta_mae  = baseline_mae  - pinn_mae   (positive = PINN improves)
  delta_rmse = baseline_rmse - pinn_rmse
  delta_r2   = pinn_r2       - baseline_r2

Aggregates fraction of wells improved, mean delta, and best lambda per metric.

Reads:
  outputs/baseline/metrics.json
  outputs/pinn/lambda_{λ}/metrics.json  (for each λ in LAMBDA_GRID)

Saves:
  outputs/figures/comparison_table.csv  — per-well per-lambda deltas
  outputs/figures/comparison_summary.json — aggregate per lambda
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

BASELINE_PATH = Path("outputs/baseline/metrics.json")
PINN_DIR = Path("outputs/pinn")
OUT_DIR = Path("outputs/figures")
LAMBDA_GRID: list[float] = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]


def _load_fold_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load per-fold metrics dict from a metrics.json file."""
    with open(path) as f:
        return json.load(f)["folds"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Step 1 — Load baseline
    # ----------------------------------------
    if not BASELINE_PATH.exists():
        print(f"ERROR: {BASELINE_PATH} not found. Run 03_train_baseline.py first.")
        sys.exit(1)

    baseline = _load_fold_metrics(BASELINE_PATH)
    wells = sorted(baseline.keys())
    print(f"Baseline: {len(wells)} wells")

    # ----------------------------------------
    # Step 2 — Load per-lambda results and compute deltas
    # ----------------------------------------
    rows = []
    for lam in LAMBDA_GRID:
        path = PINN_DIR / f"lambda_{lam}" / "metrics.json"
        if not path.exists():
            print(f"  SKIP lambda={lam} — {path} not found")
            continue

        pinn = _load_fold_metrics(path)
        for well in wells:
            if well not in pinn or well not in baseline:
                continue
            b = baseline[well]
            p = pinn[well]
            # Skip folds with NaN
            if any(v != v for v in list(b.values()) + list(p.values())):
                continue
            rows.append({
                "well":        well,
                "lambda_phys": lam,
                "base_mae":    b["mae"],
                "pinn_mae":    p["mae"],
                "delta_mae":   b["mae"] - p["mae"],
                "base_rmse":   b["rmse"],
                "pinn_rmse":   p["rmse"],
                "delta_rmse":  b["rmse"] - p["rmse"],
                "base_r2":     b["r2"],
                "pinn_r2":     p["r2"],
                "delta_r2":    p["r2"] - b["r2"],
            })

    if not rows:
        print("No valid comparison rows found.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # ----------------------------------------
    # Step 3 — Save comparison table
    # ----------------------------------------
    csv_path = OUT_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Comparison table: {len(df)} rows → {csv_path}")

    # ----------------------------------------
    # Step 4 — Aggregate per lambda
    # ----------------------------------------
    summary: dict[str, dict] = {}
    print("\n=== Paired Comparison: baseline vs PINN ===")
    print(f"{'λ':>6} {'ΔMAE mean':>11} {'% improved MAE':>15} {'ΔR² mean':>10}")
    print("-" * 50)

    for lam in LAMBDA_GRID:
        sub = df[df["lambda_phys"] == lam]
        if sub.empty:
            continue
        n = len(sub)
        n_improved_mae = int((sub["delta_mae"] > 0).sum())
        mean_delta_mae = float(sub["delta_mae"].mean())
        mean_delta_r2 = float(sub["delta_r2"].mean())
        pct_improved = 100 * n_improved_mae / n

        summary[str(lam)] = {
            "lambda_phys":      lam,
            "n_wells":          n,
            "delta_mae_mean":   mean_delta_mae,
            "delta_mae_std":    float(sub["delta_mae"].std()),
            "pct_improved_mae": pct_improved,
            "delta_r2_mean":    mean_delta_r2,
            "delta_r2_std":     float(sub["delta_r2"].std()),
        }
        print(f"{lam:>6.2f} {mean_delta_mae:>+11.5f} {pct_improved:>14.1f}% {mean_delta_r2:>+10.4f}")

    print("-" * 50)

    # Best lambda by MAE improvement
    best_lam = max(summary.items(), key=lambda x: x[1]["delta_mae_mean"])
    print(f"\nBest λ by MAE improvement: {best_lam[0]}  (ΔMAE={best_lam[1]['delta_mae_mean']:+.5f})")

    summary_path = OUT_DIR / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
