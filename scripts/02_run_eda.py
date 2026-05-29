"""
EDA script for the Kraft Prusa well log dataset.

Generates distribution plots, crossplots, and per-well quality summaries.
Outputs are saved to outputs/eda/.

Run from the project root:
    python scripts/run_eda.py [--field-dir data/raw] [--out-dir outputs/eda]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import CANONICAL_CURVES, load_field

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["GR", "RT", "RILM", "NPHI", "SP"]
TARGET_COL = "DEN"


# ----------------------------------------
# Helpers
# ----------------------------------------

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _concat_all(wells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all wells, adding a well_id column."""
    frames = []
    for wid, df in wells.items():
        tmp = df.copy()
        tmp["well_id"] = wid
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------
# Per-well quality table
# ----------------------------------------

def well_quality_table(wells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a per-well summary with row count, depth range, and mean values.

    Args:
        wells: Dict mapping well_id to clean DataFrame.

    Returns:
        DataFrame with one row per well.
    """
    rows = []
    for wid, df in sorted(wells.items()):
        row = {
            "well_id": wid,
            "n_rows": len(df),
            "depth_min_ft": df["DEPTH"].min(),
            "depth_max_ft": df["DEPTH"].max(),
        }
        for col in CANONICAL_CURVES:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------
# Distribution plots
# ----------------------------------------

def plot_distributions(all_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot histograms for each canonical curve across all wells.

    Args:
        all_df: Concatenated DataFrame from all wells.
        out_dir: Directory to save figures.
    """
    cols = CANONICAL_CURVES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        data = all_df[col].dropna()
        ax.hist(data, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplot
    if len(cols) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle("Distribution of canonical curves (all wells)", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir / "distributions_raw.png")


def plot_log_rt_comparison(all_df: pd.DataFrame, out_dir: Path) -> None:
    """Compare raw vs log10 RT distributions to justify the transform.

    Args:
        all_df: Concatenated DataFrame from all wells.
        out_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ("RT", "RILM")):
        raw = all_df[col].dropna()
        log_vals = np.log10(raw.clip(lower=1e-4))

        ax.hist(raw, bins=80, alpha=0.6, label="raw", color="tomato")
        ax2 = ax.twinx()
        ax2.hist(log_vals, bins=80, alpha=0.6, label="log₁₀", color="steelblue")
        ax.set_title(f"{col}: raw vs log₁₀")
        ax.set_xlabel(col)
        ax.spines[["top"]].set_visible(False)

    fig.suptitle("Resistivity: raw vs log₁₀ transform", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "log_rt_comparison.png")


# ----------------------------------------
# Crossplot DEN vs NPHI
# ----------------------------------------

def plot_den_nphi_crossplot(
    all_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[float, float, float]:
    """Crossplot DEN vs NPHI with linear regression fit.

    Args:
        all_df: Concatenated DataFrame from all wells.
        out_dir: Directory to save figures.

    Returns:
        Tuple of (slope a, intercept b, r_value) for DEN = a·NPHI + b.
    """
    data = all_df[["NPHI", "DEN"]].dropna()
    nphi = data["NPHI"].values
    den = data["DEN"].values

    slope, intercept, r_value, p_value, _ = stats.linregress(nphi, den)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(nphi, den, alpha=0.05, s=3, color="steelblue", rasterized=True)

    x_line = np.linspace(nphi.min(), nphi.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label=f"DEN = {slope:.3f}·NPHI + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}")

    ax.set_xlabel("NPHI (v/v)")
    ax.set_ylabel("DEN (g/cc)")
    ax.set_title("DEN vs NPHI — all wells (linear fit)")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "den_nphi_crossplot.png")

    logger.info("DEN–NPHI fit: a=%.4f  b=%.4f  R²=%.4f", slope, intercept, r_value**2)
    return float(slope), float(intercept), float(r_value)


def plot_den_nphi_by_well(wells: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Small-multiples crossplot DEN vs NPHI per well.

    Args:
        wells: Dict mapping well_id to DataFrame.
        out_dir: Directory to save figures.
    """
    n = len(wells)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes = axes.flatten()

    for ax, (wid, df) in zip(axes, sorted(wells.items())):
        data = df[["NPHI", "DEN"]].dropna()
        ax.scatter(data["NPHI"], data["DEN"], alpha=0.3, s=2, color="steelblue", rasterized=True)
        if len(data) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(data["NPHI"], data["DEN"])
            x_line = np.linspace(data["NPHI"].min(), data["NPHI"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1)
        ax.set_title(wid, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("DEN vs NPHI by well", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "den_nphi_by_well.png")


# ----------------------------------------
# Per-well boxplots
# ----------------------------------------

def plot_per_well_boxplots(wells: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Boxplot per curve showing the value distribution across all wells.

    Each box = one well's IQR. Outlier dots show the spread across wells.
    This reveals inter-well calibration differences and flags wells with
    anomalous ranges before normalization.

    Args:
        wells: Dict mapping well_id to clean DataFrame.
        out_dir: Directory to save figures.
    """
    cols = CANONICAL_CURVES
    well_ids = sorted(wells)
    n_curves = len(cols)

    fig, axes = plt.subplots(n_curves, 1, figsize=(max(12, len(well_ids) * 0.4), n_curves * 3))

    for ax, col in zip(axes, cols):
        data_per_well = [wells[wid][col].dropna().values for wid in well_ids]
        ax.boxplot(data_per_well, labels=well_ids, patch_artist=True,
                   boxprops={"facecolor": "steelblue", "alpha": 0.5},
                   medianprops={"color": "firebrick", "linewidth": 1.5},
                   flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                   whiskerprops={"linewidth": 0.8},
                   capprops={"linewidth": 0.8})
        ax.set_ylabel(col, fontsize=9)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-well value distribution — raw units (NPHI in v/v)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "per_well_boxplots.png")


# ----------------------------------------
# Depth profiles (sample wells)
# ----------------------------------------

def plot_depth_profiles(wells: dict[str, pd.DataFrame], out_dir: Path, n_sample: int = 4) -> None:
    """Log-style depth profiles for a sample of wells.

    Args:
        wells: Dict mapping well_id to DataFrame.
        out_dir: Directory to save figures.
        n_sample: Number of wells to plot.
    """
    sample_ids = sorted(wells)[:n_sample]
    cols = FEATURE_COLS + [TARGET_COL]
    n_tracks = len(cols)

    for wid in sample_ids:
        df = wells[wid].copy()
        fig, axes = plt.subplots(1, n_tracks, figsize=(n_tracks * 2, 10), sharey=True)

        for ax, col in zip(axes, cols):
            ax.plot(df[col], df["DEPTH"], linewidth=0.6, color="steelblue")
            ax.set_xlabel(col, fontsize=8)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel("Depth (ft)", fontsize=8)
        fig.suptitle(f"Well: {wid}", fontsize=11)
        fig.tight_layout()
        _save(fig, out_dir / f"profile_{wid}.png")


# ----------------------------------------
# Main
# ----------------------------------------

def main(field_dir: Path, out_dir: Path) -> None:
    """Run the full EDA pipeline.

    Args:
        field_dir: Directory containing LAS files.
        out_dir: Output directory for figures and tables.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Load data
    # ----------------------------------------
    logger.info("Loading wells from %s", field_dir)
    wells = load_field(field_dir)
    if not wells:
        logger.error("No wells loaded. Check field_dir path.")
        return

    logger.info("Loaded %d wells", len(wells))
    all_df = _concat_all(wells)
    logger.info("Total rows: %d", len(all_df))

    # ----------------------------------------
    # Per-well quality table
    # ----------------------------------------
    quality_df = well_quality_table(wells)
    quality_path = out_dir / "well_quality.csv"
    quality_df.to_csv(quality_path, index=False)
    logger.info("Saved %s", quality_path)

    # ----------------------------------------
    # Distribution plots
    # ----------------------------------------
    plot_distributions(all_df, out_dir)
    plot_log_rt_comparison(all_df, out_dir)

    # ----------------------------------------
    # DEN–NPHI crossplot + calibration
    # ----------------------------------------
    slope, intercept, r_value = plot_den_nphi_crossplot(all_df, out_dir)
    plot_den_nphi_by_well(wells, out_dir)

    # Save physics coefficients
    coeff_path = out_dir / "den_nphi_coefficients.csv"
    pd.DataFrame([{
        "a_slope": slope,
        "b_intercept": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
        "n_points": int(all_df[["NPHI", "DEN"]].dropna().__len__()),
    }]).to_csv(coeff_path, index=False)
    logger.info("Saved physics coefficients → %s", coeff_path)

    # ----------------------------------------
    # Per-well boxplots — shows inter-well range variation
    # ----------------------------------------
    plot_per_well_boxplots(wells, out_dir)

    # ----------------------------------------
    # Depth profiles
    # ----------------------------------------
    plot_depth_profiles(wells, out_dir)

    # ----------------------------------------
    # Summary statistics (raw + skewness)
    # ----------------------------------------
    stats_rows = []
    for col in CANONICAL_CURVES:
        s = all_df[col].describe()
        skew = float(all_df[col].dropna().skew())
        stats_rows.append({
            "curve": col,
            "count": int(s["count"]),
            "mean": s["mean"],
            "std": s["std"],
            "skewness": skew,
            "min": s["min"],
            "p25": s["25%"],
            "p50": s["50%"],
            "p75": s["75%"],
            "max": s["max"],
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "curve_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Saved %s", stats_path)

    # ----------------------------------------
    # Export clean per-well DataFrames to parquet
    # ----------------------------------------
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    for wid, df in wells.items():
        parquet_path = processed_dir / f"{wid}.parquet"
        df.to_parquet(parquet_path, index=False)
    logger.info("Exported %d well DataFrames to %s/", len(wells), processed_dir)

    logger.info("EDA complete. Outputs in %s", out_dir)
    print("\n=== EDA Summary ===")
    print(f"  Wells loaded  : {len(wells)}")
    print(f"  Total rows    : {len(all_df):,}")
    print(f"  DEN–NPHI fit  : DEN = {slope:.4f}·NPHI + {intercept:.4f}  (R²={r_value**2:.3f})")
    print(f"  Parquets saved: {processed_dir}/")
    print(f"  Outputs saved : {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on Kraft Prusa LAS dataset")
    parser.add_argument("--field-dir", type=Path, default=Path("data/raw"),
                        help="Directory with LAS files (default: data/raw)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eda"),
                        help="Output directory (default: outputs/eda)")
    args = parser.parse_args()
    main(args.field_dir, args.out_dir)
