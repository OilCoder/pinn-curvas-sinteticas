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
from src.plot_style import BLUE, GRAY, ORANGE, RED, FS_SMALL, apply_style
from src.preprocessing import FEATURE_BOUNDS, TARGET_BOUNDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["GR", "RT", "RILM", "NPHI", "SP"]
TARGET_COL = "DEN"


# ----------------------------------------
# Helpers
# ----------------------------------------

def _save(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    """Save figure at portfolio-grade resolution and close it."""
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
    """Plot histograms for each canonical curve (post-clip bounds applied).

    Applies FEATURE_BOUNDS and TARGET_BOUNDS before plotting so that the
    distributions reflect exactly what enters the model — RT/RILM are
    unreadable in raw scale due to tail outliers.

    Args:
        all_df: Concatenated DataFrame from all wells.
        out_dir: Directory to save figures.
    """
    _UNITS = {"GR": "GAPI", "RT": "Ohm·m", "RILM": "Ohm·m",
               "NPHI": "v/v", "SP": "mV", "DEN": "g/cc"}

    cols = CANONICAL_CURVES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        raw = all_df[col].dropna()

        # Apply clip/filter matching the preprocessing pipeline
        if col in FEATURE_BOUNDS:
            lo, hi = FEATURE_BOUNDS[col]
            data = raw.clip(lower=lo, upper=hi)
        else:
            lo, hi = TARGET_BOUNDS
            data = raw[(raw >= lo) & (raw <= hi)]

        ax.hist(data, bins=60, color=BLUE, alpha=0.80, edgecolor="none", linewidth=0)

        # Median and mean reference lines
        med = float(data.median())
        mn  = float(data.mean())
        skew = float(data.skew())
        ax.axvline(med, color=RED,  linewidth=1.2, linestyle="-",  label=f"mediana {med:.3g}")
        ax.axvline(mn,  color=GRAY, linewidth=1.0, linestyle="--", label=f"media   {mn:.3g}")

        # Skewness annotation — drives the normalization choice (see 02_preprocessing.md)
        ax.text(
            0.97, 0.97, f"asimetría = {skew:+.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FS_SMALL - 1, color="#4A5568",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#CBD5E0", "alpha": 0.9},
        )

        ax.set_title(col)
        ax.set_xlabel(f"{col} ({_UNITS[col]})", fontsize=FS_SMALL)
        ax.set_ylabel("Frecuencia", fontsize=FS_SMALL)
        ax.legend(fontsize=FS_SMALL - 1, handlelength=1.2, loc="upper left")

    if len(cols) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle(
        "Distribuciones de las curvas canónicas — post-recorte (datos listos para entrenar)",
        fontsize=13, fontweight="semibold",
    )
    fig.tight_layout()
    _save(fig, out_dir / "distributions_raw.png")


def plot_log_rt_comparison(all_df: pd.DataFrame, out_dir: Path) -> None:
    """Compare raw vs log10 RT and RILM in a 2×2 grid with separate X axes.

    Each panel has its own axis so the raw (skewed) and log₁₀ (symmetric)
    distributions are both readable — twin-axis shares an X scale, making
    one of the two histograms always invisible.

    Args:
        all_df: Concatenated DataFrame from all wells.
        out_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for row, col in enumerate(("RT", "RILM")):
        lo, hi = FEATURE_BOUNDS[col]
        raw      = all_df[col].dropna().clip(lower=lo, upper=hi)
        log_vals = np.log10(raw.clip(lower=1e-4))

        skew_raw = float(raw.skew())
        skew_log = float(log_vals.skew())

        # — Raw distribution
        ax_raw = axes[row, 0]
        ax_raw.hist(raw, bins=60, color=ORANGE, alpha=0.85, edgecolor="none")
        ax_raw.set_title(f"{col} — escala lineal  (asimetría = {skew_raw:+.2f})")
        ax_raw.set_xlabel(f"{col} (Ohm·m)", fontsize=FS_SMALL)
        ax_raw.set_ylabel("Frecuencia", fontsize=FS_SMALL)

        # — Log₁₀ distribution
        ax_log = axes[row, 1]
        ax_log.hist(log_vals, bins=60, color=BLUE, alpha=0.85, edgecolor="none")
        ax_log.set_title(f"{col} — escala log₁₀  (asimetría = {skew_log:+.2f})")
        ax_log.set_xlabel(f"log₁₀({col})", fontsize=FS_SMALL)
        ax_log.set_ylabel("Frecuencia", fontsize=FS_SMALL)

    fig.suptitle(
        "Resistividad: escala lineal vs log₁₀ — justificación de la transformación",
        fontsize=13, fontweight="semibold",
    )
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

    # Filter to physical bounds before plotting so Dolecheck_1 (NPHI 0.8–5.7)
    # does not collapse the entire scatter into a thin vertical band.
    nphi_hi = FEATURE_BOUNDS["NPHI"][1]
    den_lo, den_hi = TARGET_BOUNDS
    mask = (nphi >= 0) & (nphi <= nphi_hi) & (den >= den_lo) & (den <= den_hi)
    nphi_plot, den_plot = nphi[mask], den[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(nphi_plot, den_plot, alpha=0.06, s=3, color=BLUE, rasterized=True)

    x_line = np.linspace(nphi_plot.min(), nphi_plot.max(), 200)
    ax.plot(
        x_line, slope * x_line + intercept,
        color=RED, linewidth=2,
        label=f"DEN = {slope:.3f}·NPHI + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}",
    )

    # Note: this raw-space global fit has low R² because it mixes wells with
    # different absolute offsets. The PINN constraint is calibrated per-well in
    # normalized space (R²=0.338) — see 01_eda.md §5.
    ax.text(
        0.03, 0.05,
        "Ajuste global en espacio crudo (mezcla pozos).\n"
        "La restricción del PINN se calibra normalizada (R²=0.338).",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=FS_SMALL - 1, color="#4A5568", style="italic",
    )

    ax.set_xlabel("NPHI (v/v)")
    ax.set_ylabel("DEN (g/cc)")
    ax.set_title("DEN vs NPHI — todos los pozos, post-recorte (ajuste global)")
    ax.legend(fontsize=FS_SMALL + 1, loc="upper right")
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
        nphi_hi = FEATURE_BOUNDS["NPHI"][1]
        den_lo, den_hi = TARGET_BOUNDS
        data = df[["NPHI", "DEN"]].dropna()
        data = data[(data["NPHI"] <= nphi_hi) & (data["DEN"] >= den_lo) & (data["DEN"] <= den_hi)]
        ax.scatter(data["NPHI"], data["DEN"], alpha=0.25, s=2, color=BLUE, rasterized=True)
        if len(data) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(data["NPHI"], data["DEN"])
            x_line = np.linspace(data["NPHI"].min(), data["NPHI"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=RED, linewidth=1)
        ax.set_title(wid, fontsize=FS_SMALL)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "DEN vs NPHI por pozo — la pendiente negativa es consistente intra-pozo",
        fontsize=12, fontweight="semibold",
    )
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

    _UNITS = {"GR": "GAPI", "RT": "Ohm·m", "RILM": "Ohm·m",
               "NPHI": "v/v", "SP": "mV", "DEN": "g/cc"}

    for ax, col in zip(axes, cols):
        # Apply clip/filter matching preprocessing so outliers don't collapse scale
        if col in FEATURE_BOUNDS:
            lo, hi = FEATURE_BOUNDS[col]
            data_per_well = [
                np.clip(wells[wid][col].dropna().values, lo, hi)
                for wid in well_ids
            ]
        else:
            lo, hi = TARGET_BOUNDS
            data_per_well = [
                wells[wid][col].dropna().values
                for wid in well_ids
            ]
            data_per_well = [d[(d >= lo) & (d <= hi)] for d in data_per_well]

        ax.boxplot(
            data_per_well,
            labels=well_ids,
            patch_artist=True,
            boxprops={"facecolor": BLUE, "alpha": 0.35, "linewidth": 0.7},
            medianprops={"color": RED, "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.25, "markerfacecolor": GRAY},
            whiskerprops={"linewidth": 0.7, "color": GRAY},
            capprops={"linewidth": 0.7, "color": GRAY},
        )
        ax.set_ylabel(f"{col}\n({_UNITS[col]})", fontsize=FS_SMALL)
        ax.tick_params(axis="x", labelsize=6, rotation=55)

    fig.suptitle(
        "Distribución por pozo — post-recorte, unidades físicas\n"
        "La variación horizontal entre pozos justifica la normalización per-well",
        fontsize=12, fontweight="semibold",
    )
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
            color = RED if col == TARGET_COL else BLUE
            ax.plot(df[col], df["DEPTH"], linewidth=0.5, color=color)
            ax.set_xlabel(col, fontsize=FS_SMALL)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)

        axes[0].set_ylabel("Profundidad (ft)", fontsize=FS_SMALL)
        fig.suptitle(f"Perfil de profundidad — Pozo: {wid}", fontsize=11, fontweight="semibold")
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
    apply_style()
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
