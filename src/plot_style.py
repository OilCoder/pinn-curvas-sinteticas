"""
Project-wide matplotlib style for the Kraft Prusa PINN portfolio.

Import and call apply_style() once at the top of any script that produces
figures. Color constants are available for use in individual plot calls.

Usage:
    from src.plot_style import apply_style, BLUE, RED, GREEN
    apply_style()
"""

import matplotlib.pyplot as plt

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE    = "#2B6CB0"   # primary data — histograms, scatter, box fills
RED     = "#C53030"   # regression / baseline MLP predictions
GREEN   = "#276749"   # PINN predictions (Phase 3)
ORANGE  = "#C05621"   # anomaly markers, highlights
GRAY    = "#718096"   # secondary lines, confidence bands
LIGHT   = "#EBF4FF"   # shaded regions, fill areas

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_TITLE  = 13
FS_LABEL  = 10
FS_TICK   = 9
FS_SMALL  = 8
FS_LEGEND = 9

# ── Figure defaults ───────────────────────────────────────────────────────────
DPI = 150


def apply_style() -> None:
    """Apply project-wide rcParams. Call once before any plt/fig/ax calls."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor":      "white",
        "figure.dpi":            DPI,
        "savefig.dpi":           DPI,
        "savefig.facecolor":     "white",
        # Axes background and frame
        "axes.facecolor":        "white",
        "axes.edgecolor":        "#CBD5E0",
        "axes.linewidth":        0.8,
        "axes.labelcolor":       "#2D3748",
        "axes.labelsize":        FS_LABEL,
        "axes.titlesize":        FS_TITLE,
        "axes.titleweight":      "semibold",
        "axes.titlecolor":       "#1A202C",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        # Subtle grid — helps read values without visual noise
        "axes.grid":             True,
        "grid.color":            "#E2E8F0",
        "grid.linewidth":        0.5,
        "grid.alpha":            1.0,
        # Ticks
        "xtick.labelsize":       FS_TICK,
        "ytick.labelsize":       FS_TICK,
        "xtick.color":           "#4A5568",
        "ytick.color":           "#4A5568",
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        # Legend
        "legend.fontsize":       FS_LEGEND,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      "#CBD5E0",
        # Lines
        "lines.linewidth":       1.5,
        # Font
        "font.family":           "sans-serif",
        "text.color":            "#2D3748",
    })
