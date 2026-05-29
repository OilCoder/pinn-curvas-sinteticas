"""
Load LAS files from a field directory and return clean DataFrames per well.

Canonical curve mapping applied at load time:
  GR   → GR    (Gamma Ray)
  RILD → RT    (Deep Induction Resistivity)
  RILM → RILM  (Medium Induction Resistivity)
  CNLS/CNPOR/CNSS/CNDL/NPOR → NPHI  (Compensated Neutron)
  SP   → SP    (Spontaneous Potential)
  RHOB → DEN   (Bulk Density — prediction target)

Called by: scripts/run_eda.py, src/preprocessing.py
"""

import logging
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_CURVES = ["GR", "RT", "RILM", "NPHI", "SP", "DEN"]

_MNEMONIC_MAP: dict[str, str] = {
    # GR
    "GR": "GR",
    # RT (deep resistivity)
    "RILD": "RT",
    # RILM (medium resistivity)
    "RILM": "RILM",
    # NPHI variants
    "CNLS": "NPHI",
    "CNPOR": "NPHI",
    "CNSS": "NPHI",
    "CNDL": "NPHI",
    "NPOR": "NPHI",
    "NEU": "NPHI",
    "NEUT": "NPHI",
    "PHIN": "NPHI",
    "CNCF": "NPHI",
    "TNPH": "NPHI",
    # SP
    "SP": "SP",
    # DEN (target)
    "RHOB": "DEN",
}

_NULL_VALUES = {-9999.0, -999.25, -9999.25, -999.0}


def load_well(path: Path) -> pd.DataFrame | None:
    """Load a single LAS file and return a DataFrame with canonical column names.

    Args:
        path: Path to the LAS file.

    Returns:
        DataFrame with columns [DEPTH, GR, RT, RILM, NPHI, SP, DEN] or None
        if the file cannot be loaded or is missing required curves.
    """
    try:
        las = lasio.read(str(path), ignore_header_errors=True)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path.name, exc)
        return None

    # ----------------------------------------
    # Step 1 — Build canonical DataFrame
    # ----------------------------------------
    df = las.df().reset_index()
    df.columns = [c.upper() for c in df.columns]

    # Rename depth index (DEPT, DEPTH, MD, etc.) → DEPTH
    for depth_alias in ("DEPT", "DEPTH", "MD", "TVD"):
        if depth_alias in df.columns:
            df = df.rename(columns={depth_alias: "DEPTH"})
            break

    # Substep 1.1 — Map mnemonics to canonical names (first match wins)
    canonical_present: dict[str, str] = {}
    for col in df.columns:
        canon = _MNEMONIC_MAP.get(col)
        if canon and canon not in canonical_present:
            canonical_present[canon] = col

    if not all(c in canonical_present for c in CANONICAL_CURVES):
        missing = [c for c in CANONICAL_CURVES if c not in canonical_present]
        logger.debug("Skipping %s — missing curves: %s", path.name, missing)
        return None

    # ----------------------------------------
    # Step 2 — Build output DataFrame
    # ----------------------------------------
    cols = {"DEPTH": "DEPTH"} | {canon: raw for canon, raw in canonical_present.items()}
    out = df[[v for v in cols.values() if v in df.columns]].copy()
    out = out.rename(columns={v: k for k, v in cols.items()})
    out = out[["DEPTH"] + CANONICAL_CURVES]

    # ----------------------------------------
    # Step 3 — Fix data quality issues
    # ----------------------------------------

    # Substep 3.1 — Replace known LAS null sentinel values
    for null_val in _NULL_VALUES:
        out.replace(null_val, np.nan, inplace=True)

    # Substep 3.2 — Replace large-value resistivity sentinels (some tools encode
    # null as 1e9, 1e30, or 9999*10^N; values above 1e6 Ohm·m are non-physical)
    for res_col in ("RT", "RILM"):
        if res_col in out.columns:
            out.loc[out[res_col].abs() > 1e6, res_col] = np.nan

    # Substep 3.3 — Convert NPHI from percentage to v/v fraction if needed.
    # Kraft Prusa (and many older KGS files) store CNLS in % (0–100).
    # Heuristic: divide by 100 if the LAS unit field contains "%" OR if the
    # non-null median exceeds 1.5 (physically impossible in v/v space).
    if "NPHI" in out.columns:
        nphi_raw_col = canonical_present.get("NPHI")
        nphi_unit = ""
        if nphi_raw_col:
            try:
                nphi_unit = las.curves[nphi_raw_col].unit.strip().upper()
            except (KeyError, AttributeError):
                pass
        nphi_nonnull = out["NPHI"].dropna()
        if "%" in nphi_unit or (len(nphi_nonnull) > 0 and float(nphi_nonnull.median()) > 1.5):
            out["NPHI"] = out["NPHI"] / 100.0
            logger.debug(
                "Converted NPHI from %% to v/v for %s (unit=%r, median before=%.1f)",
                path.name,
                nphi_unit,
                float(nphi_nonnull.median()),
            )

    # ----------------------------------------
    # Step 4 — Drop NaN rows and validate minimum row count
    # ----------------------------------------
    out = out.dropna(subset=CANONICAL_CURVES).reset_index(drop=True)

    if len(out) < 100:
        logger.debug("Skipping %s — too few rows after cleaning: %d", path.name, len(out))
        return None

    return out


def load_field(
    field_dir: Path,
    exclude_files: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all valid LAS files from a field directory.

    Args:
        field_dir: Directory containing LAS files.
        exclude_files: List of filenames (without extension) to skip.

    Returns:
        Dict mapping well_id (stem of LAS filename) to clean DataFrame.
    """
    exclude = {f.lower() for f in (exclude_files or [])}
    las_files = sorted(field_dir.glob("*.las")) + sorted(field_dir.glob("*.LAS"))

    wells: dict[str, pd.DataFrame] = {}
    for path in las_files:
        if path.stem.lower() in exclude:
            logger.info("Skipping excluded file: %s", path.name)
            continue
        df = load_well(path)
        if df is not None:
            wells[path.stem] = df
            logger.info("Loaded %s — %d rows", path.stem, len(df))

    logger.info("Loaded %d / %d wells from %s", len(wells), len(las_files), field_dir.name)
    return wells
