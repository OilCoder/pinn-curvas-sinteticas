"""Tests for src/data_loader.py."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import CANONICAL_CURVES, load_well, load_field


# ----------------------------------------
# Fixtures
# ----------------------------------------

LAS_TEMPLATE = textwrap.dedent("""\
    ~VERSION ---
    VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
    ~WELL ---
    WELL.   TEST_WELL_1
    ~CURVE ---
    DEPT .FT
    GR   .GAPI
    RILD .OHMM
    RILM .OHMM
    CNLS .V/V
    SP   .MV
    RHOB .G/C3
    ~A
    {rows}
""")


def _make_las_file(tmp_path: Path, filename: str = "well_1.las") -> Path:
    """Create a minimal valid LAS file for testing."""
    rows = "\n".join(
        f"{d:.1f}  {40 + i * 0.1:.2f}  {5.0:.3f}  {4.0:.3f}  {0.25:.4f}  {-30 + i * 0.1:.2f}  {2.35:.4f}"
        for i, d in enumerate(range(200, 400))
    )
    las_content = LAS_TEMPLATE.format(rows=rows)
    path = tmp_path / filename
    path.write_text(las_content)
    return path


def _make_las_nphi_percent(tmp_path: Path, unit_str: str = "%") -> Path:
    """Create a LAS file with CNLS declared in percentage units (values 20–30)."""
    rows = "\n".join(
        f"{d:.1f}  {40.0:.2f}  {5.0:.3f}  {4.0:.3f}  {25.0:.4f}  {-30.0:.2f}  {2.35:.4f}"
        for d in range(200, 400)
    )
    content = textwrap.dedent(f"""\
        ~VERSION ---
        VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
        ~WELL ---
        WELL.   TEST_NPHI_PCT
        ~CURVE ---
        DEPT .FT
        GR   .GAPI
        RILD .OHMM
        RILM .OHMM
        CNLS .{unit_str}
        SP   .MV
        RHOB .G/C3
        ~A
        {rows}
    """)
    path = tmp_path / "nphi_pct.las"
    path.write_text(content)
    return path


def _make_las_large_rt_sentinel(tmp_path: Path) -> Path:
    """Create a LAS with one RT row equal to 1e9 (a large-value sentinel)."""
    rows = [
        f"{d:.1f}  {40.0:.2f}  {5.0:.3f}  {4.0:.3f}  {0.25:.4f}  {-30.0:.2f}  {2.35:.4f}"
        for d in range(200, 400)
    ]
    # Inject a large sentinel in row 0 (RT column, index 2)
    parts = rows[0].split()
    parts[2] = "1000000000.0"
    rows[0] = "  ".join(parts)
    content = LAS_TEMPLATE.format(rows="\n".join(rows))
    path = tmp_path / "large_sentinel.las"
    path.write_text(content)
    return path


def _make_las_missing_curve(tmp_path: Path) -> Path:
    """Create a LAS file missing the DEN (RHOB) curve."""
    rows = "\n".join(
        f"{d:.1f}  {40.0:.2f}  {5.0:.3f}  {4.0:.3f}  {0.25:.4f}  {-30.0:.2f}"
        for d in range(200, 400)
    )
    content = textwrap.dedent("""\
        ~VERSION ---
        VERS.   2.0
        ~WELL ---
        WELL.   TEST_NO_DEN
        ~CURVE ---
        DEPT .FT
        GR   .GAPI
        RILD .OHMM
        RILM .OHMM
        CNLS .V/V
        SP   .MV
        ~A
        {rows}
    """).format(rows=rows)
    path = tmp_path / "no_den.las"
    path.write_text(content)
    return path


# ----------------------------------------
# Tests — load_well
# ----------------------------------------

def test_load_well_returns_dataframe(tmp_path):
    path = _make_las_file(tmp_path)
    df = load_well(path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)


def test_load_well_has_canonical_columns(tmp_path):
    path = _make_las_file(tmp_path)
    df = load_well(path)
    assert df is not None
    for col in CANONICAL_CURVES:
        assert col in df.columns, f"Missing column: {col}"
    assert "DEPTH" in df.columns


def test_load_well_no_nulls(tmp_path):
    path = _make_las_file(tmp_path)
    df = load_well(path)
    assert df is not None
    assert df[CANONICAL_CURVES].isna().sum().sum() == 0


def test_load_well_missing_curve_returns_none(tmp_path):
    path = _make_las_missing_curve(tmp_path)
    result = load_well(path)
    assert result is None


def test_load_well_invalid_file_returns_none(tmp_path):
    path = tmp_path / "bad.las"
    path.write_text("this is not a valid LAS file @@@@")
    result = load_well(path)
    assert result is None


# ----------------------------------------
# Tests — load_field
# ----------------------------------------

def test_load_field_returns_dict(tmp_path):
    _make_las_file(tmp_path, "well_a.las")
    _make_las_file(tmp_path, "well_b.las")
    wells = load_field(tmp_path)
    assert isinstance(wells, dict)
    assert len(wells) == 2
    assert "well_a" in wells
    assert "well_b" in wells


def test_load_field_excludes_files(tmp_path):
    _make_las_file(tmp_path, "well_a.las")
    _make_las_file(tmp_path, "well_b.las")
    wells = load_field(tmp_path, exclude_files=["well_a"])
    assert "well_a" not in wells
    assert "well_b" in wells


def test_load_field_skips_incomplete_wells(tmp_path):
    _make_las_file(tmp_path, "good.las")
    _make_las_missing_curve(tmp_path)
    wells = load_field(tmp_path)
    assert "good" in wells
    assert "no_den" not in wells


# ----------------------------------------
# Tests — NPHI unit conversion
# ----------------------------------------

def test_nphi_percent_unit_converted_to_fraction(tmp_path):
    """CNLS declared as '%' with values ~25 should yield NPHI ~0.25."""
    path = _make_las_nphi_percent(tmp_path, unit_str="%")
    df = load_well(path)
    assert df is not None
    assert df["NPHI"].max() <= 1.0, "NPHI must be in v/v after % conversion"
    assert df["NPHI"].mean() == pytest.approx(0.25, abs=1e-3)


def test_nphi_heuristic_converts_when_unit_unknown(tmp_path):
    """CNLS with no unit declared but values ~25 should still be converted."""
    path = _make_las_nphi_percent(tmp_path, unit_str="")
    df = load_well(path)
    assert df is not None
    assert df["NPHI"].max() <= 1.0, "NPHI must be in v/v via heuristic conversion"


def test_nphi_vv_unit_not_converted(tmp_path):
    """CNLS declared as 'V/V' with values ~0.25 must not be divided."""
    path = _make_las_file(tmp_path)  # uses CNLS .V/V with value 0.25
    df = load_well(path)
    assert df is not None
    assert df["NPHI"].mean() == pytest.approx(0.25, abs=0.05)


# ----------------------------------------
# Tests — large-value sentinel handling
# ----------------------------------------

def test_large_rt_sentinel_row_dropped(tmp_path):
    """A row with RT = 1e9 (large sentinel) should be dropped after NaN replacement."""
    path = _make_las_large_rt_sentinel(tmp_path)
    df = load_well(path)
    assert df is not None
    assert df["RT"].max() < 1e6, "Large RT sentinel must be removed"
    assert len(df) == 199, "Exactly one row (the sentinel row) should be dropped"
