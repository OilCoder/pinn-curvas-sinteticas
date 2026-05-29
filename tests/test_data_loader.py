"""Tests for src/data_loader.py."""

import textwrap
from pathlib import Path

import pandas as pd

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
