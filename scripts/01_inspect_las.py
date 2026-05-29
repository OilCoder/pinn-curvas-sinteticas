"""
Inspect LAS files in a field directory and produce a per-well inventory CSV.

Reports curve mnemonics, depth units, depth range, total data rows,
and null percentage per curve for each well.
"""

import os
import re
import csv
import glob

NULL_VALUES = {-9999.0, -999.25, -9999.25}
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Kraft Prusa")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "las_inventory.csv")


def parse_las(path: str) -> dict:
    """Parse a single LAS file and return a summary dict."""
    result = {
        "file": os.path.basename(path),
        "curves": [],
        "depth_unit": "",
        "depth_min": None,
        "depth_max": None,
        "total_rows": 0,
        "null_pct": {},
        "error": "",
    }

    try:
        with open(path, "r", errors="ignore") as fh:
            lines = fh.readlines()
    except Exception as e:
        result["error"] = str(e)
        return result

    # ----------------------------------------
    # Step 1 — Parse header sections
    # ----------------------------------------
    section = ""
    curves = []
    depth_unit = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("~"):
            section = stripped[1].upper()
            continue

        if section == "W":
            # Well section — look for STRT/STOP/STEP
            pass

        if section == "C":
            # Curve section — extract mnemonic and unit
            match = re.match(r"^([A-Z0-9_]+)\s*\.\s*(\S*)", stripped, re.IGNORECASE)
            if match:
                mnemonic = match.group(1).upper()
                unit = match.group(2)
                if mnemonic not in ("DEPT", "DEPTH", "MD", "TVD"):
                    curves.append(mnemonic)
                else:
                    depth_unit = unit
            continue

        if section in ("A", "a"):
            break

    result["curves"] = curves
    result["depth_unit"] = depth_unit

    # ----------------------------------------
    # Step 2 — Parse data section
    # ----------------------------------------
    in_data = False
    rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("~A") or stripped.startswith("~a"):
            in_data = True
            continue
        if not in_data:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values = [float(v) for v in stripped.split()]
            rows.append(values)
        except ValueError:
            continue

    if not rows:
        return result

    result["total_rows"] = len(rows)

    # Substep 2.1 — depth range from first column
    depths = [r[0] for r in rows]
    result["depth_min"] = round(min(depths), 2)
    result["depth_max"] = round(max(depths), 2)

    # Substep 2.2 — null % per curve (columns 1..N)
    for i, curve in enumerate(curves):
        col_idx = i + 1
        if col_idx >= len(rows[0]):
            result["null_pct"][curve] = None
            continue
        col_vals = [r[col_idx] for r in rows]
        null_count = sum(1 for v in col_vals if v in NULL_VALUES)
        result["null_pct"][curve] = round(100.0 * null_count / len(col_vals), 1)

    return result


# ----------------------------------------
# Step 3 — Classify curves into canonical families
# ----------------------------------------
GR_VARIANTS    = {"GR", "SGR", "CGR", "GRD", "GRC"}
RT_VARIANTS    = {"RT", "ILD", "ILM", "LLD", "LLS", "MSFL", "RLLD", "RLLS",
                  "RD", "RS", "RILD", "RILM", "AT90", "AT10", "AT20", "AT60",
                  "M2RX", "M2R6", "HDRS", "HMRS"}
NPHI_VARIANTS  = {"NPHI", "PHIN", "CNCF", "NEUT", "TNPH", "NPOR", "CNPOR",
                  "CNLS", "CNSS", "CNDL"}
DEN_VARIANTS   = {"RHOB", "DEN", "RHOZ", "DPHI", "ZDEN", "RHOZ", "ROML", "RHO"}


def classify(curves: list[str]) -> dict[str, str | None]:
    """Return the first matching mnemonic for each canonical curve family."""
    found: dict[str, str | None] = {"GR": None, "RT": None, "NPHI": None, "DEN": None}
    for c in curves:
        if found["GR"] is None and c in GR_VARIANTS:
            found["GR"] = c
        if found["RT"] is None and c in RT_VARIANTS:
            found["RT"] = c
        if found["NPHI"] is None and c in NPHI_VARIANTS:
            found["NPHI"] = c
        if found["DEN"] is None and c in DEN_VARIANTS:
            found["DEN"] = c
    return found


def main() -> None:
    """Run inventory over all LAS files in INPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    las_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.las")) +
        glob.glob(os.path.join(INPUT_DIR, "*.LAS"))
    )

    if not las_files:
        print(f"No LAS files found in {INPUT_DIR}")
        return

    print(f"Found {len(las_files)} LAS files. Parsing...")

    records = []
    for path in las_files:
        info = parse_las(path)
        canonical = classify(info["curves"])
        record = {
            "file":        info["file"],
            "depth_unit":  info["depth_unit"],
            "depth_min":   info["depth_min"],
            "depth_max":   info["depth_max"],
            "total_rows":  info["total_rows"],
            "all_curves":  "|".join(info["curves"]),
            "GR":          canonical["GR"] or "",
            "RT":          canonical["RT"] or "",
            "NPHI":        canonical["NPHI"] or "",
            "DEN":         canonical["DEN"] or "",
            "has_all_4":   int(all(canonical[k] for k in ("GR", "RT", "NPHI", "DEN"))),
            "null_pct_GR":   info["null_pct"].get(canonical["GR"] or "", ""),
            "null_pct_RT":   info["null_pct"].get(canonical["RT"] or "", ""),
            "null_pct_NPHI": info["null_pct"].get(canonical["NPHI"] or "", ""),
            "null_pct_DEN":  info["null_pct"].get(canonical["DEN"] or "", ""),
            "error":       info["error"],
        }
        records.append(record)

    # ----------------------------------------
    # Step 4 — Write CSV
    # ----------------------------------------
    fieldnames = list(records[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # ----------------------------------------
    # Step 5 — Print summary
    # ----------------------------------------
    total        = len(records)
    has_gr       = sum(1 for r in records if r["GR"])
    has_rt       = sum(1 for r in records if r["RT"])
    has_nphi     = sum(1 for r in records if r["NPHI"])
    has_den      = sum(1 for r in records if r["DEN"])
    has_all      = sum(1 for r in records if r["has_all_4"])
    errors       = sum(1 for r in records if r["error"])

    rows_vals    = [r["total_rows"] for r in records if r["total_rows"]]
    avg_rows     = int(sum(rows_vals) / len(rows_vals)) if rows_vals else 0

    print(f"\n{'='*50}")
    print(f"Campo: Kraft Prusa — {total} pozos analizados")
    print(f"{'='*50}")
    print(f"  GR  (o variante):        {has_gr:3d} / {total}")
    print(f"  RT  (o variante):        {has_rt:3d} / {total}")
    print(f"  NPHI (o variante):       {has_nphi:3d} / {total}")
    print(f"  DEN  (o variante):       {has_den:3d} / {total}")
    print(f"  {'─'*35}")
    print(f"  Todas 4 curvas:          {has_all:3d} / {total}  ← clave para LOWO")
    print(f"  Errores de parseo:       {errors:3d}")
    print(f"  Promedio filas/pozo:     {avg_rows:,}")
    print(f"\nCSV guardado en: {OUTPUT_CSV}")

    # Substep 5.1 — show RT variants found
    rt_found = {}
    for r in records:
        if r["RT"]:
            rt_found[r["RT"]] = rt_found.get(r["RT"], 0) + 1
    if rt_found:
        print(f"\nVariantes de RT encontradas: {dict(sorted(rt_found.items(), key=lambda x: -x[1]))}")

    # Substep 5.2 — show DEN variants found
    den_found = {}
    for r in records:
        if r["DEN"]:
            den_found[r["DEN"]] = den_found.get(r["DEN"], 0) + 1
    if den_found:
        print(f"Variantes de DEN encontradas: {dict(sorted(den_found.items(), key=lambda x: -x[1]))}")

    # Substep 5.3 — show wells with all 4 and their depth ranges
    viable = [r for r in records if r["has_all_4"] and not r["error"]]
    if viable:
        depths = [r["depth_max"] - r["depth_min"] for r in viable
                  if r["depth_max"] and r["depth_min"]]
        if depths:
            print("\nPozos viables (4 curvas) — rango de columna:")
            print(f"  Min: {min(depths):.0f}  Max: {max(depths):.0f}  Avg: {sum(depths)/len(depths):.0f} [{records[0]['depth_unit']}]")


if __name__ == "__main__":
    main()
