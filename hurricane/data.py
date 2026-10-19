"""Load and clean NOAA's table of continental US hurricane landfalls.

The table is NOAA Hurricane Research Division's "Continental United States Hurricane
Impacts/Landfalls" (https://www.aoml.noaa.gov/hrd/hurdat/UShurrs_detailed.html), rows from
1901 to 1970, copied into ``Hurricane_dataset - Sheet1.csv``. NOAA's definitions:

- RMW: the radius of maximum winds, in nautical miles.
- OCI: the pressure of the outer closed isobar, in mb (the outer limit of the circulation).
- Size: "the average radius of the OCI", in nautical miles. A radius, not a diameter.
- A central pressure in parentheses is an estimate from winds, not a measurement. The
  spreadsheet stored those as negative numbers.
- A mark after the date is a footnote: ``*`` no US landfall but hurricane winds over land,
  ``#`` landfall in Mexico, ``%`` the strongest winds came before landfall.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

#: The six storm measures the models use: everything the app asks for, and no dates.
STORM_MEASURES = ["Central_Pressure", "Max_Winds_kt", "RMW_nm", "OCI_mb", "Lat", "Long"]

#: Saffir-Simpson categories by 1-minute winds in knots (NOAA's bands).
CATEGORY_WINDS = {1: (64, 82), 2: (83, 95), 3: (96, 112), 4: (113, 136), 5: (137, 180)}

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "Hurricane_dataset - Sheet1.csv"


def parse_coordinate(text: str) -> float:
    """``36.0N`` -> 36.0, ``75.8W`` -> -75.8."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([NSEW])\s*", str(text), re.IGNORECASE)
    if not m:
        return np.nan
    value = float(m.group(1))
    return -value if m.group(2).upper() in "SW" else value


def parse_date(text: str) -> tuple[pd.Timestamp, str]:
    """``8/27/1909#`` -> (1909-08-27, "#"). The mark is kept, not allowed to lose the date."""
    s = str(text).strip()
    mark = "".join(ch for ch in s if ch in "*#%$&")
    digits = re.sub(r"[^0-9/]", "", s)
    return pd.to_datetime(digits, format="%m/%d/%Y"), mark


def parse_number(text) -> float:
    """``---`` and blanks are missing; everything else is a number."""
    s = str(text).strip().replace(",", "")
    if not s or set(s) <= {"-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _storm_name(text) -> str | None:
    s = str(text).strip().strip('"').strip()
    return None if not s or set(s) <= {"-"} else s


def assign_storms(dates: pd.Series, sizes: pd.Series, names: pd.Series, gap_days: int = 4) -> np.ndarray:
    """Give consecutive landfalls of one storm the same id.

    A storm that crosses several coasts has one row per landfall and the same Size on each.
    Two rows are the same storm when they are at most ``gap_days`` apart and share the size or
    the storm's name. (The 1933 Jupiter and Brownsville storms, a day apart, stay separate.)
    """
    ids = np.zeros(len(dates), dtype=int)
    for i in range(1, len(dates)):
        same_size = sizes.iloc[i] == sizes.iloc[i - 1]
        same_name = names.iloc[i] is not None and names.iloc[i] == names.iloc[i - 1]
        close = (dates.iloc[i] - dates.iloc[i - 1]).days <= gap_days
        ids[i] = ids[i - 1] if close and (same_size or same_name) else ids[i - 1] + 1
    return ids


def load_landfalls(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """One row per landfall, cleaned, with a ``storm`` id for keeping a storm's rows together."""
    raw = pd.read_csv(path, dtype=str)
    raw = raw[[c for c in raw.columns if not c.startswith("Unnamed")]]
    out = pd.DataFrame(index=raw.index)
    parsed = raw["Date"].map(parse_date)
    out["Date"] = [d for d, _ in parsed]
    out["Mark"] = [m for _, m in parsed]
    out["Hour"] = raw["Time"].str.extract(r"(\d{2})\d{2}Z", expand=False).astype(float)
    out["Lat"] = raw["Lat"].map(parse_coordinate)
    out["Long"] = raw["Long"].map(parse_coordinate)
    out["Max_Winds_kt"] = raw["Max_Winds_kt"].map(parse_number)
    out["SSHWS"] = raw["SSHWS"].map(parse_number)
    out["RMW_nm"] = raw["RMW_nm"].map(parse_number)
    pressure = raw["Central_Pressure"].map(parse_number)
    out["Central_Pressure"] = pressure.abs()
    out["Pressure_Estimated"] = pressure < 0
    oci = raw["OCI_mb"].map(parse_number)
    # NOAA's own table gives 125 mb for 15 Sep 1945 19:30Z (the same storm's next row: 1011).
    # No outer isobar is that low, so it is treated as missing.
    out["OCI_mb"] = oci.where(oci >= 900)
    out["Size_nm"] = raw["Size_nm"].map(parse_number)
    out["States_Affected"] = raw["States_Affected"].fillna("").str.strip()
    out["Storm_Name"] = raw["Strom_Name"].map(_storm_name)
    out["storm"] = assign_storms(out["Date"], out["Size_nm"], out["Storm_Name"])
    return out
