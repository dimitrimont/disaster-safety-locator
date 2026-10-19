"""What a predicted storm covers: rings, distances and the exposure of address points.

No Streamlit here, so it can be tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FEATURES

NM_TO_M = 1852.0
EARTH_RADIUS_M = 6_371_000.0
WILMINGTON = (34.2257, -77.9447)

#: Outer closed isobar pressures in the landfall table run from 1001 to 1018 mb.
OCI_RANGE_MB = (1000, 1020)


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)) / NM_TO_M


def storm_row(lat, lon, max_winds_kt, central_pressure_mb, rmw_nm, oci_mb) -> pd.DataFrame:
    """One storm as the model's input: exactly its features, in order, with no padding."""
    values = {
        "Central_Pressure": central_pressure_mb,
        "Max_Winds_kt": max_winds_kt,
        "RMW_nm": rmw_nm,
        "OCI_mb": oci_mb,
        "Lat": lat,
        "Long": lon,
    }
    missing = set(FEATURES) - set(values)
    if missing:
        raise KeyError(f"model needs {sorted(missing)}")
    return pd.DataFrame([[float(values[c]) for c in FEATURES]], columns=FEATURES)


def rings(size_nm: float, rmw_nm: float) -> list[tuple[str, float]]:
    """(label, radius in metres). NOAA's Size is already a radius, so it is not halved."""
    return [
        ("Outer closed isobar (predicted size)", size_nm * NM_TO_M),
        ("Radius of maximum winds", rmw_nm * NM_TO_M),
    ]


def exposure(addresses: pd.DataFrame, center: tuple[float, float], size_nm: float, low_elev_m: float) -> pd.DataFrame:
    """Distance to the centre, whether inside the circulation, and a risk score per address.

    Risk falls from 1 at the centre to 0 at the outer isobar, times how far below
    ``low_elev_m`` the address sits. Where elevation is unknown (no DEM) the address is not
    treated as sea level: its risk is the distance part alone and it is not counted as low.
    """
    out = addresses.copy()
    d = haversine_nm(out["lat"].to_numpy(float), out["lon"].to_numpy(float), *center)
    out["distance_nm"] = d
    out["inside"] = d <= size_nm
    near = np.clip((size_nm - d) / max(size_nm, 1e-6), 0, 1)
    elev = out["elevation"].to_numpy(float)
    known = ~np.isnan(elev)
    low = np.where(known, np.clip((low_elev_m - np.nan_to_num(elev)) / max(low_elev_m, 1e-6), 0, 1), 1.0)
    out["risk"] = near * low
    out["low_elevation"] = known & (elev <= low_elev_m)
    out["risk_level"] = pd.cut(out["risk"], bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    return out


def summary(exposed: pd.DataFrame) -> dict:
    inside = exposed["inside"]
    return {
        "total": int(len(exposed)),
        "inside": int(inside.sum()),
        "low_elevation_inside": int((inside & exposed["low_elevation"]).sum()),
        "elevation_known": bool(exposed["elevation"].notna().any()),
    }
