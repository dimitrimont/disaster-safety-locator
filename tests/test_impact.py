import numpy as np
import pandas as pd
import pytest

from hurricane.impact import NM_TO_M, exposure, haversine_nm, rings, storm_row, summary
from hurricane.model import FEATURES


def test_one_degree_of_latitude_is_60_nm():
    # 6,371 km earth: one degree is 111.195 km = 60.04 nm.
    assert haversine_nm(34.0, -78.0, 35.0, -78.0) == pytest.approx(60.04, abs=0.01)


def test_rings_use_the_full_size():
    # NOAA's Size is a radius. The first version halved it (a 283.94 nm storm drew a 141.97 nm ring).
    (outer_label, outer), (_, inner) = rings(283.94, 30)
    assert outer == pytest.approx(283.94 * NM_TO_M)
    assert inner == pytest.approx(30 * NM_TO_M)


def test_storm_row_is_exactly_the_model_inputs():
    row = storm_row(34.2257, -77.9447, 89, 970, 30, 1010)
    assert list(row.columns) == FEATURES
    assert row.loc[0, "OCI_mb"] == 1010 and row.loc[0, "Long"] == -77.9447


def test_exposure_counts_and_unknown_elevation():
    center = (34.0, -78.0)
    # 0, 30 and 90 nm north of the centre, the middle one at sea level, the last unknown.
    pts = pd.DataFrame({"lat": [34.0, 34.5, 35.5], "lon": [-78.0, -78.0, -78.0], "elevation": [10.0, 0.0, np.nan]})
    out = exposure(pts, center, size_nm=60, low_elev_m=5)
    assert list(out["inside"]) == [True, True, False]
    assert out["distance_nm"].iloc[1] == pytest.approx(30.02, abs=0.01)
    # Risk: (60 - d) / 60 times how far below 5 m. 10 m -> 0; 0 m at 30 nm -> 0.5.
    assert out["risk"].iloc[0] == 0
    assert out["risk"].iloc[1] == pytest.approx((60 - 30.02) / 60, abs=1e-3)
    assert not out["low_elevation"].iloc[2]  # unknown is not "sea level"
    assert summary(out) == {"total": 3, "inside": 2, "low_elevation_inside": 1, "elevation_known": True}
