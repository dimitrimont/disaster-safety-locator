import numpy as np
import pandas as pd

from hurricane.data import assign_storms, load_landfalls, parse_coordinate, parse_date, parse_number


def test_coordinates_west_and_south_are_negative():
    assert parse_coordinate("36.0N") == 36.0
    assert parse_coordinate("75.8W") == -75.8
    assert parse_coordinate("12.5S") == -12.5
    assert np.isnan(parse_coordinate("---"))


def test_footnote_marks_keep_the_date():
    # The first version's pd.to_datetime(errors="coerce") turned these 19 dates into NaT.
    assert parse_date("8/27/1909#") == (pd.Timestamp("1909-08-27"), "#")
    assert parse_date("9/15/1960%") == (pd.Timestamp("1960-09-15"), "%")
    assert parse_date("09/01/1932") == (pd.Timestamp("1932-09-01"), "")


def test_dashes_are_missing():
    assert np.isnan(parse_number("---"))
    assert parse_number("1,010") == 1010.0


def test_table_known_rows():
    df = load_landfalls()
    assert len(df) == 170
    assert (df["Mark"] != "").sum() == 19 and df["Date"].notna().all()
    first = df.iloc[0]  # 7/11/1901 0700Z 36.0N 75.8W 70 kt, (983) mb, OCI 1013, size 175
    assert (first["Lat"], first["Long"], first["Hour"]) == (36.0, -75.8, 7.0)
    assert first["Central_Pressure"] == 983 and first["Pressure_Estimated"]
    assert first["OCI_mb"] == 1013 and first["Size_nm"] == 175
    # NOAA prints OCI 125 for 15 Sep 1945 19:30Z; no outer isobar is that low, so it is missing.
    typo = df[(df["Date"] == "1945-09-15") & (df["Hour"] == 19)].iloc[0]
    assert np.isnan(typo["OCI_mb"])
    assert df["OCI_mb"].between(1001, 1018).sum() == 169


def test_storm_ids_keep_landfalls_of_one_storm_together():
    df = load_landfalls()
    barbara = df[df["Storm_Name"] == "Barbara"]
    assert len(barbara) == 3 and barbara["storm"].nunique() == 1
    # 4 Sep 1933 (Jupiter, FL) and 5 Sep 1933 (Brownsville, TX) are different storms.
    sept_1933 = df[df["Date"].isin([pd.Timestamp("1933-09-04"), pd.Timestamp("1933-09-05")])]
    assert sept_1933["storm"].nunique() == 2
    assert df["storm"].nunique() == 137


def test_assign_storms_rule():
    dates = pd.Series(pd.to_datetime(["1950-08-31", "1950-08-31", "1950-09-05", "1950-09-06"]))
    sizes = pd.Series([250, 250, 325, 300])
    names = pd.Series([None, None, "Easy", "Easy"])
    assert list(assign_storms(dates, sizes, names)) == [0, 0, 1, 1]
