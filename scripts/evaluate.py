"""Reproduce every number in the README: the first version's split, and both models judged on
storms they have not seen. Writes results/evaluation.json and results/evaluation.md.

    python scripts/evaluate.py            # forest and network, about 10 minutes
    python scripts/evaluate.py --no-network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hurricane.data import load_landfalls  # noqa: E402
from hurricane.model import FEATURES, TARGET, evaluate, make_forest, storm_splits  # noqa: E402

# The first version's thirteen inputs (its notebook, cell 4).
FIRST_FEATURES = ["Lat", "Long", "Max_Winds_kt", "SSHWS", "RMW_nm", "Central_Pressure", "OCI_mb",
                  "Year", "Month", "Day", "Hour", "Abs_Lat", "Wind_to_Pressure"]


def first_version_table(df):
    """The first notebook's inputs: its date parse lost footnote-marked dates, and it kept NOAA's
    125 mb outer isobar."""
    import pandas as pd

    from hurricane.data import DEFAULT_CSV, parse_number

    t = df.copy()
    t["OCI_mb"] = pd.read_csv(DEFAULT_CSV, dtype=str)["OCI_mb"].map(parse_number)
    lost = t["Mark"] != ""
    t["Year"] = t["Date"].dt.year.where(~lost)
    t["Month"] = t["Date"].dt.month.where(~lost)
    t["Day"] = t["Date"].dt.day.where(~lost)
    t["Abs_Lat"] = t["Lat"].abs()
    t["Wind_to_Pressure"] = t["Max_Winds_kt"] / (t["Central_Pressure"] + 1e-6)
    return t


def notebook_split(t, seed):
    """Cell 5 and 6 of the first notebook: 75% train, the rest halved into validation and test."""
    X = t[FIRST_FEATURES].astype(float)
    X = X.fillna(X.median())
    y = t[TARGET].astype(float)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=seed)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    rf = RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=42, n_jobs=-1).fit(X_train, y_train)
    pred = rf.predict(X_test)
    return r2_score(y_test, pred), mean_absolute_error(y_test, pred)


def describe(name, res):
    return {
        "model": name,
        "mae_nm": round(float(res["mae"].mean()), 1),
        "r2_median": round(float(np.median(res["r2"])), 3),
        "mean_guess_mae_nm": round(float(res["mean_guess_mae"].mean()), 1),
        "splits": int(len(res["mae"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=300, help="random splits for the first version's luck check")
    ap.add_argument("--storm-splits", type=int, default=100)
    ap.add_argument("--network-splits", type=int, default=20)
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()

    df = load_landfalls()
    first = first_version_table(df)
    luck = [notebook_split(first, s) for s in range(args.splits)]
    r2s = np.array([r for r, _ in luck])
    out = {
        "landfalls": int(len(df)),
        "storms": int(df["storm"].nunique()),
        "dates_with_marks": int((df["Mark"] != "").sum()),
        "first_version_split": {
            "seed_42_r2": round(float(r2s[42]), 3),
            "seed_42_mae_nm": round(float(luck[42][1]), 2),
            "median_r2": round(float(np.median(r2s)), 3),
            "r2_10th_90th": [round(float(v), 2) for v in np.percentile(r2s, [10, 90])],
            "splits_at_or_above_seed_42": int((r2s >= r2s[42]).sum()),
            "splits": args.splits,
        },
    }

    splits = storm_splits(df, n_splits=args.storm_splits)
    rows = []
    first_forest = lambda k: RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=k, n_jobs=-1)  # noqa: E731
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    rows.append(describe("first forest, 13 inputs", evaluate(first, lambda k: make_pipeline(SimpleImputer(strategy="median"), first_forest(k)), FIRST_FEATURES, splits)))
    rows.append(describe("revised forest, 6 storm measures", evaluate(df, make_forest, FEATURES, splits)))
    rand = list(ShuffleSplit(n_splits=args.storm_splits, test_size=0.2, random_state=1).split(df))
    rows.append(describe("first forest, random rows (for comparison)", evaluate(first, lambda k: make_pipeline(SimpleImputer(strategy="median"), first_forest(k)), FIRST_FEATURES, rand)))
    if not args.no_network:
        from hurricane.deep import SizeNetwork

        few = splits[: args.network_splits]
        rows.append(describe("revised network, 6 storm measures", evaluate(df, lambda k: SizeNetwork(seed=k), FEATURES, few)))
        rows.append(describe("revised forest on the same splits", evaluate(df, make_forest, FEATURES, few)))
    out["storms_held_out"] = rows

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "evaluation.json").write_text(json.dumps(out, indent=2) + "\n")
    fv = out["first_version_split"]
    md = [
        "# Evaluation",
        "",
        f"{out['landfalls']} landfalls from {out['storms']} storms (NOAA HRD, 1901-1970).",
        "",
        "## The first version's split",
        "",
        f"The notebook's split (seed 42) scores R^2 {fv['seed_42_r2']} (MAE {fv['seed_42_mae_nm']} nm). "
        f"Across {fv['splits']} splits of the same kind the median is {fv['median_r2']} "
        f"(10th-90th percentile {fv['r2_10th_90th'][0]} to {fv['r2_10th_90th'][1]}); "
        f"{fv['splits_at_or_above_seed_42']} of {fv['splits']} reach seed 42's score.",
        "",
        "## Storms held out",
        "",
        "| Model | MAE (nm) | median R^2 | always guessing the mean (nm) | splits |",
        "|---|---|---|---|---|",
    ]
    md += [f"| {r['model']} | {r['mae_nm']} | {r['r2_median']} | {r['mean_guess_mae_nm']} | {r['splits']} |" for r in rows]
    (ROOT / "results" / "evaluation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
