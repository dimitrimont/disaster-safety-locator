"""The size model: a random forest on the six storm measures, judged on storms it has not seen.

The first version (``Hurricane_Predicton_RandomForest.ipynb``) scored one random split of
landfalls. Two things made that number look better than the model is:

- One split is a draw. Its split (seed 42) scored R^2 0.79; across 300 splits of the same kind
  the middle one scores 0.35, and none of the other 299 reaches 0.79.
- A storm that crosses several coasts has one row per landfall with the same Size, so a random
  split puts the same storm on both sides. Here every evaluation holds whole storms out.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from .data import STORM_MEASURES

FEATURES = STORM_MEASURES
TARGET = "Size_nm"
MODEL_DIR = Path(__file__).resolve().parent.parent / "Hurricane_model"
FOREST_FILE = "size_forest.joblib"


def make_forest(seed: int = 0, n_estimators: int = 300):
    """Median imputation (RMW is missing on 40 landfalls) and a forest with leaves of 5 or more."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=5, random_state=seed, n_jobs=-1
        ),
    )


def storm_splits(df: pd.DataFrame, n_splits: int = 100, test_size: float = 0.2, seed: int = 1):
    """Train/test splits that never put one storm's landfalls on both sides."""
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return list(splitter.split(df, df[TARGET], groups=df["storm"]))


def evaluate(df: pd.DataFrame, make_model, features=FEATURES, splits=None) -> dict:
    """Mean absolute error and R^2 on each split, beside always guessing the training mean."""
    splits = splits if splits is not None else storm_splits(df)
    X = df[features].astype(float)
    y = df[TARGET].astype(float).to_numpy()
    mae, r2, guess = [], [], []
    for k, (train, test) in enumerate(splits):
        model = make_model(k).fit(X.iloc[train], y[train])
        pred = model.predict(X.iloc[test])
        mae.append(mean_absolute_error(y[test], pred))
        r2.append(r2_score(y[test], pred))
        guess.append(mean_absolute_error(y[test], np.full(len(test), y[train].mean())))
    return {"mae": np.array(mae), "r2": np.array(r2), "mean_guess_mae": np.array(guess)}


def train(df: pd.DataFrame, seed: int = 0):
    """The forest the app uses, trained on every landfall."""
    return make_forest(seed).fit(df[FEATURES].astype(float), df[TARGET].astype(float))


def save(model, directory: Path = MODEL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / FOREST_FILE)
    (directory / "feature_columns.txt").write_text("\n".join(FEATURES) + "\n")
    return directory / FOREST_FILE


def load(directory: Path = MODEL_DIR):
    return joblib.load(directory / FOREST_FILE)
