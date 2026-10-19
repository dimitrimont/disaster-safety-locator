import numpy as np

from hurricane.data import load_landfalls
from hurricane.model import FEATURES, make_forest, storm_splits, train


def test_no_storm_on_both_sides_of_a_split():
    df = load_landfalls()
    for train_idx, test_idx in storm_splits(df, n_splits=20):
        assert not set(df["storm"].iloc[train_idx]) & set(df["storm"].iloc[test_idx])


def test_forest_uses_only_the_six_storm_measures():
    df = load_landfalls()
    model = train(df)
    assert list(model.feature_names_in_) == FEATURES
    assert not {"Year", "Month", "Day", "Hour"} & set(FEATURES)


def test_outer_isobar_moves_the_prediction():
    # The first version's OCI slider (120-360) never changed its answer. In mb it does.
    df = load_landfalls()
    model = make_forest(0).fit(df[FEATURES], df["Size_nm"])
    import pandas as pd

    rows = pd.DataFrame([[970, 89, 30, oci, 34.2257, -77.9447] for oci in (1002, 1010, 1018)], columns=FEATURES)
    pred = model.predict(rows)
    assert np.ptp(pred) > 10


def test_network_answers_in_nautical_miles():
    # The first notebook refitted its one scaler to the sizes, so its network answered in 0-1
    # units (and a raw input row gave -65). The fixed network inverts its own size scaler.
    import pytest

    pytest.importorskip("tensorflow")
    from hurricane.deep import SizeNetwork

    df = load_landfalls()
    net = SizeNetwork(seed=0, epochs=60).fit(df[FEATURES], df["Size_nm"])
    pred = net.predict(df[FEATURES].iloc[:20])
    assert 75 <= pred.min() and pred.max() <= 475
