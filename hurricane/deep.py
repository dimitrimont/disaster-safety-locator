"""The neural network from ``Deep_ML_Implementation/Hurricane_DeepML.ipynb``, fixed.

Same layers and training as the notebook (Dense 64 + LeakyReLU 0.1 + Dropout 0.2, Dense 32
ReLU, Dense 1; log-cosh loss, Adam 0.001, 300 epochs, batches of 80). What changed:

- The notebook fitted one MinMaxScaler to the inputs and then refitted the same object to the
  sizes, so the input scaling was lost and nothing could turn a prediction back into nautical
  miles. Inputs and sizes now have their own scalers, fitted on training rows only.
- ``num`` (the row number, which runs in date order) and the date parts were inputs. The
  network now sees the same six storm measures as the forest.
- Rows with any missing value were dropped (54 of 170); missing values are now imputed.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

NETWORK_FILE = "size_network.keras"
SCALERS_FILE = "size_network_scalers.joblib"


def build(n_inputs: int):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(n_inputs,)),
            keras.layers.Dense(64),
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(loss=keras.losses.LogCosh(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
    return model


class SizeNetwork:
    """fit/predict like a scikit-learn regressor; predictions are in nautical miles."""

    def __init__(self, seed: int = 0, epochs: int = 300, batch_size: int = 80):
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size

    def fit(self, X, y):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        self.x_scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        self.y_scaler_ = MinMaxScaler().fit(y)
        Xs = self.x_scaler_.transform(self.imputer_.transform(X))
        self.model_ = build(X.shape[1])
        self.model_.fit(Xs, self.y_scaler_.transform(y), epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        Xs = self.x_scaler_.transform(self.imputer_.transform(np.asarray(X, dtype=float)))
        return self.y_scaler_.inverse_transform(self.model_.predict(Xs, verbose=0)).ravel()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model_.save(directory / NETWORK_FILE)
        joblib.dump({"imputer": self.imputer_, "x": self.x_scaler_, "y": self.y_scaler_}, directory / SCALERS_FILE)

    @classmethod
    def load(cls, directory: Path) -> "SizeNetwork":
        from tensorflow import keras

        net = cls()
        net.model_ = keras.models.load_model(directory / NETWORK_FILE)
        parts = joblib.load(directory / SCALERS_FILE)
        net.imputer_, net.x_scaler_, net.y_scaler_ = parts["imputer"], parts["x"], parts["y"]
        return net
