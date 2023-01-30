"""
Base types and objects to inherit from.
"""
from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_X = t.TypeVar("_X", pd.DataFrame, np.ndarray)
_Y = t.TypeVar("_Y", pd.DataFrame, pd.Series, np.ndarray)
_W = t.TypeVar("_W", pd.Series, np.ndarray)


class Estimator(t.Protocol):
    """
    An estimator protocol encapsulating methods strictly necessary for
    the main algorithm's functioning.
    """

    def fit(self, x, y, **kwargs) -> Estimator:
        """
        Fit the estimator.
        """

    def predict(self, x: _X, **kwargs) -> np.ndarray:
        """
        Make predictions.
        """

    def get_params(self) -> dict[str, t.Any]:
        """
        Get a dict with the estimator's params.
        """


_E = t.TypeVar("_E", RandomForestClassifier, RandomForestRegressor, Estimator)

if __name__ == "__main__":
    raise RuntimeError
