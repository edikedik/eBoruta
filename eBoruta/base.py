"""
Base types and objects to inherit from.
"""
from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

if t.TYPE_CHECKING:
    from eBoruta import TrialData

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


class ImportanceGetter(t.Protocol):
    def __call__(
        self, estimator: _E, trial_data: TrialData | None = None
    ) -> np.ndarray:
        ...


# class CVImportanceGetter:
#     # TODO: A special type of importance getter: `fit` is ommitted in the core loop and instead performed
#     # within this class, computing importances in a CV manner and aggregating the results.
#     # Thus, should be as abstract as possible allowing for custom importance evaluations and CV protocols.
#     pass

class ValidationError(ValueError):
    """
    Cases of failure to validate data.
    """
    pass


if __name__ == "__main__":
    raise RuntimeError
