import typing as t

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_X = t.TypeVar("_X", pd.DataFrame, np.ndarray)
_Y = t.TypeVar("_Y", pd.DataFrame, pd.Series, np.ndarray)
_W = t.TypeVar("_W", pd.Series, np.ndarray)


class Estimator(t.Protocol):
    # feature_importances_: t.Union[np.ndarray, t.List[int], t.List[t.List[int]]]
    # In general, something that defines a fit method
    def fit(self, x, y, **kwargs) -> "Estimator":
        ...

    def predict(self, x: _X, **kwargs) -> np.ndarray:
        ...

    def get_params(self) -> t.Dict[str, t.Any]:
        ...


_E = t.TypeVar("_E", RandomForestClassifier, RandomForestRegressor, Estimator)

if __name__ == "__main__":
    raise RuntimeError
