import logging
import typing as t
from functools import partial

import numpy as np
import pandas as pd

from eBoruta.base import _E, _X
from eBoruta.containers import Features, Dataset, TrialData

CallbackReturn = t.Tuple[_E, Features, Dataset, TrialData, t.Dict[str, t.Any]]
Score = t.Callable[[_E, _X, _X], np.ndarray]

LOGGER = logging.getLogger(__name__)


class CallbackFN(t.Protocol):
    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        ...


class CallbackClass(t.Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        ...


Callback = t.Union[CallbackFN, CallbackClass]


def reduce_by_fraction(num_features: int, frac: float):
    return int(num_features * frac)


def change_params_and_reinit(estimator: _E, update: t.Mapping[str, t.Any]):
    params = estimator.get_params()
    params.update(**update)
    estimator = estimator.__class__(**params)
    return estimator


class IterationAdjuster:
    def __init__(
        self,
        param_name: str,
        min_value: int,
        reducer: t.Callable[[int], int] = partial(reduce_by_fraction, frac=0.5),
    ):
        self.param_name = param_name
        self.min_value = min_value
        self.reducer = reducer
        self.history: t.List[t.Tuple[int, int]] = []

    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        num_features = len(features.tentative) * 2
        new_num_features = max([self.min_value, self.reducer(num_features)])
        estimator = change_params_and_reinit(
            estimator, {self.param_name: new_num_features}
        )
        self.history.append((num_features, new_num_features))
        return estimator, features, dataset, trial_data, kwargs


class Scorer:
    def __init__(self, scorers: t.Mapping[str, Score], verbose: int = 2):
        self.scorers = scorers
        self.score_hist = pd.DataFrame()
        self.verbose = verbose
        if verbose == 2:
            self._log_fn = LOGGER.info
        elif verbose == 1:
            self._log_fn = LOGGER.debug
        else:
            self._log_fn = lambda x: x

    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        d = {
            score_name: score(estimator, trial_data.x_test, trial_data.y_test)
            for score_name, score in self.scorers.items()
        }
        d["Features"] = ";".join(trial_data.x_test.columns)
        self.score_hist = pd.concat([self.score_hist, pd.DataFrame.from_records([d])])
        self._log_fn(d)
        return estimator, features, dataset, trial_data, kwargs


class CatFeaturesSupplier:
    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        if hasattr(estimator, "cat_features"):
            cat_features = [
                i
                for i, c in enumerate(trial_data.x_test.columns)
                if pd.api.types.is_categorical_dtype(trial_data.x_test[c])
            ]
            estimator = change_params_and_reinit(
                estimator, {"cat_features": cat_features}
            )
        return estimator, features, dataset, trial_data, kwargs


class EvalSetSupplier:
    def __init__(self, param_name: str = "eval_set"):
        self.param_name = param_name

    def __call__(
        self,
        estimator: _E,
        features: Features,
        dataset: Dataset,
        trial_data: TrialData,
        **kwargs,
    ) -> CallbackReturn:
        if self.param_name in kwargs:
            LOGGER.debug(f"Overwriting existing {self.param_name}")
        kwargs[self.param_name] = [(trial_data.x_test, trial_data.y_test)]
        return estimator, features, dataset, trial_data, kwargs


if __name__ == "__main__":
    raise RuntimeError
