import logging
import typing as t
from functools import partial

import numpy as np
import pandas as pd

from Boruta.base import _E, _X
from Boruta.structures import Features, Dataset, TrialData

CallbackReturn = t.Tuple[_E, Features, Dataset, TrialData]
Score = t.Callable[[_E, _X, _X], np.ndarray]


LOGGER = logging.getLogger(__name__)


class CallbackFN(t.Protocol):
    def __call__(self, estimator: _E, features: Features,
                 dataset: Dataset, trial_data: TrialData, **kwargs) -> CallbackReturn: ...


class CallbackClass(t.Protocol):
    def __init__(self, *args, **kwargs): ...

    def __call__(self, estimator: _E, features: Features,
                 dataset: Dataset, trial_data: TrialData, **kwargs) -> CallbackReturn: ...


Callback = t.Union[CallbackFN, CallbackClass]


def reduce_by_fraction(num_features: int, frac: float):
    return int(num_features * frac)


class IterationAdjuster:
    def __init__(self, param_name: str, min_value: int,
                 reducer: t.Callable[[int], int] = partial(reduce_by_fraction, frac=0.5)):
        self.param_name = param_name
        self.min_value = min_value
        self.reducer = reducer

    def __call__(
            self, estimator: _E, features: Features, dataset: Dataset, trial_data: TrialData) -> CallbackReturn:
        num_features = len(features.tentative) * 2
        new_num_features = self.reducer(num_features)
        params = estimator.get_params()
        params[self.param_name] = max([self.min_value, new_num_features])
        estimator = estimator.__class__(**params)
        return estimator, features, dataset, trial_data


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
            self, estimator: _E, features: Features, dataset: Dataset, trial_data: TrialData) -> CallbackReturn:
        scores = {
            score_name: score(estimator, trial_data.x_test, trial_data.y_test) for score_name, score in self.scorers}
        self.score_hist = pd.concat([self.score_hist, pd.DataFrame.from_records([scores])])
        self._log_fn(scores)
        return estimator, features, dataset, trial_data


if __name__ == '__main__':
    raise RuntimeError
