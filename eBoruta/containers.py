"""
Types holding intermediate and final data for the algorithm.
"""
import logging
import typing as t
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

from eBoruta.base import _X, _Y, _W, _E
from eBoruta.utils import convert_to_array, get_duplicates

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialData(t.Generic[_Y]):
    """
    Data for a Boruta trial.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: _Y
    y_test: _Y
    #: Weights for train and test folds
    w_train: t.Optional[np.ndarray] = None
    w_test: t.Optional[np.ndarray] = None

    @property
    def shapes(self) -> str:
        """
        Descriptor property.

        :return: a string with shapes of x and y attributes.
        """
        return (
            f"x_train: {self.x_train.shape}, y_train: {self.y_train.shape}, "
            f"x_test: {self.x_test.shape}, y_test: {self.y_test.shape}"
        )


@dataclass
class Dataset(t.Generic[_X, _Y]):
    """
    A container holding permanent data (x, y and weights) for
    training/validation/testing/etc.
    """

    x: _X
    y: _Y
    w: t.Optional[np.ndarray] = None
    min_features: t.Optional[int] = 5

    def __post_init__(self):
        self.x = self.convert_x(self.x)
        self.y = self.convert_y(self.y)
        self.w = self.convert_w(self.w)
        x_missing = self._check_input(self.x)
        y_missing = self._check_input(self.y)

        if y_missing:
            raise AttributeError("Missing values in y")
        if x_missing:
            LOGGER.warning("Detected missing values in x")

    @staticmethod
    def convert_x(x: _X) -> pd.DataFrame:
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                raise ValueError("Reshape your data: 1D input for x is not allowed")
            x = pd.DataFrame(x, columns=list(map(str, range(1, x.shape[1] + 1))))
        elif isinstance(x, pd.DataFrame):
            x = x.copy().reset_index(drop=True)
        else:
            LOGGER.warning("Trying to convert x into an array")
            x = convert_to_array(x)
            num_features = x.shape[1] if len(x.shape) == 2 else 1
            x = pd.DataFrame(x, columns=list(map(str, range(1, num_features + 1))))
        check_array(
            x.values, force_all_finite="allow-nan", ensure_2d=False, accept_sparse=False
        )
        return x

    @staticmethod
    def convert_y(y: _Y) -> np.ndarray:
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, np.ndarray):
            pass
        else:
            LOGGER.warning("Trying to convert y into an array")
            y = convert_to_array(y)
        check_array(y, ensure_2d=False)
        return y

    @staticmethod
    def convert_w(w: t.Optional[_W]) -> t.Optional[np.ndarray]:
        if w is None:
            return None

        if isinstance(w, pd.Series):
            w = w.values
        elif isinstance(w, np.ndarray):
            pass
        else:
            LOGGER.warning("Trying to convert w into an array")
            w = convert_to_array(w)
        check_array(w, ensure_2d=False)
        return w

    @staticmethod
    def _check_input(a: t.Union[pd.DataFrame, pd.Series, np.ndarray]) -> bool:
        try:
            if isinstance(a, pd.DataFrame):
                return a.isna().any().any()
            if isinstance(a, pd.Series):
                return a.isna().any()
            if isinstance(a, np.ndarray):
                return np.isnan(a).any()
            LOGGER.warning(f"Unsupported input array type {type(a)}")
            return False
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning(f"Failed to check input for missing values due to {e}")
            return False

    def generate_trial_sample(
        self, columns: t.Union[None, t.List[str], np.ndarray] = None, **kwargs
    ) -> TrialData:
        if columns is None:
            columns = list(self.x.columns)
        if not isinstance(columns, list):
            columns = list(columns)

        x_init = self.x[columns].copy()
        LOGGER.debug(f"Using columns {columns} as features")
        x_shadow = (
            self.x[columns]
            .copy()
            .sample(frac=1)
            .reset_index(drop=True)
            .rename(columns={c: f"shadow_{c}" for c in columns})
        )

        if self.min_features is not None:
            n_add_samples = self.min_features - len(columns)
            if n_add_samples > 0:
                sampled = (
                    x_shadow.sample(n=n_add_samples, axis=1, replace=True)
                    .sample(frac=1)
                    .reset_index(drop=True)
                )
                name_counts = Counter(sampled.columns)
                for name, count in name_counts.items():
                    d = {name: [f"{name}_{i}" for i in range(count)]}
                    sampled.rename(
                        columns=lambda c: d[c].pop(0) if c in d.keys() else c,
                        inplace=True,
                    )
                x_shadow = pd.concat([x_shadow, sampled], axis=1)
                LOGGER.debug(
                    f"Added {sampled.shape[1]} columns to reach "
                    f"the min number of features {self.min_features}"
                )

        LOGGER.debug(
            f"Created a dataset of shadow features with shape {x_shadow.shape}"
        )
        x = pd.concat([x_init, x_shadow], axis=1)
        duplicates = list(get_duplicates(x.columns))
        if duplicates:
            raise RuntimeError(f"Features contain duplicate names {duplicates}")
        LOGGER.debug(
            f"Merged with initial dataset to get a dataset with shape {x.shape}"
        )
        y = self.y.copy()
        w = self.w.copy() if self.w is not None else self.w

        test_size = kwargs.get("test_size")
        if test_size is None:
            return TrialData(x, x, y, y, w, w)
        if self.w is None:
            return TrialData(*train_test_split(x, y, **kwargs))
        return TrialData(*train_test_split(x, y, w, **kwargs))


@dataclass
class Features:
    names: np.ndarray
    accepted_mask: np.ndarray = field(init=False)
    rejected_mask: np.ndarray = field(init=False)
    tentative_mask: np.ndarray = field(init=False)
    hit_history: pd.DataFrame = field(init=False)
    imp_history: pd.DataFrame = field(init=False)
    dec_history: pd.DataFrame = field(init=False)
    _history: t.Optional[pd.DataFrame] = None

    def __post_init__(self):
        n = len(self.names)
        self.accepted_mask, self.rejected_mask = np.zeros(n).astype(bool), np.zeros(
            n
        ).astype(bool)
        self.tentative_mask = np.ones(n).astype(bool)
        self.hit_history = pd.DataFrame(columns=self.names)
        self.imp_history = pd.DataFrame(columns=self.names)
        self.dec_history = pd.DataFrame(columns=self.names)

    @property
    def accepted(self) -> np.ndarray:
        return self.names[self.accepted_mask]

    @property
    def rejected(self) -> np.ndarray:
        return self.names[self.rejected_mask]

    @property
    def tentative(self) -> np.ndarray:
        return self.names[self.tentative_mask]

    @property
    def history(self) -> pd.DataFrame:
        if self._history is None:
            self._history = self.compose_history()
        return self._history

    def compose_history(self) -> pd.DataFrame:
        if self._history is not None:
            LOGGER.warning(
                f"Overwriting existing history with shape {self._history.shape}"
            )
        self.reset_history_index()
        imp = self.melt_history(
            self.imp_history.drop(columns="Threshold"), "Importance"
        )
        hit = self.melt_history(self.hit_history, "Hit")
        dec = self.melt_history(self.dec_history, "Decision")
        threshold = self.imp_history.reset_index().rename(columns={"index": "Step"})[
            ["Step", "Threshold"]
        ]
        _steps = imp["Step"].values
        _feature = imp["Feature"].values
        df = pd.concat(
            (_x.drop(columns=["Step", "Feature"]) for _x in [imp, hit, dec]), axis=1
        )
        df["Step"] = _steps
        df["Feature"] = _feature
        df = df[["Feature", "Step", "Importance", "Hit", "Decision"]].merge(
            threshold, on="Step", how="left"
        )
        df["Decision"] = df["Decision"].map(
            {0: "Tentative", -1: "Rejected", 1: "Accepted"}
        )
        df["Step"] += 1
        return df

    @staticmethod
    def melt_history(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        df = df.copy()
        columns = df.columns
        df["Step"] = np.arange(len(df), dtype=int)
        df = df.melt(
            id_vars="Step",
            value_vars=columns,
            var_name="Feature",
            value_name=value_name,
        )
        return df

    def reset_history_index(self) -> None:
        for df in [self.imp_history, self.dec_history, self.hit_history]:
            df.reset_index(drop=True, inplace=True)


class ImportanceGetter(t.Protocol):
    def __call__(
        self, estimator: _E, trial_data: t.Optional[TrialData] = None
    ) -> np.ndarray:
        ...


class CVImportanceGetter:
    # TODO: A special type of importance getter: `fit` is ommitted in the core loop and instead performed
    # within this class, computing importances in a CV manner and aggregating the results.
    # Thus, should be as abstract as possible allowing for custom importance evaluations and CV protocols.
    pass


if __name__ == "__main__":
    raise RuntimeError
