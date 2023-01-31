"""
Types holding intermediate and final data for the algorithm.
"""
from __future__ import annotations

import logging
import typing as t
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

from eBoruta.base import _X, _Y, _W
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
        self.x = self.prepare_x(self.x)
        self.y = self.convert_y(self.y)
        self.w = self.convert_w(self.w)
        x_missing = self._check_input(self.x)
        y_missing = self._check_input(self.y)

        if y_missing:
            raise AttributeError("Missing values in y")
        if x_missing:
            LOGGER.warning("Detected missing values in x")

    @staticmethod
    def prepare_x(x: _X) -> pd.DataFrame:
        """
        Prepare input variables.

        If it's a 2D array or something convertible
        to one, create a ``pd.DataFrame`` with variables named "1", ... "N",
        where "N" is the number of columns.
        If it's already a ``DataFrame``, copy and reset it's index.

        Then, the ``DataFrame`` is validated by :func:`sklearn.util.
        validation.check_array` to contain 2D, not sparce and potentially
        NaN-containing array of values.

        :param x: Input data.
        :return: A DataFrame verified for the algorithm's usage.
        """
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
        # TODO: consider dropping forcing 1D input to allow multiple targets.
        """
        Prepare target variables.

        If ``y`` is a ``pd.DataFrame`` or ``pd.Series``, take its values and
        apply ``np.squeeze`` to remove redundant dimensions.
        If ``y`` is an ``np.array``, pass. Otherwise, try converting into
        an ``np.array``. Finally, check array doesn't contain ``NaN``.

        :param y: input data.
        :return: an array containing target variable.
        """
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.squeeze(y.values)
        elif isinstance(y, np.ndarray):
            pass
        else:
            LOGGER.warning("Trying to convert y into an array")
            y = convert_to_array(y)
        check_array(y, ensure_2d=False)
        return y

    @staticmethod
    def convert_w(w: _W | None) -> np.ndarray | None:
        """
        Prepare sample weights.

        :param w: A series, array or something convertible to a 1D array.
        :return: Sample weights applied in models supporting ones.
        """
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
    def _check_input(a: t.Any) -> t.TypeGuard[pd.DataFrame | pd.Series | np.ndarray]:
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
        self, columns: None | list[str] | np.ndarray = None, **kwargs
    ) -> TrialData:
        """
        Generates data for a single Boruta trial based on :attr:`x`, :attr:`y`,
        and :attr:`w`. Creates a copy of :attr:`x`, permutes rows, and renames
        columns as "shadow_{original_name}". Concatenates original dataframe
        and the one with the shadow features to create a copy of the learning
        data with at least twice as many features.

        If the number of features in :attr:`x` after selecting by ``columns``
        is below :attr:`min_features`, randomly oversample existing features
        to account for the difference. Thus, the returned dataframe to always
        have at least :attr:`min_features` columns.

        :param columns: An optional list or array of columns to select from
            :attr:`x`.
        :param kwargs: Keyword args passed to :func:`train_test_split` used to
            create train/test splits. Enable this feature by passing
            ``test_size={f}`` where ``f`` is the test size fraction.
            This allows using different datasets for training and importance
            computation.
        :return: A prepared trial data.
        :raises RuntimeError: If resulting features have duplicate names.
        """
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
    """
    A dynamic container representing a set of features used by Boruta
    throughout the run.
    """
    #: An array of feature names.
    names: np.ndarray
    accepted_mask: np.ndarray = field(init=False)
    rejected_mask: np.ndarray = field(init=False)
    tentative_mask: np.ndarray = field(init=False)
    hit_history: pd.DataFrame = field(init=False)
    imp_history: pd.DataFrame = field(init=False)
    dec_history: pd.DataFrame = field(init=False)
    _history: pd.DataFrame | None = None

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
        """
        return: An array of feature names marked as accepted.
        """
        return self.names[self.accepted_mask]

    @property
    def rejected(self) -> np.ndarray:
        """
        :return: An array of feature names marked as rejected.
        """
        return self.names[self.rejected_mask]

    @property
    def tentative(self) -> np.ndarray:
        """
        :return: An array of feature names marked as tentative.
        """
        return self.names[self.tentative_mask]

    @property
    def history(self) -> pd.DataFrame:
        """
        :return: A history dataframe created using :meth:`compose_summary` if
            it doesn't exist.
        """
        if self._history is None:
            self._history = self.compose_history()
        return self._history

    def compose_history(self) -> pd.DataFrame:
        """
        Access the selection history and compose a summary table.

        :return: A history dataframe.
        """
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
        """
        Bulk-:meth:`pd.DataFrame.reset_index`. of importance, decision and
        hit history dataframes.
        """
        for df in [self.imp_history, self.dec_history, self.hit_history]:
            df.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    raise RuntimeError
