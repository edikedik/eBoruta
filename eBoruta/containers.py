"""
Types holding intermediate and final data for the algorithm.
"""
from __future__ import annotations

import logging
import typing as t
from collections import abc, Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from eBoruta.base import _X, _Y, ValidationError
from eBoruta.dataprep import prepare_x, prepare_y, prepare_w, has_missing
from eBoruta.utils import get_duplicates

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


# @dataclass
class Dataset(t.Generic[_X, _Y]):
    """
    A container holding permanent data (x, y and weights) for
    training/validation/testing/etc.
    """

    def __init__(self, x: t.Any, y: t.Any, w: t.Any = None, min_features: int = 5):
        self._x, self._y, self._w = prepare_x(x), prepare_y(y), prepare_w(w)
        self.min_features = min_features

        if has_missing(self.y):
            raise ValidationError("Missing values in y")
        if len(self.x) != len(self.y):
            raise ValidationError(
                f"The number of observations in x {len(self.x)} "
                f"does not match the number in y {len(self.y)}"
            )
        if self.w is not None and len(self.x) != len(self.w):
            raise ValidationError(
                f"The number of observations in x {len(self.x)} "
                f"does not match the number in w {len(self.w)}"
            )

    @property
    def x(self) -> pd.DataFrame:
        """
        :return: Variables' dataframe.
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        :return: Target variables' array.
        """
        return self._y

    @property
    def w(self) -> np.ndarray | None:
        """
        :return: Sample weights' array.
        """
        return self._w

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
    # TODO: consider adding slicing on steps and selecting columns pandas-like
    """
    A dynamic container representing a set of features used by Boruta
    throughout the run.

    It's created internally and maintained by :class:`eBoruta.algorithm.eBoruta`.
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

    def __len__(self) -> int:
        return len(self.hit_history)

    def __getitem__(self, item: t.Any) -> t.Self:
        def get_selectors() -> tuple[int | slice, list]:
            match item:
                case [int() | slice(), abc.Sequence()]:
                    if len(item) != 2:
                        raise IndexError('Too many indexing items')
                    return item
                case slice():
                    return item, list(self.names)
                case list():
                    return slice(1, len(self.names)), item
                case _:
                    raise IndexError('Unsupported idx type')

        steps, cols = get_selectors()
        if isinstance(steps, int):
            steps = [steps]

        new = Features(np.intersect1d(self.names, cols))

        new.hit_history = self.hit_history.iloc[steps][cols].copy()
        new.imp_history = self.imp_history.iloc[steps][cols + ['Threshold']].copy()
        new.dec_history = self.dec_history.iloc[steps][cols].copy()

        decisions = new.dec_history.iloc[-1]
        new.accepted_mask = decisions == 1
        new.rejected_mask = decisions == -1
        new.tentative_mask = decisions == 0

        return new

    @property
    def shape(self) -> tuple[int, int]:
        """
        :return: (# steps, # features)
        """
        return len(self), len(self.names)

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

    def accepted_at_step(self, step: int) -> np.ndarray:
        """
        :param step: Step (trial) number.
        :return: Feature names accepted at `step`.
        """
        df = self.history
        return df[(df.Step == step) & (df.Decision == 'Accepted')]['Feature'].values

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
