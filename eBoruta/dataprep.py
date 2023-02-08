"""
Data preparation utils.
"""
from __future__ import annotations

import logging
import typing as t
from collections import abc

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from eBoruta.base import ValidationError
from eBoruta.utils import convert_to_array

LOGGER = logging.getLogger(__name__)


def _check_array(a: t.Any, **kwargs):
    try:
        check_array(a, **kwargs)
    except Exception as e:
        raise ValidationError("Failed to validate an array with `check_array()`") from e


def prepare_x(x: t.Any) -> pd.DataFrame:
    """
    Prepare input variables.

    If it's a 2D array or something convertible
    to one, create a ``pd.DataFrame`` with variables named "1", ... "N",
    where "N" is the number of columns.
    If it's already a ``DataFrame``, copy and reset its index.

    Then, the ``DataFrame`` is validated by :func:`sklearn.util.
    validation.check_array` to contain 2D, not sparce and potentially
    NaN-containing array of values.

    :param x: Input data.
    :return: A DataFrame verified for the algorithm's usage.
    """
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            raise ValidationError("Reshape your data: 1D input for x is not allowed")
        x = pd.DataFrame(x, columns=list(map(str, range(1, x.shape[1] + 1))))
    elif isinstance(x, pd.DataFrame):
        x = x.copy().reset_index(drop=True)
    else:
        LOGGER.debug("Trying to convert x into an array")
        x = convert_to_array(x)
        num_features = x.shape[1] if len(x.shape) == 2 else 1
        x = pd.DataFrame(x, columns=list(map(str, range(1, num_features + 1))))
    assert isinstance(x, pd.DataFrame), "Unsuccessful converting to df"
    _check_array(
        x.values, force_all_finite="allow-nan", ensure_2d=False, accept_sparse=False
    )
    return x


def prepare_y(y: t.Any) -> np.ndarray:
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
        y = y.values
    elif isinstance(y, np.ndarray):
        pass
    else:
        LOGGER.debug("Trying to convert y into an array")
        y = convert_to_array(y)
    assert isinstance(y, np.ndarray), "Failed converting to np array"
    y = np.squeeze(y)  # type: ignore  # ignore type-arg
    _check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_features=True)
    return y


def prepare_w(w: t.Any) -> np.ndarray | None:
    """
    Prepare sample weights.

    :param w: A series, array or something convertible to a 1D array.
    :return: Sample weights applied in models supporting ones.
    """
    if w is None:
        return None
    if isinstance(w, (pd.DataFrame, pd.Series)):
        w = w.values
    elif isinstance(w, np.ndarray):
        pass
    else:
        LOGGER.debug("Trying to convert w into an array")
        w = convert_to_array(w)
    w = np.squeeze(w)
    assert isinstance(w, np.ndarray), "Failed converting to np array"
    _check_array(w, ensure_2d=False, force_all_finite=True, ensure_min_features=True)
    if len(w.shape) != 1:
        raise ValidationError(f"Weights must be 1D. Got shape(w)={w.shape}")
    return w


def has_missing(a: t.Any) -> bool:
    """
    Check if an array-like object has missing values.

    :param a: Any object. The function checks only types ``pd.DataFrame``,
        ``pd.Series``, and ``np.array``.
    :return: ``True`` if input has missing values (NaN) else ``False``. Also,
        if attempting to find missing values results in exception, return
        ``False``.
    """
    try:
        if isinstance(a, pd.DataFrame):
            return a.isna().any().any()
        if isinstance(a, pd.Series):
            return a.isna().any()
        if isinstance(a, np.ndarray):
            res = np.any(np.isnan(a).flatten())
            return res
        if isinstance(a, abc.Sequence):
            res = np.isnan(np.array(a)).any()
            return res
        LOGGER.warning(f"Unsupported input array type {type(a)}")
        return False
    except Exception as e:
        LOGGER.warning(f"Failed to check input for missing values due to {e}")
        LOGGER.exception(e)
        return False


if __name__ == "__main__":
    raise RuntimeError
