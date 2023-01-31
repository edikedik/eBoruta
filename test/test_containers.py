import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from eBoruta import Dataset
from eBoruta.base import ValidationError


@pytest.mark.parametrize('inputs', [
    make_classification(n_samples=50, n_features=10),
    make_classification(n_samples=10, n_features=5),
    make_regression()
])
def test_dataset(inputs):
    def verify_x(ds):
        assert isinstance(ds.x, pd.DataFrame)
        assert ds.x.shape == x.shape
        assert ds.x.columns[0] == '1' and ds.x.columns[-1] == str(ds.x.shape[1])

    def verify_y(ds):
        assert isinstance(ds.y, np.ndarray)
        assert ds.y.shape == y.shape

    def verify_w(ds):
        assert isinstance(ds.w, np.ndarray)
        assert len(ds.y) == len(ds.w)

    # test valid cases
    x, y = inputs

    # check x
    # converting to df from array
    ds = Dataset(x, y)
    verify_x(ds)
    # passing df changes nothing
    assert (Dataset(ds.x, y).x == ds.x).all().all()
    # converting to df from list of lists
    lol = [list(a) for a in x]
    verify_x(Dataset(lol, y))

    # check y
    verify_y(Dataset(x, pd.Series(y)))
    verify_y(Dataset(x, pd.DataFrame({'Y': y})))
    verify_y(Dataset(x, list(y)))

    with pytest.raises(ValidationError):
        Dataset(x, y[:-1])
    _y = y.copy().astype(float)
    _y[0] = np.nan
    with pytest.raises(ValidationError):
        Dataset(x, _y)

    # check w
    w = np.arange(len(y)).astype(float)
    verify_w(Dataset(x, y, w))
    verify_w(Dataset(x, y, list(w)))
    verify_w(Dataset(x, y, pd.Series(w)))
    w[0] = np.nan
    with pytest.raises(ValidationError):
        verify_w(Dataset(x, y, w))