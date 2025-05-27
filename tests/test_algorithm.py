import numpy as np
import pytest
# from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    AdaBoostRegressor,
)
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from eBoruta import eBoruta
from eBoruta.utils import sample_dataset


def get_tree_models():
    # two-element tuples: (1) is regressor (2) model
    return [
        (False, RandomForestClassifier),
        (True, RandomForestRegressor),
        (False, ExtraTreesClassifier),
        (False, XGBClassifier),
        (True, XGBRegressor),
        # (False, CatBoostClassifier),
        # (True, CatBoostRegressor),
    ]


def get_non_tree_models():
    return [
        (True, AdaBoostRegressor),
        (False, RidgeClassifier),
        (False, LogisticRegression),
    ]


def make_dataset(is_reg, **kwargs):
    if is_reg:
        if "make_redundant" in kwargs:
            del kwargs["make_redundant"]
        return make_regression(**kwargs)
    return make_classification(**kwargs)


def test_simple_case():
    x, y = make_classification(n_informative=5)
    boruta = eBoruta()
    boruta.fit(x, y)
    assert 0 < len(boruta.features_.accepted) <= 5


@pytest.mark.parametrize("model", get_tree_models())
@pytest.mark.parametrize("use_weights", [True, False])
@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("n_features", [20])
@pytest.mark.parametrize("n_informative", [5])
def test_models(
    model,
    use_weights,
    n_samples,
    n_features,
    n_informative,
):
    is_reg, model = model
    x, y = make_dataset(
        is_reg, n_features=n_features, n_samples=n_samples, n_informative=n_informative
    )
    w = np.ones(len(y), dtype=float) if use_weights else None
    boruta = eBoruta()
    boruta.fit(x, y, w, model_type=model)
    features = boruta.features_
    exp_tentative = x.shape[1] - len(features.accepted) - len(features.rejected)
    assert exp_tentative == len(features.tentative)
    assert len(features.accepted) > 0


@pytest.mark.parametrize("model", get_non_tree_models())
@pytest.mark.parametrize("use_weights", [True, False])
@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("n_features", [20])
@pytest.mark.parametrize("n_informative", [5])
def test_non_tree_models(
    model,
    use_weights,
    n_samples,
    n_features,
    n_informative,
):
    is_reg, model = model
    x, y = make_dataset(
        is_reg, n_features=n_features, n_samples=n_samples, n_informative=n_informative
    )
    w = np.ones(len(y), dtype=float) if use_weights else None
    boruta = eBoruta()
    boruta.fit(x, y, w, model_type=model)
    features = boruta.features_
    exp_tentative = x.shape[1] - len(features.accepted) - len(features.rejected)
    assert exp_tentative == len(features.tentative)
    assert len(features.accepted) > 0


def test_rank():
    x, y = sample_dataset()
    boruta = eBoruta()
    boruta.fit(x, y)
    hist = boruta.features_.history

    ranks = boruta.rank()
    assert len(ranks) == len(boruta.features_)

    if len(boruta.features_) == 0:
        return

    step_fst_accepted = (
        hist[hist.Decision == "Accepted"].sort_values("Step").iloc[0].Step
    )
    accepted_at_fst = list(hist[
        (hist.Decision == "Accepted") & (hist.Step == step_fst_accepted)
    ].Feature.unique())
    ranks1 = boruta.rank(accepted_at_fst)
    ranks2 = boruta.rank(step=step_fst_accepted)
    ranks3 = boruta.rank(accepted_at_fst, step_fst_accepted)
    assert len(ranks1) == len(ranks2) == len(ranks3) == len(accepted_at_fst)
    assert set(ranks1.Feature) == set(ranks2.Feature)
    assert set(ranks1.Feature) == set(ranks3.Feature)
    assert set(ranks2.Feature) == set(ranks3.Feature)