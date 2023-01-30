import logging

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from eBoruta.eBoruta import eBoruta

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


@pytest.fixture
def classification_data():
    return make_classification()


@pytest.fixture
def regression_data():
    return make_regression()


@pytest.fixture()
def classifiers():
    models = [('RF', RandomForestClassifier())]
    try:
        from catboost import CatBoostClassifier
        models.append(('catboost', CatBoostClassifier(iterations=20, verbose=False)))
    except ImportError:
        LOGGER.info('failed to find catboost for testing')
    try:
        from xgboost import XGBClassifier
        models.append(('xgboost', XGBClassifier(n_estimators=20, verbosity=0)))
    except ImportError:
        LOGGER.info('failed to find xgboost for testing')

    return models


@pytest.fixture()
def classifiers_multiobjective():
    models = [('RF', RandomForestClassifier())]
    try:
        from xgboost import XGBClassifier
        models.append(('xgboost', XGBClassifier(n_estimators=20, verbosity=0)))
    except ImportError:
        LOGGER.info('failed to find xgboost for testing')
    return models


@pytest.fixture()
def regressors():
    models = [('RF', RandomForestRegressor())]
    try:
        from catboost import CatBoostRegressor
        models.append(('catboost', CatBoostRegressor(iterations=20, verbose=False)))
    except ImportError:
        LOGGER.info('failed to find catboost for testing')
    try:
        from xgboost import XGBRegressor
        models.append(('xgboost', XGBRegressor(n_estimators=20, verbosity=0)))
    except ImportError:
        LOGGER.info('failed to find xgboost for testing')

    return models


@parametrize_with_checks([eBoruta()])
def test_sklearn_compatible(estimator, check):
    # not expected to pass all the tests (even sklearn models don't seem to), although passes ~90%
    # even though, the test is convenient to check which functionality is absent
    check(estimator)


def test_models(classifiers, classification_data, regressors, regression_data):
    x, y = classification_data
    boruta = eBoruta(n_iter=20, verbose=0)
    for name, model in classifiers:
        try:
            boruta.shap_approximate = name != 'catboost'
            boruta.fit(x, y, model=model)
        except Exception as e:
            assert False, f'Failed to run classifier {name} due to {e}'
    x, y = regression_data
    boruta = eBoruta(n_iter=20, verbose=0, classification=False, test_stratify=False)
    for name, model in regressors:
        try:
            boruta.shap_approximate = name != 'catboost'
            boruta.fit(x, y, model=model)
        except Exception as e:
            assert False, f'Failed to run regressor {name} due to {e}'


def test_params(classification_data):
    x, y = classification_data
    params = [{'shap_importance': False}, {'standardize_imp': True}, {'use_test': False}, {'rough_fix': False}]
    for param_set in params:
        try:
            boruta = eBoruta(verbose=0, n_iter=10, **param_set)
            boruta.fit(x, y)
        except Exception as e:
            assert False, f'Failed on param set {param_set} due to {e}'


def test_multiobjective(classification_data, classifiers_multiobjective):
    x, y = make_classification()
    y = np.array([[_y, _y] for _y in y])
    boruta = eBoruta(n_iter=20, verbose=0)
    for name, model in classifiers_multiobjective:
        try:
            boruta.fit(x, y, model=model)
        except Exception as e:
            assert False, f'Failed to run classifier {name} due to {e}'


if __name__ == '__main__':
    pass
