import typing as t

import numpy as np
import pandas as pd

_X = t.TypeVar('_X', pd.DataFrame, np.ndarray)
_Y = t.TypeVar('_Y', pd.DataFrame, pd.Series, np.ndarray)
_W = t.TypeVar('_W', pd.Series, np.ndarray)
_E = t.TypeVar('_E', 'RandomForestClassifier', 'XGBClassifier', 'CatBoostClassifier',
               'RandomForestRegressor', 'XGBRegressor', 'CatBoostRegressor')

if __name__ == '__main__':
    raise RuntimeError
