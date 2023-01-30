import logging
import typing as t
from copy import deepcopy
from inspect import signature

import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from eBoruta.base import _X, _Y, _E
from eBoruta.callbacks import Callback, CallbackReturn
from eBoruta.containers import Dataset, Features, TrialData, ImportanceGetter
from eBoruta.utils import zip_partition

LOGGER = logging.getLogger(__name__)


class eBoruta(BaseEstimator, TransformerMixin):
    """
    Flexible sklearn-compatible feature selection wrapper method.
    """

    def __init__(
        self,
        n_iter: int = 20,
        classification: bool = True,
        percentile: int = 100,
        pvalue: float = 0.05,
        use_test: bool = True,
        test_size: t.Union[float, int] = 0.3,
        test_stratify: t.Optional[bool] = True,
        shap_importance: bool = True,
        shap_use_gpu: bool = False,
        shap_approximate: bool = True,
        shap_check_additivity: bool = False,
        importance_getter: t.Optional[ImportanceGetter] = None,
        standardize_imp: bool = False,
        verbose: int = 1,
    ):
        """
        :param n_iter: The number of trials to run the algorithm.
        :param classification: `True` if the task is classification else `False`.
        :param percentile: Percentile of the shadow features as alternative to `max` in original Boruta.
        :param pvalue: Level of rejecting the null hypothesis (the absence of a feature's importance).
        :param use_test: Use test set instead of the full training set to compute feature importances.
        :param test_size: The `test_size` paramter passed to `train_test_split`. Can be a number or a fraction.
        :param test_stratify: Stratify the test examples based on the `y` class values to balance the split.
        :param shap_importance: Use importance values based on `shap` package, specifically the `Tree` explainer.
        :param shap_use_gpu: Use `GPUTree` explainer.
        :param shap_approximate: Approximate shap importance values. Caution! some estimators may not support it.
        :param shap_check_additivity: Passed to the explainer. Consult with `shap` documentation.
        :param importance_getter: A callable accepting either an estimator or an estimator and `TrialData` instance
            and returning a numpy array of length equal to the number of features in `TrialData.x_test`.
        :param standardize_imp: Standardize importance values.
        :param verbose: 0 -- no output; 1 -- progress bar; 2 -- progress bar and info; 3 -- debug mode
        """

        self.n_iter = n_iter
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.test_stratify = test_stratify
        self.use_test = use_test
        self.test_size = test_size
        self.standardize_imp = standardize_imp
        self.shap_importance = shap_importance
        self.shap_use_gpu = shap_use_gpu
        self.shap_approximate = shap_approximate
        self.shap_check_additivity = shap_check_additivity
        self.importance_getter = importance_getter

        if verbose >= 3:
            LOGGER.setLevel(logging.DEBUG)
        elif verbose == 2:
            LOGGER.setLevel(logging.INFO)
        elif verbose == 1:
            LOGGER.setLevel(logging.WARNING)
        elif verbose <= 0:
            LOGGER.setLevel(logging.ERROR)
        self.verbose = verbose

        self._check_params()

    def _check_params(self):
        try:
            assert self.n_iter >= 1
            assert 0 < self.percentile <= 100
            assert 0 < self.pvalue < 1
        except AssertionError as e:
            raise AttributeError(f"Failed to validate the input parameters due to {e}")
        if self.use_test and self.test_stratify and not self.classification:
            raise AttributeError(
                f'Using "test_stratify" with regressors is not possible'
            )

    @staticmethod
    def _check_model(model):
        has_fit = hasattr(model, "fit")
        has_predict = hasattr(model, "predict")

        if not (has_fit and has_predict):
            raise AttributeError(
                "Model must contain both the fit() and predict() methods"
            )

    @staticmethod
    def _get_importance(model: _E) -> np.ndarray:
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                'Builtin importance getter failed due to model not having "feature_importances_'
            )
        values = model.feature_importances_
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if len(values.shape) == 2:
            values = np.mean(values, axis=0)
        return values

    def importance(self, model: _E, trial_data: TrialData, abs_: bool = True):
        """

        :param model: Estimator with `fit` method. In case of using `shap`
            importance, a tree-based estimator supported by `Tree` explainer
            is expected.
        :param trial_data:
        :param abs_:
        :return:
        """
        if self.shap_importance:
            explainer_type = (
                shap.explainers.GPUTree if self.shap_use_gpu else shap.explainers.Tree
            )
            explainer = explainer_type(model)
            values = explainer.shap_values(
                trial_data.x_test,
                approximate=self.shap_approximate,
                check_additivity=self.shap_check_additivity,
            )
            # values is matrix of the same shape as x in case of single objective,
            # and a list of such matrices in case of multi-objective classification/regression
            # importance per objective is a mean of absolute shap values per feature
            # importance in multiple objectives is a mean of such means
            if isinstance(values, np.ndarray):
                values = [values]
            importances = np.vstack([np.abs(v).mean(0) for v in values]).mean(0)
            LOGGER.debug(
                f"Calculated {importances.shape} importances using {explainer_type.__name__} explainer"
            )
        else:
            if self.importance_getter is None:
                importances = self._get_importance(self.model_)
            else:
                if "trial_data" in signature(self.importance_getter).parameters:
                    importances = self.importance_getter(model, trial_data)
                else:
                    importances = self.importance_getter(model)
            LOGGER.debug(f"Got array {importances.shape} of builtin importances")

        if self.standardize_imp:
            importances = (importances - importances.mean()) / importances.std()

        if abs_:
            importances = np.abs(importances)

        return importances

    def _stat_tests(
        self, features: Features, iter_i: int
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        def _test_hits(a: np.ndarray, alternative: str) -> np.ndarray:
            p_val = np.array(
                [binomtest(x, iter_i, alternative=alternative).pvalue for x in a]
            )
            passed_fdr, p_val_corr = fdrcorrection(p_val)
            passed_bon = p_val <= self.pvalue / iter_i
            return (passed_fdr * passed_bon).astype(bool)

        hits_total = features.hit_history[features.tentative].sum().values
        accepted = _test_hits(hits_total, "greater")
        rejected = _test_hits(hits_total, "less")

        return accepted, rejected

    def _call_callbacks(
        self, trial_data: TrialData, callbacks: t.Sequence[Callback], **kwargs
    ) -> CallbackReturn:
        for c in callbacks:
            LOGGER.debug(f"Running callback {c}")
            self.model_, self.features_, self.dataset_, trial_data, kwargs = c(
                self.model_, self.features_, self.dataset_, trial_data
            )
        return self.model_, self.features_, self.dataset_, trial_data, kwargs

    @staticmethod
    def _report_trial(
        features: Features,
        accepted: np.ndarray,
        rejected: np.ndarray,
        tentative: np.ndarray,
        pbar: t.Optional[tqdm] = None,
    ):
        names = features.names[tentative]
        accepted_names = names[accepted]
        rejected_names = names[rejected]
        tentative_names = names[~accepted & ~rejected]
        round_tentative_initial = tentative.sum()
        report_round = dict(
            zip(
                ["accepted", "rejected", "tentative"],
                [accepted_names, rejected_names, tentative_names],
            )
        )
        counts_round = {k: len(v) for k, v in report_round.items()}
        report_total = {
            "accepted": features.accepted,
            "rejected": features.rejected,
            "tentative": features.tentative,
        }
        counts_total = {k: len(v) for k, v in report_total.items()}

        LOGGER.info(f"Out of {round_tentative_initial}: {counts_round}")
        LOGGER.info(f"Total summary: {counts_total}")

        if pbar is not None:
            pbar.set_postfix(counts_total)

        for k, v in report_round.items():
            LOGGER.debug(f"{k}: {v}")

    def report_features(
        self, features: t.Optional[Features] = None, full: bool = False
    ):
        counts = {
            "accepted": len(self.features_.accepted),
            "rejected": len(self.features_.rejected),
            "tentative": len(self.features_.tentative),
        }
        features = self.features_ if features is None else features
        history = features.history.dropna()
        max_steps = history["Step"].max()
        LOGGER.info(f"Stopped at {max_steps} step. Final results: {counts}")
        if full:
            for g, gg in history.groupby("Feature"):
                total_hits = gg["Hit"].sum()
                imp = gg["Importance"]
                imp_desc = {
                    "min": imp.min(),
                    "max": imp.max(),
                    "median": imp.median(),
                    "mean": imp.mean(),
                    "std": imp.std(),
                }
                imp_desc = {k: round(v, 2) for k, v in imp_desc.items()}
                last_step = gg.iloc[-1]
                LOGGER.info(
                    f'Feature {g} was marked at step {last_step["Step"]} and threshold '
                    f'{round(last_step["Threshold"], 2)} as {last_step["Decision"]}, having '
                    f'{round(last_step["Importance"], 2)} importance ({imp_desc}) '
                    f"and total number of hits {total_hits}"
                )

    def _fit(
        self,
        x: pd.DataFrame,
        y: _Y,
        sample_weight: t.Optional[np.ndarray] = None,
        model: t.Any = None,
        callbacks_trial_start: t.Optional[t.Sequence[Callback]] = None,
        callbacks_trial_end: t.Optional[t.Sequence[Callback]] = None,
        **kwargs,
    ) -> "eBoruta":
        self.dataset_ = Dataset(x, y, sample_weight)
        self.features_ = Features(self.dataset_.x.columns.to_numpy())
        if model is None:
            if self.classification:
                self.model_ = RandomForestClassifier()
            else:
                self.model_ = RandomForestRegressor()
        else:
            self.model_ = model

        self._check_model(self.model_)

        iters = range(1, self.n_iter + 1)
        if self.verbose > 0:
            iters = tqdm(iters, desc="Boruta trials")

        generator_kwargs = {}
        if self.use_test:
            generator_kwargs["test_size"] = self.test_size
            # TODO: test how it works in case of multiple objectives
            generator_kwargs["stratify"] = self.dataset_.y.copy()

        for trial_n in iters:
            trial_data = self.dataset_.generate_trial_sample(
                columns=self.features_.tentative, **generator_kwargs
            )
            LOGGER.info(
                f"Trial {trial_n}: sampled trial data with shapes {trial_data.shapes}"
            )

            if callbacks_trial_start is not None:
                (
                    self.model_,
                    self.features_,
                    self.dataset_,
                    trial_data,
                    kwargs,
                ) = self._call_callbacks(trial_data, callbacks_trial_start, **kwargs)

            self.model_.fit(
                trial_data.x_train,
                trial_data.y_train,
                sample_weight=trial_data.w_train,
                **kwargs,
            )
            LOGGER.debug("Fitted the model")

            imp = self.importance(self.model_, trial_data)
            LOGGER.debug(f"Calculated {len(imp)} importance values")

            real_imp, shadow_imp = map(
                list,
                zip_partition(
                    lambda _x: isinstance(_x, str) and "shadow" in _x,
                    imp,
                    trial_data.x_test.columns,
                ),
            )
            assert len(real_imp) == len(
                self.features_.tentative
            ), "size of real_imp == size of initital columns"
            LOGGER.debug(
                f"Separated into {len(real_imp)} real and {len(shadow_imp)} shadow importance values"
            )

            threshold = np.percentile(shadow_imp, self.percentile)
            LOGGER.debug(
                f"Calculated {self.percentile}-percentile threshold: {threshold}"
            )

            hits = (real_imp > threshold).astype(np.int)
            hits_total = hits.sum()
            LOGGER.info(
                f"{round(hits_total / len(hits) * 100, 2)}% ({hits_total}) recorded as hits"
            )
            hit_upd = dict(zip(self.features_.tentative, hits))
            imp_upd = dict(zip(self.features_.tentative, real_imp))
            imp_upd["Threshold"] = threshold
            self.features_.imp_history = pd.concat(
                [self.features_.imp_history, pd.DataFrame.from_records([imp_upd])]
            )
            self.features_.hit_history = pd.concat(
                [self.features_.hit_history, pd.DataFrame.from_records([hit_upd])]
            )

            accepted, rejected = self._stat_tests(self.features_, trial_n)
            initial_tentative = self.features_.tentative_mask

            self.features_.accepted_mask[self.features_.tentative_mask] = accepted
            self.features_.rejected_mask[self.features_.tentative_mask] = rejected
            self.features_.tentative_mask = (
                ~self.features_.accepted_mask & ~self.features_.rejected_mask
            )
            decisions = np.zeros(len(self.features_.names), dtype=int)
            decisions[self.features_.accepted_mask] = 1
            decisions[self.features_.rejected_mask] = -1
            dec_upd = dict(zip(self.features_.names, decisions))
            self.features_.dec_history = pd.concat(
                [self.features_.dec_history, pd.DataFrame.from_records([dec_upd])]
            )

            if callbacks_trial_end is not None:
                (
                    self.model_,
                    self.features_,
                    self.dataset_,
                    trial_data,
                    kwargs,
                ) = self._call_callbacks(trial_data, callbacks_trial_end)

            self._report_trial(
                self.features_,
                accepted,
                rejected,
                initial_tentative,
                iters if isinstance(iters, tqdm) else None,
            )

            if not self.features_.tentative_mask.sum():
                LOGGER.info(f"No tentative features left, stopping at trial {trial_n}")
                break

        if self.verbose > 0:
            self.report_features(full=self.verbose == 2)

        return self

    def _transform(self, x: _X, tentative: bool = False) -> pd.DataFrame:
        check_is_fitted(self, ["features_", "dataset_", "model_"])
        x = self.dataset_.convert_x(x)

        if x.shape[1] != self.dataset_.x.shape[1]:
            raise ValueError(
                f"Reshape your data: number of input features {x.shape[1]} does not match "
                f"the one used during fit {self.dataset_.x.shape[1]}"
            )

        inp_unique = set(x.columns) - set(self.dataset_.x.columns)
        ds_unique = set(self.dataset_.x.columns) - set(x.columns)
        if len(inp_unique) != 0:
            raise ValueError(
                f'Features {inp_unique} were absent in the dataset used during "fit"'
            )
        if len(ds_unique) != 0:
            raise ValueError(f"Features {ds_unique} are missing in the input dataset")

        sel_columns = list(self.features_.accepted)
        if tentative:
            sel_columns += list(self.features_.tentative)

        return x[sel_columns]

    def fit(
        self,
        x: _X,
        y: _Y,
        sample_weight=None,
        model: t.Any = None,
        callbacks_trial_start: t.Optional[t.Sequence[Callback]] = None,
        callbacks_trial_end: t.Optional[t.Sequence[Callback]] = None,
        **kwargs,
    ) -> "eBoruta":
        return self._fit(
            x,
            y,
            sample_weight=sample_weight,
            model=model,
            callbacks_trial_start=callbacks_trial_start,
            callbacks_trial_end=callbacks_trial_end,
            **kwargs,
        )

    def transform(self, x: _X, tentative: bool = False) -> pd.DataFrame:
        return self._transform(x, tentative)

    def rough_fix(self, n_last_steps: t.Optional[int] = None) -> Features:
        if not hasattr(self, "features_"):
            raise ValueError("Applying rough fix with no recorded features")
        num_tentative = len(self.features_.tentative)
        features = deepcopy(self.features_)
        if not num_tentative:
            LOGGER.info("No tentative features to apply rough fix to")
            return features
        if n_last_steps is None:
            n_last_steps = len(features.imp_history)
        LOGGER.info(
            f'Applying "rough fix" to {num_tentative} tentative features using {n_last_steps} last steps'
        )
        tentative_median = features.imp_history.iloc[:n_last_steps][
            features.tentative
        ].median()
        threshold_median = features.imp_history.iloc[:n_last_steps][
            "Threshold"
        ].median()
        LOGGER.info(
            f"Median importance for tentative features throughout history: {tentative_median.values}"
        )
        LOGGER.info(
            f"Median {self.percentile}-th percentile threshold: {threshold_median}"
        )
        rejected = np.less(tentative_median, threshold_median)
        accepted_names = features.names[features.tentative_mask][~rejected]
        rejected_names = features.names[features.tentative_mask][rejected]
        LOGGER.info(
            f"Accepted {(~rejected).sum()} ({accepted_names}) feature(s). "
            f"Rejected {rejected.sum()} ({rejected_names}) feature(s)"
        )
        features.accepted_mask[features.tentative_mask] = ~rejected
        features.rejected_mask[features.tentative_mask] = rejected
        features.tentative_mask[features.tentative_mask] = False
        return features


if __name__ == "__main__":
    raise RuntimeError
