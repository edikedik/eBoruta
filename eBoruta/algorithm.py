"""
A module containing eBoruta masterclass encapsulating algorithm's execution.
"""
from __future__ import annotations

import logging
import typing as t
from collections import abc
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

from eBoruta.base import _X, _Y, _E, ImportanceGetter, ValidationError
from eBoruta.callbacks import Callback, CallbackReturn
from eBoruta.containers import Dataset, Features, TrialData
from eBoruta.utils import zip_partition

LOGGER = logging.getLogger(__name__)


class eBoruta(BaseEstimator, TransformerMixin):
    """
    Flexible sklearn-compatible feature selection wrapper method.
    """

    def __init__(
        self,
        n_iter: int = 30,
        classification: bool = True,
        percentile: int = 100,
        pvalue: float = 0.05,
        test_size: int | float = 0,
        test_stratify: bool = False,
        shap_tree: bool = True,
        shap_gpu_tree: bool = False,
        shap_approximate: bool = False,
        shap_check_additivity: bool = False,
        importance_getter: ImportanceGetter | None = None,
        verbose: int = 1,
    ):
        """
        :param n_iter: The number of trials to run the algorithm.
        :param classification: `True` if the task is classification else
            `False`.
        :param percentile: Percentile of the shadow features as alternative
            to `max` in original Boruta.
        :param pvalue: Level of rejecting the null hypothesis
            (the absence of a feature's importance).
        :param test_size: The `test_size` param passed to :func:`train_test_split`.
            Can be a number or a fraction.
        :param test_stratify: Stratify the test examples based on the `y` class
            values to balance the split.
        :param shap_tree: Use :class:`shap.Tree` explainer.
        :param shap_gpu_tree: Use :class:`shap.GPUTree` explainer.
        :param shap_approximate: Approximate shap importance values.
            Caution! some estimators may not support it (e.g., CatBoost).
        :param shap_check_additivity: Passed to the explainer.
            Consult with `shap` documentation.
        :param importance_getter: A callable accepting either an estimator or
            an estimator and `TrialData` instance and returning a numpy array
            of length equal to the number of features in `TrialData.x_test`.
        :param verbose: 0 -- no output; 1 -- progress bar; 2 --
            progress bar and info; 3 -- debug mode
        """

        self.n_iter = n_iter
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.test_stratify = test_stratify
        self.test_size = test_size
        self.shap_tree = shap_tree
        self.shap_gpu_tree = shap_gpu_tree
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
            raise ValidationError(
                f"Failed to validate the input parameters due to {e}"
            ) from e
        if self.test_size > 0 and self.test_stratify and not self.classification:
            raise ValidationError(
                'Using "test_stratify" with regressors is not possible'
            )

    @staticmethod
    def _check_model(model):
        has_fit = hasattr(model, "fit")
        has_predict = hasattr(model, "predict")

        if not (has_fit and has_predict):
            raise ValidationError(
                "Model must contain both the fit() and predict() methods"
            )

    @staticmethod
    def _get_importance(model: _E) -> np.ndarray:
        if not hasattr(model, "feature_importances_"):
            raise ValidationError(
                "Builtin importance getter failed due to model not having "
                '"feature_importances_'
            )
        values = model.feature_importances_
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if len(values.shape) == 2:
            values = np.mean(values, axis=0)
        return values

    def _init_model(self, model_type: t.Type[_E] | None) -> _E:
        kw = self.model_init_kwargs_
        if model_type is None:
            if self.classification:
                model = RandomForestClassifier(**kw)
            else:
                model = RandomForestRegressor(**kw)
        else:
            model = model_type(**kw)

        return model

    def calculate_importance(
        self, model: _E, trial_data: TrialData, abs_: bool = True
    ) -> np.ndarray:
        """
        :param model: Estimator with `fit` method. In case of using `shap`
            importance, a tree-based estimator supported by `Tree` explainer
            is expected.
        :param trial_data: Datasets generated for the current trial.
            see :meth:`eBoruta.containers.Dataset.generate_trial_sample`.
        :param abs_: Take absolute value of the importance array. ``True`` by
            default since shap contributions may be negative.
        :return: An array of importance values.
        """
        if self.importance_getter is not None:
            if "trial_data" in signature(self.importance_getter).parameters:
                importance_a = self.importance_getter(model, trial_data)
            else:
                importance_a = self.importance_getter(model)
            LOGGER.debug(f"Got array {importance_a.shape} of builtin importance_a")
        else:
            if self.shap_tree or self.shap_gpu_tree:
                explainer_type = (
                    shap.GPUTreeExplainer if self.shap_gpu_tree else shap.TreeExplainer
                )
                explainer = explainer_type(model)
                values = explainer.shap_values(
                    trial_data.x_test,
                    approximate=self.shap_approximate,
                    check_additivity=self.shap_check_additivity,
                )
                # values is matrix of the same shape as x in case of single
                # objective, and a list of such matrices in case of multi-objective
                # classification/regression.
                # importance per objective is a mean of absolute shap values per
                # feature importance in multiple objectives is a mean of such means
                if isinstance(values, np.ndarray):
                    values = [values]
                for i, v in enumerate(values):
                    if len(v.shape) == 3:
                        values[i] = v.mean(axis=-1)
                importance_a = np.vstack([np.abs(v).mean(0) for v in values]).mean(0)
                LOGGER.debug(
                    f"Calculated {importance_a.shape} importance array using "
                    f"{explainer_type.__name__} explainer"
                )
            else:
                importance_a = self._get_importance(self.model_)

        if abs_:
            importance_a = np.abs(importance_a)

        return importance_a

    def stat_tests(
        self, features: Features, iter_i: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # TODO: should I elaborate docs explaining the calculations here?
        """
        Test features at current iteration based on importance values.

        :param features: Features state at current iteration.
        :param iter_i: Iteration starting from 1.
        :return: Accepted and rejected feature names. Features present in
            :attr:`eBoruta.containers.Features.names` but not accepted or
            rejected are considered "tentative".
        """

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
        self, trial_data: TrialData, callbacks: abc.Sequence[Callback], **kwargs
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
        pbar: tqdm | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
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

        return counts_round, counts_total

    def report_features(
        self, features: Features | None = None, full: bool = False
    ) -> str:
        """
        Create a text report of the optimization progress. It's sane
        to print it out when debugging and using a few features.

        :param features: Current features.
        :param full: Create a very lengthy report for each individual feature.
        :return: Concatenated report messages as a single string.
        """
        if features is None:
            if self.features_ is None:
                raise ValidationError(
                    "Features were not provided and missing `features_` attr"
                )
            features = self.features_
        counts = {
            "accepted": len(self.features_.accepted),
            "rejected": len(self.features_.rejected),
            "tentative": len(self.features_.tentative),
        }
        history = features.history.dropna()
        max_steps = history["Step"].max()
        msg = f"Reporting at max step {max_steps}. Feature counts: {counts}"
        LOGGER.info(msg)
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
                msg_feature = (
                    f'Feature {g} was marked at step {last_step["Step"]} and threshold '
                    f'{round(last_step["Threshold"], 2)} as {last_step["Decision"]}, '
                    f'having {round(last_step["Importance"], 2)} importance '
                    f"({imp_desc}) and total number of hits {total_hits}"
                )
                LOGGER.info(msg_feature)
                msg += f"\n{msg_feature}"
        return msg

    def fit(
        self,
        x: _X,
        y: _Y,
        sample_weight: np.ndarray | None = None,
        model_type: t.Type[_E] | None = None,
        callbacks_trial_start: abc.Sequence[Callback] | None = None,
        callbacks_trial_end: abc.Sequence[Callback] | None = None,
        model_init_kwargs: dict[str, t.Any] | None = None,
        **kwargs,
    ) -> eBoruta:
        """
        Fit the boruta algorithm.

        :param x: Features collection as a 2D array or a ``pd.DataFrame``.
        :param y: Response variable(s).
        :param sample_weight: Optional sample weight for each instance in ``x``
            for models that support it within the ``fit()`` method.
        :param model_type: An uninitialized estimator type with a ``fit(x, y)``
            and ``predict(x)`` methods. If ``None``,
            use :class:`RandomForestClassifier` if :attr:`classification`
            is ``True`` else use :class:`RandomForestRegressor`.
        :param callbacks_trial_start: Callbacks to call at each trial's start.
        :param callbacks_trial_end: Callbacks to call at each trial's end.
        :param model_init_kwargs: Optional keyword arguments to initialize the
            estimator type with. If not provided, it is assumed that the
            estimator can be initialized without any arguments.
        :param kwargs: Passed to ``model.fit()`` method.
        :return: :class:`eBoruta` object.
        """
        self.dataset_ = Dataset(x, y, sample_weight)
        self.features_ = Features(self.dataset_.x.columns.to_numpy())
        self.model_init_kwargs_ = {} if model_init_kwargs is None else model_init_kwargs

        iters = range(1, self.n_iter + 1)
        if self.verbose > 0:
            iters = tqdm(iters, desc="Boruta trials")

        generator_kwargs = {}
        if self.test_size > 0:
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
            self.model_ = self._init_model(model_type)
            self._check_model(self.model_)

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

            imp = self.calculate_importance(self.model_, trial_data)
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
            ), "size of real_imp != size of initial columns"
            LOGGER.debug(
                f"Separated into {len(real_imp)} real and {len(shadow_imp)} "
                f"shadow importance values"
            )

            threshold = np.percentile(shadow_imp, self.percentile)
            LOGGER.debug(
                f"Calculated {self.percentile}-percentile threshold: {threshold}"
            )

            hits = (real_imp > threshold).astype(int)
            hits_total = hits.sum()
            LOGGER.info(
                f"{round(hits_total / len(hits) * 100, 2)}% ({hits_total}) "
                "recorded as hits"
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

            accepted, rejected = self.stat_tests(self.features_, trial_n)
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
            self.report_features(full=self.verbose >= 2)

        return self

    def transform(self, x: _X, tentative: bool = False) -> pd.DataFrame:
        """
        Transform input data using fitted model. Transformation means selecting
        accepted features from ``x``. Note that due to inheriting the from the
        ``TransformerMixin``.

        :param x: Data used to :meth:`fit` the algorithm
        :param tentative: Also select tentative features.
        :return: A subset of ``x``.
        """
        check_is_fitted(self, ["features_", "dataset_", "model_"])
        x = self.dataset_.prepare_x(x)

        if x.shape[1] != self.dataset_.x.shape[1]:
            raise ValueError(
                f"Reshape your data: number of input features {x.shape[1]} does not "
                f"match the one used during fit {self.dataset_.x.shape[1]}"
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

    def rough_fix(self, n_last_trials: int | None = None) -> Features:
        """
        Apply "rough fix" strategy to handle remaining tentative features.

        Features having ``median(importance) > median(thresholds)`` where
        importance and thresholds are history records of tentative features'
        importance and percentile thresholds used to mark "hits" after each
        trial.

        :param n_last_trials: Consider only this number of last trials.
            If ``None``, defaults to all trials.
        :return: Modified features with resolved tentative ones.
        """
        if not hasattr(self, "features_"):
            raise ValueError("Applying rough fix with no recorded features")
        num_tentative = len(self.features_.tentative)
        features = deepcopy(self.features_)
        if not num_tentative:
            LOGGER.info("No tentative features to apply rough fix to")
            return features
        if n_last_trials is None:
            n_last_trials = len(features.imp_history)
        LOGGER.info(
            f'Applying "rough fix" to {num_tentative} tentative features '
            f"using {n_last_trials} last steps"
        )
        tentative_median = features.imp_history.iloc[:n_last_trials][
            features.tentative
        ].median()
        threshold_median = features.imp_history.iloc[:n_last_trials][
            "Threshold"
        ].median()
        LOGGER.info(
            "Median importance for tentative features throughout "
            f"history: {tentative_median.values}"
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

    def rank(
        self,
        features: abc.Sequence[str] | np.ndarray | None = None,
        step: int | None = None,
        fit: bool = True,
        model: _E | None = None,
        gen_sample: bool = False,
        sort: bool = False,
    ) -> pd.DataFrame:
        """
        Rank (sort) features by feature importance values.

        Uses :meth:`calculate_importance` to obtain importance of selected
        features and :attr:`dataset_` to obtain the data.

        :param features: A sequence of features to select.
        :param step: A step (trial) number. If provided, will the method will
            select accepted features at this trial. If `features` were provided,
            will intersect with this list.
        :param fit: Fit the model before calculating importance. In most cases,
            this should be ``True``, since otherwise the features used to fit
            the :attr:`model_` would be different from the features being
            ranked (for which the :attr:`model_` will be queried in order to
            calculate the importance values).
        :param model: Use a prefit model instead of the stored :attr:`model_`.
        :param gen_sample: Generate trial sample from the :attr:`dataset_` using
            :attr:`test_size` and :attr:`stratify` values provided during init.
        :param sort: Sort results by importance values in descending order.
        :return: A DataFrame with `Feature` and `Importance` columns.
        """

        if len(self.features_) == 0:
            raise ValidationError("No steps accumulated yet")

        if self.dataset_ is None:
            raise ValidationError("Missing dataset")

        if features is None:
            if step is None:
                features = list(self.features_.names)
            else:
                if step < 1:
                    raise ValidationError(f"Negative step {step}")
                if step >= len(self.features_):
                    LOGGER.warning(
                        f"Step {step} exceeds max iteration {len(self.features_)}"
                    )
                features = list(self.features_.accepted_at_step(step))

        if len(features) == 0:
            return pd.DataFrame(columns=["Feature", "Importance"])

        if gen_sample:
            generator_kwargs = {}
            if self.test_size > 0:
                generator_kwargs["test_size"] = self.test_size
            trial_data = self.dataset_.generate_trial_sample(features)
        else:
            x = self.dataset_.x[features]
            y = self.dataset_.y
            trial_data = TrialData(x, x, y, y)

        if model is None:
            if fit:
                self.model_.fit(
                    trial_data.x_train,
                    trial_data.y_train,
                    sample_weight=trial_data.w_train,
                )
            model = self.model_

        imp = self.calculate_importance(model, trial_data)

        df = pd.DataFrame({"Feature": trial_data.x_test.columns, "Importance": imp})
        if sort:
            return df.sort_values("Importance", ascending=False)
        return df


if __name__ == "__main__":
    raise RuntimeError
