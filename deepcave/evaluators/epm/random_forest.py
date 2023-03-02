from typing import Dict, Optional, Tuple

import functools
import warnings
from random import random

import numpy as np
import pyrfr.regression as regression
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from deepcave.evaluators.epm.utils import get_types

VERY_SMALL_NUMBER = 1e-10
PYRFR_MAPPING = {
    "n_trees": "num_trees",
    "bootstrapping": "do_bootstrapping",
    "max_features": "tree_opts.max_features",
    "min_samples_split": "tree_opts.min_samples_to_split",
    "min_samples_leaf": "tree_opts.min_samples_in_leaf",
    "max_depth": "tree_opts.max_depth",
    "eps_purity": "tree_opts.epsilon_purity",
    "max_nodes": "tree_opts.max_num_nodes",
}


class RandomForest:
    """
    A random forest wrapper for pyrfr. This is handy because we only need to pass the configspace
    and have a working version without specifying e.g. types and bounds.

    Note
    ----
    This wrapper also supports instances.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_trees: int = 16,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        max_nodes: int = 2**20,
        eps_purity: float = 1e-8,
        bootstrapping: bool = True,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = 2,
        log_y: bool = False,
        seed: Optional[int] = None,
    ):
        self.cs = configspace
        self.log_y = log_y
        self.seed = seed

        # Set types and bounds automatically
        self.types, self.bounds = get_types(configspace, instance_features)

        # Prepare everything for PCA
        self.n_params = len(configspace.get_hyperparameters())
        self.n_features = 0
        if instance_features is not None:
            self.n_features = instance_features.shape[1]

        self._pca_applied = False
        self.pca_components = pca_components
        self.pca = PCA(n_components=self.pca_components)
        self.scaler = MinMaxScaler()
        self.instance_features = instance_features

        # Calculate max number of features
        max_features = max(1, int(len(self.types) * ratio_features)) if ratio_features <= 1.0 else 0

        # Prepare the model
        self._model = self._get_model()
        self._model.options = self._get_model_options(
            n_trees=n_trees,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_nodes=max_nodes,
            eps_purity=eps_purity,
            bootstrapping=bootstrapping,
        )

    def _get_model(self) -> regression.base_tree:
        """
        Returns the internal model.

        Returns
        -------
        model : regression.base_tree
            Model which is used internally.
        """
        return regression.binary_rss_forest()

    def _get_model_options(self, **kwargs) -> regression.forest_opts:
        """
        Get model options from kwargs. The mapping `PYRFR_MAPPING` is used in combination with
        a recursive attribute setter to set the options for the pyrfr model.

        Returns
        -------
        options : regression.forest_opts
            Random forest options.
        """
        # Now we set the options
        options = regression.forest_opts()

        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)

            return functools.reduce(_getattr, [obj] + attr.split("."))

        def rsetattr(obj, attr, val):
            pre, _, post = attr.rpartition(".")
            return setattr(rgetattr(obj, pre) if pre else obj, post, val)

        for k, v in kwargs.items():
            new_k = PYRFR_MAPPING[k]
            rsetattr(options, new_k, v)

        return options

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        """
        Imputs inactive values in X.

        Parameters
        ----------
        X : np.ndarray
            Data points.

        Returns
        -------
        np.ndarray
            Imputed data points.

        Raises
        ------
        ValueError
            If hyperparameter is not supported.
        """
        conditional = {}  # type: Dict[int, bool]
        impute_values = {}  # type: Dict[int, float]

        X = X.copy()
        for idx, hp in enumerate(self.cs.get_hyperparameters()):
            if idx not in conditional:
                parents = self.cs.get_parents_of(hp.name)
                if len(parents) == 0:
                    conditional[idx] = False
                else:
                    conditional[idx] = True
                    if isinstance(hp, CategoricalHyperparameter):
                        impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        impute_values[idx] = 1
                    else:
                        raise ValueError

            if conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = impute_values[idx]

        return X

    def _check_dimensions(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
        """
        Checks if the dimensions of X and Y are correct wrt features.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        Y : Optional[np.ndarray], optional
            Target values. By default None.

        Raises
        ------
        ValueError
            If any dimension of X or Y is incorrect or unsuitable.
        """
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2d array, got {len(X.shape)}d array.")

        if X.shape[1] != self.n_params + self.n_features:
            raise ValueError(
                f"Feature mismatch: X should have {self.n_params} features, but has {X.shape[1]}"
            )

        if Y is not None:
            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"X.shape[0] ({X.shape[0]}) != y.shape[0] ({Y.shape[0]})")

    def _get_data_container(
        self, X: np.ndarray, y: np.ndarray
    ) -> regression.default_data_container:
        """
        Fills a pyrfr default data container s.t. the forest knows
        categoricals and bounds for continuous data.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points.
        y : np.ndarray [n_samples, ]
            Target values.

        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """

        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        return data

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the random forest on X and Y. Transforms X if PCA is applied.
        Afterwards, `_train` is called.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            Target values. `n_objectives` must match the number of target names specified in
            the constructor.
        """
        self._check_dimensions(X, Y)

        # Reduce dimensionality of features of larger than PCA_DIM
        self._pca_applied = False
        if (
            self.pca_components
            and X.shape[0] > self.pca.n_components
            and self.n_features >= self.pca_components
        ):
            X_features = X[:, -self.n_features :]

            # Scale features
            X_features = self.scaler.fit_transform(X_features)
            X_features = np.nan_to_num(X_features)  # if features with max == min

            # PCA
            X_features = self.pca.fit_transform(X_features)
            X = np.hstack((X[:, : self.n_params], X_features))

            # Adopt types
            self.types = np.array(
                np.hstack((self.types[: self.n_params], np.zeros((X_features.shape[1])))),
                dtype=np.uint,
            )
            self._pca_applied = True

        self._train(X, Y)

    def _train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the random forest on X and Y.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        Y : np.ndarray
            Target values.
        """
        # Now we can start to prepare the data for the pyrfr
        data = self._get_data_container(X, Y.flatten())
        seed = self.seed
        if seed is None:
            seed = int(random() * 9999)

        rng = regression.default_random_engine(seed)

        # Set more specific model options and finally fit it
        self._model.options.num_data_points_per_tree = X.shape[0]
        self._model.fit(data, rng=rng)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict means and variances for a given X.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Training samples.

        Returns
        -------
        means : np.ndarray [n_samples, n_objectives]
            Predictive mean.
        vars : Optional[np.ndarray] [n_samples, n_objectives] or [n_samples, n_samples]
            Predictive variance or standard deviation.
        """
        self._check_dimensions(X)

        if self._pca_applied:
            try:
                X_features = X[:, -self.n_features :]
                X_features = self.scaler.transform(X_features)
                X_features = self.pca.transform(X_features)
                X = np.hstack((X[:, : self.n_params], X_features))
            except NotFittedError:
                pass  # PCA not fitted if only one training sample

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Predicted variances are smaller than 0. Setting those variances to 0."
            )
            mean, var = self._predict(X)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))

        if var is not None and len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict means and variances for a given X.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray [n_samples, 1]
            Predictive mean.
        vars : np.ndarray [n_samples, 1]
            Predictive variance.
        """

        self._check_dimensions(X)
        X = self._impute_inactive(X)

        if self.log_y:
            all_preds = []
            third_dimension = 0

            # Gather data in a list of 2d arrays and get statistics about the required size of the 3d array
            for row_X in X:
                preds_per_tree = self._model.all_leaf_values(row_X)
                all_preds.append(preds_per_tree)
                max_num_leaf_data = max(map(len, preds_per_tree))
                third_dimension = max(max_num_leaf_data, third_dimension)

            # Transform list of 2d arrays into a 3d array
            preds_as_array = (
                np.zeros((X.shape[0], self._model_options.num_trees, third_dimension)) * np.NaN
            )
            for i, preds_per_tree in enumerate(all_preds):
                for j, pred in enumerate(preds_per_tree):
                    preds_as_array[i, j, : len(pred)] = pred

            # Do all necessary computation with vectorized functions
            preds_as_array = np.log(np.nanmean(np.exp(preds_as_array), axis=2) + VERY_SMALL_NUMBER)

            # Compute the mean and the variance across the different trees
            means = preds_as_array.mean(axis=1)
            vars_ = preds_as_array.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                mean_, var = self._model.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance marginalized over all instances.
        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        self._check_dimensions(X)

        if self.instance_features is None or len(self.instance_features) == 0:
            mean_, var = self.predict(X)
            assert var is not None  # please mypy

            var[var < VERY_SMALL_NUMBER] = VERY_SMALL_NUMBER
            var[np.isnan(var)] = VERY_SMALL_NUMBER
            return mean_, var

        X = self._impute_inactive(X)

        # marginalized predictions for each tree
        dat_ = np.zeros((X.shape[0], self._model_options.num_trees))
        for i, x in enumerate(X):

            # marginalize over instances
            # 1. get all leaf values for each tree
            # type: list[list[float]]
            preds_trees = [[] for i in range(self._model_options.num_trees)]

            for feat in self.instance_features:
                x_ = np.concatenate([x, feat])
                preds_per_tree = self._model.all_leaf_values(x_)
                for tree_id, preds in enumerate(preds_per_tree):
                    preds_trees[tree_id] += preds

            # 2. average in each tree
            if self.log_y:
                for tree_id in range(self._model_options.num_trees):
                    dat_[i, tree_id] = np.log(np.exp(np.array(preds_trees[tree_id])).mean())
            else:
                for tree_id in range(self._model_options.num_trees):
                    dat_[i, tree_id] = np.array(preds_trees[tree_id]).mean()

        # 3. compute statistics across trees
        mean_ = dat_.mean(axis=1)
        var = dat_.var(axis=1)

        var[var < VERY_SMALL_NUMBER] = VERY_SMALL_NUMBER

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var

    def get_leaf_values(self, x: np.ndarray):
        return self._model.all_leaf_values(x)
