# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D400
"""
# RandomForest

This module can be used for training and using a Random Forest Regression model.

A pyrfr wrapper is used for simplification.

## Classes
    - RandomForest: A random forest wrapper for pyrfr.

## Constants
    VERY_SMALL_NUMBER : float
    PYRFR_MAPPING : Dict[str, str]
"""

from typing import Any, Dict, Optional, Tuple, Union

import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.decomposition import PCA

# import pyrfr.regression as regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from deepcave.evaluators.epm.utils import get_types

VERY_SMALL_NUMBER = 1e-10
RFR_MAPPING = {
    "n_trees": "n_estimators",
    "bootstrapping": "bootstrap",
    "max_features": "max_features",
    "min_samples_split": "min_samples_split",
    "min_samples_leaf": "min_samples_leaf",
    "max_depth": "max_depth",
}


class RandomForest:
    """
    A random forest wrapper for pyrfr.

    This is handy because only the configuration space needs to be passed.
    and have a working version without specifying e.g. types and bounds.

    Note
    ----
    This wrapper also supports instances.

    Properties
    ----------
    cs : ConfigurationSpace
        The configuration space.
    log_y : bool
        Whether y should be treated as a logarithmic transformation.
    seed : int
        The seed. If not provided, it is random.
    types : List[int]
        The types of the Hyperparameters.
    bounds : List[Tuple[float, float]]
        The bounds of the Hyperparameters.
    n_params : int
        The number of Hyperparameters in the configuration space.
    n_features : int
        The number of features.
    pca_components : int
        The number of components to keep for the principal component analysis (PCA).
    pca : PCA
        The principal component analysis (PCA) object.
    scaler : MinMaxScaler
        A MinMaxScaler to scale the features.
    instance_features : ndarray
        The instance features.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_trees: int = 16,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        bootstrapping: bool = True,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = 2,
        log_y: bool = False,
        seed: Optional[int] = 0,
    ):
        self.cs = configspace
        self.log_y = log_y
        self.seed = seed

        # Set types and bounds automatically
        types, self.bounds = get_types(configspace, instance_features)
        self.types = np.array(types)

        # Prepare everything for PCA
        self.n_params = len(list(configspace.values()))
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
            bootstrapping=bootstrapping,
        )

    def _get_model(self) -> RandomForestRegressor:
        """
        Return the internal model.

        Returns
        -------
        model : regression.base_tree
            Model which is used internally.
        """
        return RandomForestRegressor()

    def _get_model_options(self, **kwargs: Union[int, float, bool]) -> Dict[str, Any]:
        """
        Get model options from kwargs.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            The key word arguments for the model options.

        Returns
        -------
        options : regression.forest_opts
            Random forest options.
        """
        # Now the options are set
        options = self._model.get_params()

        for k, v in kwargs.items():
            new_k = RFR_MAPPING[k]

            # Handle nested keys like "model.learning_rate"
            keys = new_k.split(".")
            d = options
            for key in keys[:-1]:
                d = d.setdefault(key, {})  # drill down, create dict if missing
            d[keys[-1]] = v

        return options

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        """
        Impute inactive values in X.

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
            If Hyperparameter is not supported.
        """
        conditional: Dict[int, bool] = {}
        impute_values: Dict[int, float] = {}

        X = X.copy()
        for idx, hp in enumerate(list(self.cs.values())):
            if idx not in conditional:
                parents = self.cs.parents_of[hp.name]
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
        Check if the dimensions of X and Y are correct with respect to features.

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

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the random forest on X and Y.

        Transform X if principal component analysis (PCA) is applied.
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
        Train the random forest on X and Y.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        Y : np.ndarray
            Target values.
        """
        self._model.fit(X, Y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        vars : np.ndarray [n_samples, n_objectives] or [n_samples, n_samples]
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

    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict means and variances for a given X.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config + instance features)]

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
            all_tree_preds = np.array([tree.predict(X) for tree in self._model.estimators_])

            means = all_tree_preds.mean(axis=1)
            vars_ = all_tree_preds.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                all_tree_preds = np.array(
                    [tree.predict([row_X]) for tree in self._model.estimators_]
                )

                mean_ = all_tree_preds.mean(axis=0)
                var = all_tree_preds.var(axis=0)

                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance marginalized over all instances.

        Return the predictive mean and variance marginalized over all
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
        # Mean per tree across instances
        dat_ = np.array([tree.predict(X) for tree in self._model.estimators_])  # shape: (n_trees,)

        # 3. compute statistics across trees
        mean_ = dat_.mean(axis=1)
        var = dat_.var(axis=1)

        var[var < VERY_SMALL_NUMBER] = VERY_SMALL_NUMBER

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var
