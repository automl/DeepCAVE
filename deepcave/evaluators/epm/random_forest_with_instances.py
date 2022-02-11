from typing import Optional

import numpy as np
import pyrfr.regression as regression
from smac.configspace import ConfigurationSpace
from smac.utils.constants import N_TREES, VERY_SMALL_NUMBER

from deepcave.evaluators.epm.forest import Forest


class RandomForestWithInstances(Forest):
    """
    Simple wrapper to calculate types and bounds automatically.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        seed: int,
        log_y: bool = False,
        num_trees: int = N_TREES,
        bootstrapping: bool = True,
        points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_num_nodes: int = 2**20,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ):

        super().__init__(configspace, seed, instance_features, pca_components)

        max_features = 0
        if ratio_features <= 1.0:
            max_features = max(1, int(len(self.types) * ratio_features))

        self._set_model_options(
            {
                "num_trees": num_trees,
                "do_bootstrapping": bootstrapping,
                "tree_opts.max_features": max_features,
                "tree_opts.min_samples_to_split": min_samples_split,
                "tree_opts.min_samples_in_leaf": min_samples_leaf,
                "tree_opts.max_depth": max_depth,
                "tree_opts.epsilon_purity": eps_purity,
                "tree_opts.max_num_nodes": max_num_nodes,
                "compute_law_of_total_variance": False,
            }
        )

        self.log_y = log_y
        self.points_per_tree = points_per_tree

    def _get_model(self):
        return regression.binary_rss_forest()

    def _predict(
        self, X: np.ndarray, cov_return_type: Optional[str] = "diagonal_cov"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.
        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.
        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
        if X.shape[1] != len(self.types):
            raise ValueError(
                "Rows in X should have %d entries but have %d!"
                % (len(self.types), X.shape[1])
            )
        if cov_return_type != "diagonal_cov":
            raise ValueError(
                "'cov_return_type' can only take 'diagonal_cov' for this model"
            )

        X = self._impute_inactive(X)

        if self.log_y:
            all_preds = []
            third_dimension = 0

            # Gather data in a list of 2d arrays and get statistics about the required size of the 3d array
            for row_X in X:
                preds_per_tree = self.forest.all_leaf_values(row_X)
                all_preds.append(preds_per_tree)
                max_num_leaf_data = max(map(len, preds_per_tree))
                third_dimension = max(max_num_leaf_data, third_dimension)

            # Transform list of 2d arrays into a 3d array
            preds_as_array = (
                np.zeros((X.shape[0], self.forest_options.num_trees, third_dimension))
                * np.NaN
            )
            for i, preds_per_tree in enumerate(all_preds):
                for j, pred in enumerate(preds_per_tree):
                    preds_as_array[i, j, : len(pred)] = pred

            # Do all necessary computation with vectorized functions
            preds_as_array = np.log(
                np.nanmean(np.exp(preds_as_array), axis=2) + VERY_SMALL_NUMBER
            )

            # Compute the mean and the variance across the different trees
            means = preds_as_array.mean(axis=1)
            vars_ = preds_as_array.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                mean_, var = self.forest.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized_over_instances(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.
        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.
        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates
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

        if self.instance_features is None or len(self.instance_features) == 0:
            mean_, var = self.predict(X)
            assert var is not None  # please mypy

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean_, var

        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError(
                "Rows in X should have %d entries but have %d!"
                % (len(self.bounds), X.shape[1])
            )

        X = self._impute_inactive(X)

        # marginalized predictions for each tree
        dat_ = np.zeros((X.shape[0], self.forest_options.num_trees))
        for i, x in enumerate(X):

            # marginalize over instances
            # 1. get all leaf values for each tree
            # type: list[list[float]]
            preds_trees = [[] for i in range(self.forest_options.num_trees)]

            for feat in self.instance_features:
                x_ = np.concatenate([x, feat])
                preds_per_tree = self.forest.all_leaf_values(x_)
                for tree_id, preds in enumerate(preds_per_tree):
                    preds_trees[tree_id] += preds

            # 2. average in each tree
            if self.log_y:
                for tree_id in range(self.forest_options.num_trees):
                    dat_[i, tree_id] = np.log(
                        np.exp(np.array(preds_trees[tree_id])).mean()
                    )
            else:
                for tree_id in range(self.forest_options.num_trees):
                    dat_[i, tree_id] = np.array(preds_trees[tree_id]).mean()

        # 3. compute statistics across trees
        mean_ = dat_.mean(axis=1)
        var = dat_.var(axis=1)

        var[var < self.var_threshold] = self.var_threshold

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var
