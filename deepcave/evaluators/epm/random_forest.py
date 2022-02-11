from typing import Optional

import numpy as np
import pyrfr.regression as regression
from smac.configspace import ConfigurationSpace
from smac.utils.constants import N_TREES, VERY_SMALL_NUMBER

from deepcave.evaluators.epm.forest import Forest


class RandomForest(Forest):
    """
    Simple wrapper to calculate types and bounds automatically.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        seed: int,
        num_trees: int = N_TREES,
        bootstrapping: bool = True,
        points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_num_nodes: int = 2**20,
    ):

        super().__init__(configspace, seed)

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

        means, vars_ = [], []
        for row_X in X:
            mean_, var = self.model.predict_mean_var(row_X)
            means.append(mean_)
            vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means, vars_  # means.reshape((-1, 1)), vars_.reshape((-1, 1))
