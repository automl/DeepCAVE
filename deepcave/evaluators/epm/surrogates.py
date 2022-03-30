import random

import numpy as np
import ConfigSpace as CS

from pyPDP.surrogate_models import SurrogateModel
from deepcave.evaluators.epm.random_forest import RandomForest


class RandomForestSurrogate(SurrogateModel):
    def __init__(
            self,
            cs: CS.ConfigurationSpace,
            *,
            seed: int = None,
            num_trees: int = 16,
            bootstrapping: bool = True,
            points_per_tree: int = -1,
            ratio_features: float = 5.0 / 6.0,
            min_samples_split: int = 3,
            min_samples_leaf: int = 3,
            max_depth: int = 2 ** 20,
            eps_purity: float = 1e-8,
            max_num_nodes: int = 2 ** 20,
    ):
        super().__init__(cs, seed=seed)
        if seed is None:
            seed = random.randint(0, 2 ** 16 - 1)
        self.forest = RandomForest(
            cs,
            seed,
            num_trees,
            bootstrapping,
            points_per_tree,
            ratio_features,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            eps_purity,
            max_num_nodes
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return
        """
        means, stds = self.forest.predict(X)
        return means[:, 0], stds[:, 0]

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.forest.train(X, y)
