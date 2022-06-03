from typing import Tuple

import ConfigSpace as CS
import numpy as np
from pyPDP.surrogate_models import SurrogateModel

from deepcave.evaluators.epm.random_forest import RandomForest


class RandomForestSurrogate(SurrogateModel):
    """
    Random forest surrogate for the pyPDP package.
    """

    def __init__(
        self,
        configspace: CS.ConfigurationSpace,
        seed: int = None,
    ):
        super().__init__(configspace, seed=seed)
        self._model = RandomForest(configspace=configspace, seed=seed)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means, stds = self._model.predict(X)
        return means[:, 0], stds[:, 0]

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self._model.train(X, y)
