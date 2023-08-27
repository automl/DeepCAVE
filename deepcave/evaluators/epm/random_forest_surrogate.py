# noqa: D400
"""
# RandomForest Surrogate

This module provides an implementation for a RandomForest Surrogate model.

Mean and standard deviation values can be computed for a given input with this module.

## Contents
    - predict: Predict the deviations
    - _fit: Fits the model
"""
from typing import Tuple

import ConfigSpace as CS
import numpy as np
from pyPDP.surrogate_models import SurrogateModel

from deepcave.evaluators.epm.random_forest import RandomForest


class RandomForestSurrogate(SurrogateModel):
    """Random forest surrogate for the pyPDP package."""

    def __init__(
        self,
        configspace: CS.ConfigurationSpace,
        seed: int = None,
    ):  # noqa: D107
        super().__init__(configspace, seed=seed)
        self._model = RandomForest(configspace=configspace, seed=seed)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D102
        means, stds = self._model.predict(X)
        return means[:, 0], stds[:, 0]

    def _fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D102
        self._model.train(X, y)
