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
# RandomForest Surrogate

This module provides a RandomForest Surrogate model.

Mean and standard deviation values can be predicted for a given input with this module.

## Classes
    - RandomForestSurrogate: Random forest surrogate for the pyPDP package.
"""

from typing import Optional, Tuple

import ConfigSpace as CS
import numpy as np
from pyPDP.surrogate_models import SurrogateModel

from deepcave.evaluators.epm.random_forest import RandomForest


class RandomForestSurrogate(SurrogateModel):
    """
    Random forest surrogate for the pyPDP package.

    Predict deviations and fit the model.
    """

    def __init__(
        self,
        configspace: CS.ConfigurationSpace,
        seed: Optional[int] = None,
        n_trees: int = 16,
    ):
        super().__init__(configspace, seed=seed)
        self._model = RandomForest(configspace=configspace, seed=seed, n_trees=n_trees)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the deviations.

        Parameters
        ----------
        X : np.ndarray
            The data points on which to predict.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The means and standard deviation.
        """
        means, stds = self._model.predict(X)
        return means[:, 0], stds[:, 0]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        y : np.ndarray
            Corresponding target values.
        """
        self._model.train(X, y)
