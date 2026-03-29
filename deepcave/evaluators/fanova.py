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

#  noqa: D400
"""
# fANOVA

This module provides a tool for assessing the importance of an algorithms Hyperparameters.

Utilities provide calculation of the data wrt the budget and train the forest on the encoded data.

## Classes
    - fANOVA: Calculate and provide midpoints and sizes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import optuna
from optuna.importance import FanovaImportanceEvaluator

from deepcave.constants import COMBINED_COST_NAME
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger


class fANOVA:
    """
    Calculate and provide midpoints and sizes.

    Properties
    ----------
    run : AbstractRun
        The Abstract Run used for the calculation.
    cs : ConfigurationSpace
        The configuration space of the run.
    hps : List[Hyperparameters]
        The Hyperparameters of the configuration space.
    hp_names : List[str]
        The corresponding names of the Hyperparameters.
    n_trees : int
        The number of trees.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self.hps = list(self.cs.values())
        self.hp_names = list(self.cs.keys())
        self.logger = get_logger(self.__class__.__name__)

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        n_trees: int = 16,
        seed: int = 0,
        y: Any = None,
    ) -> Any:
        """Create Optuna study from data and fit Fanova evaluator."""
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        # Get data
        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()

        # Combined cost name includes the cost of all selected objectives
        if y is not None:
            Y = y
        else:
            Y = df[COMBINED_COST_NAME].to_numpy()

        self.study = optuna.create_study()

        params: dict = {}
        distributions: dict = {}

        for i in range(len(Y)):
            params = {}
            distributions = {}

            for j, hp in enumerate(self.hps):
                name = hp.name
                val = X[i, j]

                if hp.__class__.__name__ == "CategoricalHyperparameter":
                    # For categorical, val is assumed to be an index
                    idx = int(val)

                    params[name] = hp.choices[idx]  # type: ignore
                    distributions[name] = optuna.distributions.CategoricalDistribution(
                        hp.choices  # type: ignore
                    )

                elif hp.__class__.__name__ in (
                    "UniformFloatHyperparameter",
                    "UniformIntegerHyperparameter",
                    "OrdinalHyperparameter",
                ):
                    params[name] = float(val)
                    distributions[name] = optuna.distributions.FloatDistribution(low=0, high=1)

                else:
                    raise NotImplementedError(f"Unknown hyperparameter type: {hp}")

            trial = optuna.trial.create_trial(
                params=params, distributions=distributions, value=float(Y[i])
            )
            self.study.add_trial(trial)

        self.evaluator = FanovaImportanceEvaluator(seed=seed, n_trees=n_trees)

    def get_importances(
        self, hp_names: Optional[List[str]] = None, depth: int = 1, sort: bool = True
    ) -> Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]:
        """
        Return the importance scores from the passed Hyperparameter names.

        Warning
        -------
        Using a depth higher than 1 might take much longer.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from. If None, all
            Hyperparameters of the configuration space are used.
        depth : int, optional
            How often dimensions should be combined. By default 1.
        sort : bool, optional
            Whether the Hyperparameters should be sorted by importance. By default True.

        Returns
        -------
        Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]
            Dictionary with Hyperparameter names and the corresponding importance scores.
            The values are tuples of the form (mean individual, var individual, mean total,
            var total). Note that individual and total are the same if depth is 1.

        Raises
        ------
        RuntimeError
            If there is zero total variance in all trees.
        """
        if hp_names is None:
            hp_names = self.cs.get_hyperparameter_names()

        hp_ids = []
        for hp_name in hp_names:
            hp_ids.append(self.cs.index_of[hp_name])

        importances = self.evaluator.evaluate(self.study)

        # Sort by total mean fraction
        if sort:
            importances = {
                k: (v, 0) for k, v in sorted(importances.items(), key=lambda item: item[1])
            }

        return importances
