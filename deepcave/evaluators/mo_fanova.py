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

from typing import List, Optional, Union

import pandas as pd

from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.fanova import fANOVA
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.multi_objective_importance import get_weightings


class MOfANOVA(fANOVA):
    """
    Multi-Objective fANOVA.

    Calculate and provide midpoints and sizes from the forest's split values in order to get
    the marginals.
    Override: to train the random forest with an arbitrary weighting of the objectives
    (multi-objective case).
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        super().__init__(run)
        self.importances_ = None

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        n_trees: int = 100,
        seed: int = 0,
    ) -> None:
        """
        Get the data with respect to budget and train the forest on the encoded data.

        Calculates weighted fanova for multiple objectives.

        Note
        ----
        Right now, only `n_trees` is used. It can be further specified if needed.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        n_trees : int, optional
            How many trees should be used. By default 100.
        seed : int
            Random seed. By default 0.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        self.n_trees = n_trees

        # Get data
        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )

        # normalize objectives
        assert isinstance(objectives, list)
        objectives_normed = list()
        for obj in objectives:
            normed = obj.name + "_normed"
            df[normed] = (df[obj.name] - df[obj.name].min()) / (
                df[obj.name].max() - df[obj.name].min()
            )
            if obj.optimize == "upper":
                df[normed] = 1 - df[normed]
            objectives_normed.append(normed)
        df = df.dropna(subset=objectives_normed)
        X = df[self.hp_names].to_numpy()
        weightings = get_weightings(objectives_normed, df)
        df_all = pd.DataFrame([])

        # calculate importance for each weighting generated from the pareto efficient points
        for w in weightings:
            Y = sum(df[obj] * weighting for obj, weighting in zip(objectives_normed, w)).to_numpy()

            self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
            self._model.train(X, Y)
            df_res = (
                pd.DataFrame(super(MOfANOVA, self).get_importances(hp_names=None))
                .loc[0:1]
                .T.reset_index()
            )
            df_res["weight"] = w[0]
            df_all = pd.concat([df_all, df_res])
        self.importances_ = df_all.rename(
            columns={0: "importance", 1: "variance", "index": "hp_name"}
        ).reset_index(drop=True)

    def get_importances_(self, hp_names: Optional[List[str]] = None) -> str:
        """
        Return the importance scores from the passed Hyperparameter names.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from. If None, all
            Hyperparameters of the configuration space are used.
        sort : bool, optional
            Whether the Hyperparameters should be sorted by importance. By default True.

        Returns
        -------
        Dict
            Dictionary with Hyperparameter names and the corresponding importance scores and
            variances.

        Raises
        ------
        RuntimeError
            If the important scores are not calculated.
        """
        if self.importances_ is None:
            raise RuntimeError("Importance scores must be calculated first.")

        if hp_names:
            return self.importances_.loc[self.importances_["hp_name"].isin(hp_names)].to_json()
        else:
            return self.importances_.to_json()
