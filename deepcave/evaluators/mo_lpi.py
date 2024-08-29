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
# LPI

This module provides utilities to calculate the local parameter importance (LPI).

## Classes
    - LPI: This class calculates the local parameter importance (LPI).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from ConfigSpace.c_util import change_hp_value
from ConfigSpace.util import impute_inactive_values

from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.lpi import LPI
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.multi_objective_importance import get_weightings


# https://github.com/automl/ParameterImportance/blob/f4950593ee627093fc30c0847acc5d8bf63ef84b/pimp/evaluator/local_parameter_importance.py#L27
class MOLPI(LPI):
    """
    Calculate the multi-objective local parameter importance (LPI).

    Override: to train the random forest with an arbitrary weighting of the objectives
    (multi-objective case).

    Properties
    ----------
    run : AbstractRun
        The AbstractRun to get the importance from.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        The names of the Hyperparameters.
    variances : Dict[Any, list]
        The overall variances per tree.
    importances : dict
        The importances of the Hyperparameters.
    continuous_neighbors : int
        The number of neighbors chosen for continuous Hyperparameters.
    incumbent : Configuration
        The incumbent of the run.
    default : Configuration
        A configuration containing Hyperparameters with default values.
    incumbent_array : numpy.ndarray
        The internal vector representation of the incumbent.
    seed : int
        The seed. If not provided it will be random.
    rs : RandomState
        A random state with a given seed value.
    """

    def __init__(self, run: AbstractRun):
        super().__init__(run)
        self.importances: Optional[pd.DataFrame] = None

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        continous_neighbors: int = 500,
        n_trees: int = 10,
        seed: int = 0,
    ) -> None:
        """
        Prepare the data and train a RandomForest model.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default, None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default, None. If None, the highest budget is chosen.
        continuous_neighbors : int, optional
            How many neighbors should be chosen for continuous hyperparameters (HPs).
            By default, 500.
        n_trees : int, optional
            The number of trees for the fanova forest.
            Default is 10.
        seed : Optional[int], optional
            The seed. By default None. If None, a random seed is chosen.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        # Set variables
        self.continous_neighbors = continous_neighbors
        self.default = self.cs.get_default_configuration()

        self.seed = seed
        self.rs = np.random.RandomState(seed)

        # Get data
        df = self.run.get_encoded_data(
            objectives=objectives,
            budget=budget,
            specific=True,
            include_combined_cost=True,
            include_config_ids=True,
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
        df_all = pd.DataFrame([])
        weightings = get_weightings(objectives_normed, df)

        # calculate importance for each weighting generated from the pareto efficient points
        for w in weightings:
            Y = sum(df[obj] * weighting for obj, weighting in zip(objectives_normed, w)).to_numpy()
            # Use same forest as for fanova
            self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
            self._model.train(X, Y)

            incumbent_cfg_id = np.argmin(sum(df[obj] * w for obj, w in zip(objectives_normed, w)))
            self.incumbent = self.run.get_config(df.iloc[incumbent_cfg_id]["config_id"])
            self.incumbent_array = self.incumbent.get_array()
            importances = self.calc_one_weighting()
            df_res = pd.DataFrame(importances).loc[0:1].T.reset_index()
            df_res["weight"] = w[0]
            df_all = pd.concat([df_all, df_res])
        self.importances = df_all.rename(
            columns={0: "importance", 1: "variance", "index": "hp_name"}
        ).reset_index(drop=True)
        self.importances = self.importances.map(
            lambda x: max(x, 0) if not isinstance(x, str) else x
        )  # no negative values

    def calc_one_weighting(self) -> Dict[str, Tuple[float, float]]:
        """
        Prepare the data after a model has be trained for one weighting.

        Returns
        -------
        imp_var_dict: Dict[str, Tuple[float, float]]
            Dictionary of importances and variances.
        """
        # Get neighborhood sampled on an unit-hypercube.
        neighborhood = self._get_neighborhood()

        # The delta performance is needed from the default configuration and the incumbent
        def_perf, def_var = self._predict_mean_var(self.default)
        inc_perf, inc_var = self._predict_mean_var(self.incumbent)
        delta = def_perf - inc_perf

        # These are used for plotting and hold the predictions for each neighbor of each parameter.
        # That means performances holds the mean, variances the variance of the forest.
        performances: Dict[str, List[np.ndarray]] = {}
        variances: Dict[str, List[np.ndarray]] = {}
        # These are used for importance and hold the corresponding importance/variance over
        # neighbors. Only import if NOT quantifying importance via performance-variance across
        # neighbors.

        # Nested list of values per tree in random forest.
        predictions: Dict[str, List[List[np.ndarray]]] = {}

        # Iterate over parameters
        for hp_idx, hp_name in enumerate(self.incumbent.keys()):
            if hp_name not in neighborhood:
                continue

            performances[hp_name] = []
            variances[hp_name] = []
            predictions[hp_name] = []
            incumbent_added = False
            incumbent_idx = 0

            # Iterate over neighbors
            for unit_neighbor, neighbor in zip(neighborhood[hp_name][0], neighborhood[hp_name][1]):
                if not incumbent_added:
                    # Detect incumbent
                    if unit_neighbor > self.incumbent_array[hp_idx]:
                        performances[hp_name].append(inc_perf)
                        variances[hp_name].append(inc_var)
                        incumbent_added = True
                    else:
                        incumbent_idx += 1

                # Create the neighbor-Configuration object
                new_array = self.incumbent_array.copy()
                new_array = change_hp_value(
                    self.cs, new_array, hp_name, unit_neighbor, self.cs.index_of[hp_name]
                )
                new_config = impute_inactive_values(Configuration(self.cs, vector=new_array))

                # Get the leaf values
                x = np.array(new_config.get_array())
                leaf_values = self._model.get_leaf_values(x)

                # And the prediction/performance/variance
                predictions[hp_name].append([np.mean(tree_pred) for tree_pred in leaf_values])
                performances[hp_name].append(np.mean(predictions[hp_name][-1]))
                variances[hp_name].append(np.var(predictions[hp_name][-1]))

            if len(neighborhood[hp_name][0]) > 0:
                neighborhood[hp_name][0] = np.insert(
                    neighborhood[hp_name][0], incumbent_idx, self.incumbent_array[hp_idx]
                )
                neighborhood[hp_name][1] = np.insert(
                    neighborhood[hp_name][1], incumbent_idx, self.incumbent[hp_name]
                )
            else:
                neighborhood[hp_name][0] = np.array(self.incumbent_array[hp_idx])
                neighborhood[hp_name][1] = [self.incumbent[hp_name]]

            if not incumbent_added:
                performances[hp_name].append(inc_perf)
                variances[hp_name].append(inc_var)

            # Avoid division by zero
            if delta == 0:
                delta = 1

        # Creating actual importance value (by normalizing over sum of vars)
        num_trees = len(list(predictions.values())[0][0])
        hp_names = list(performances.keys())

        overall_var_per_tree = {}
        for hp_name in hp_names:
            hp_variances = []
            for tree_idx in range(num_trees):
                variance = np.var([neighbor[tree_idx] for neighbor in predictions[hp_name]])
                hp_variances += [variance]

            overall_var_per_tree[hp_name] = hp_variances

        # Sum up variances per tree across parameters
        sum_var_per_tree = [
            sum([overall_var_per_tree[hp_name][tree_idx] for hp_name in hp_names])
            for tree_idx in range(num_trees)
        ]

        # Normalize
        overall_var_per_tree = {
            p: [
                t / sum_var_per_tree[idx] if sum_var_per_tree[idx] != 0.0 else np.nan
                for idx, t in enumerate(trees)
            ]
            for p, trees in overall_var_per_tree.items()
        }
        imp_var_dict = {
            k: (np.mean(overall_var_per_tree[k]), np.var(overall_var_per_tree[k]))
            for k in overall_var_per_tree
        }
        return imp_var_dict

    def get_importances_(self, hp_names: List[str]) -> str:
        """
        Return the importance scores from the passed Hyperparameter names.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from. If None, all
            Hyperparameters of the configuration space are used.

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
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")

        if hp_names:
            return self.importances.loc[self.importances["hp_name"].isin(hp_names)].to_json()
        else:
            return self.importances.to_json()
