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
# Ablation Paths

This module evaluates the ablation paths.

Ablation Paths is a method to analyze the importance of hyperparameters in a configuration space.
Starting from a default configuration, the default configuration is iteratively changed to the
incumbent configuration by changing one hyperparameter at a time, choosing the
hyperparameter that leads to the largest improvement in the objective function at each step.

## Classes:
    - Ablation: Provide an evaluator of the ablation paths.
"""

from typing import Any, List, Optional, Tuple, Union

import copy

import numpy as np
import pandas as pd

from deepcave.evaluators.ablation import Ablation
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


class MOAblation(Ablation):
    """
    Provide an evaluator of the ablation paths.

    Override: Multi-Objective case

    Properties
    ----------
    run : AbstractRun
        The run to analyze.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        A list of the hyperparameter names.
    performances : Optional[Dict[Any, Any]]
        A dictionary containing the performances for each HP.
    improvements : Optional[Dict[Any, Any]]
        A dictionary containing the improvements over the respective previous step for each HP.
    objectives : Optional[Union[Objective, List[Objective]]]
        The objective(s) of the run.
    default_config : Configurations
        The default configuration of this configuration space.
        Gets changed step by step towards the incumbent configuration.
    """

    def __init__(self, run: AbstractRun):
        super().__init__(run)
        self.models: List = []
        self.df_importances = pd.DataFrame([])

    def get_importances(self) -> str:
        """
        Return the importance scores.

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
        if self.df_importances is None:
            raise RuntimeError("Importance scores must be calculated first.")

        return self.df_importances.to_json()

    def get_weightings(self, objectives_normed: List[str], df: pd.DataFrame) -> np.ndarray:
        """
        Calculate the weighting for the weighted importance using the points on the pareto-front.

        Parameters
        ----------
        objectives_normed : List[str]
            The normalized objective names as a list of strings.
        df : pandas.dataframe
            The dataframe containing the encoded data.

        Returns
        -------
        weightings : numpy.ndarray[numpy.ndarray]
             The weightings as a list of lists.
        """
        optimized = self.is_pareto_efficient(df[objectives_normed].to_numpy())
        return (
            df[optimized][objectives_normed]
            .T.apply(lambda values: values / values.sum())
            .T.to_numpy()
        )

    def is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find the pareto-efficient points.

        Parameters
        ----------
        costs : numpy.ndarray
            An (n_points, n_costs) array.

        Returns
        -------
        is_efficient : numpy.ndarray
             A (n_points, ) boolean array, indicating whether each point is Pareto efficient.
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
                np.any(costs[i + 1 :] > c, axis=1)
            )
        return is_efficient

    def predict(self, cfg, weighting):
        """
        Predict the performance of the input configuration.

        The model results are weighted by the input weightings and summed.

        Parameters
        ----------
        cfg : Dict
            Configuration.
        weighting : List[float]
            Weightings.

        Returns
        -------
        mean : float
             The mean of the weighted sum of predictions.
        var : float
             The variance of the weighted sum of predictions.
        """
        mean, var = 0, 0
        for model, w in zip(self.models, weighting):
            pred, var_ = model.predict(np.array([cfg]))
            mean += w * pred[0]
            var += w * var_[0]
        return mean, var

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]],  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        n_trees: int = 50,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """
        Calculate the MO ablation path performances and improvements.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]]
            The objective(s) to be considered.
        budget : Optional[Union[int, float]]
            The budget to be considered. If None, all budgets of the run are considered.
            Default is None.
        n_trees : int
            The number of trees for the surrogate model.
            Default is 50.
        seed : int
            The seed for the surrogate model.
            Default is 0.
        """
        for objective in objectives:
            assert isinstance(objective, Objective)

        df = self.run.get_encoded_data(objectives, budget, specific=True, include_config_ids=True)

        # Obtain all configurations with theirs costs
        df = df.dropna(subset=[obj.name for obj in objectives])
        X = df[list(self.run.configspace.keys())].to_numpy()

        # normalize objectives
        objectives_normed = list()
        for obj in objectives:
            normed = obj.name + "_normed"
            df[normed] = (df[obj.name] - df[obj.name].min()) / (
                df[obj.name].max() - df[obj.name].min()
            )
            objectives_normed.append(normed)

            # train one model per objective
            Y = df[normed].to_numpy()
            model = RandomForestSurrogate(self.cs, seed=seed, n_trees=n_trees)
            model._fit(X, Y)
            self.models.append(model)

        weightings = self.get_weightings(objectives_normed, df)

        # calculate importance for each weighting generated from the pareto efficient points
        for w in weightings:
            df_res = self.calculate_ablation_path(df, objectives_normed, w, budget)
            df_res["weight"] = w[0]
            self.df_importances = pd.concat([self.df_importances, df_res])
        self.df_importances = self.df_importances.reset_index(drop=True)

    def calculate_ablation_path(self, df: pd.DataFrame, objectives_normed: List[str], weighting: np.ndarray, budget: Optional[Union[int, float]]) -> pd.DataFrame:
        """
        Calculate the ablation path performances.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with encoded data.
        objectives_normed : List[str]
            The normed objective names to be considered.
        weighting : np.ndarray
            The weighting of the objective values.
        budget : Optional[Union[int, float]]
            The budget to be considered. If None, all budgets of the run are considered.
            Default is None.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with results of the ablation calculation.
        """
        # Get the incumbent configuration
        incumbent_cfg_id = np.argmin(
            sum(df[obj] * w for obj, w in zip(objectives_normed, weighting))
        )
        incumbent_config = self.run.get_config(df.iloc[incumbent_cfg_id]["config_id"])

        # Get the default configuration
        self.default_config = self.cs.get_default_configuration()
        default_encode = self.run.encode_config(self.default_config, specific=True)

        # Obtain the predicted cost of the default and incumbent configuration
        def_cost, def_std = self.predict(default_encode, weighting)
        inc_cost, _ = self.predict(
            self.run.encode_config(incumbent_config, specific=True), weighting
        )

        # TODO make sure objectives are minimized

        if inc_cost > def_cost:
            self.logger.warning(
                "The predicted incumbent objective is worse than the predicted default "
                f"objective for budget: {budget}. Aborting ablation path calculation."
            )
            return None
        else:
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()
            df_abl = pd.DataFrame([])
            df_abl = pd.concat(
                [
                    df_abl,
                    pd.DataFrame(
                        {
                            "hp_name": "Default",
                            "importance": 0,
                            "variance": def_std,
                            "new_performance": def_cost,
                        },
                        index=[0],
                    ),
                ]
            )

            for i in range(len(hp_it)):
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_cost, max_hp_std = self.ablation(
                    budget, incumbent_config, def_cost, hp_it, weighting
                )

                if not continue_ablation:
                    break

                diff = def_cost - max_hp_cost
                def_cost = max_hp_cost

                df_abl = pd.concat(
                    [
                        df_abl,
                        pd.DataFrame(
                            {
                                "hp_name": max_hp,
                                "importance": diff,
                                "variance": max_hp_std,
                                "new_performance": max_hp_cost,
                            },
                            index=[i + 1],
                        ),
                    ]
                )

                # Remove the current best hp for keeping the order right
                hp_it.remove(max_hp)
            return df_abl.reset_index(drop=True)

    def ablation(
        self,
        budget: Optional[Union[int, float]],
        incumbent_config: Any,
        def_cost: Any,
        hp_it: List[str],
        weighting: Tuple[float, float],
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Calculate the ablation importance for each hyperparameter.

        Parameters
        ----------
        budget: Optional[Union[int, float]]
            The budget of the run.
        incumbent_config: Any
            The incumbent configuration.
        def_cost: Any
            The default cost.
        hp_it: List[str]
            A list of the HPs that still have to be looked at.
        weighting : List[float]
            The weighting of the objective values.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            continue_ablation, max_hp, max_hp_performance, max_hp_std
        """
        max_hp = ""
        max_hp_difference = 0

        for hp in hp_it:
            if hp in incumbent_config.keys() and hp in self.default_config.keys():
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]

                new_cost, _ = self.predict(
                    self.run.encode_config(config_copy, specific=True), weighting
                )
                difference = def_cost - new_cost

                # Check for the maximum difference hyperparameter in this round
                if difference > max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
        hp_count = len(list(self.cs.keys()))
        if max_hp != "":
            # For the maximum impact hyperparameter, switch the default with the incumbent value
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_cost, max_hp_std = self.predict(
                self.run.encode_config(self.default_config, specific=True), weighting
            )
            return True, max_hp, max_hp_cost, max_hp_std
        else:
            self.logger.info(
                f"End ablation at step {hp_count - len(hp_it) + 1}/{hp_count} "
                f"for budget {budget} (remaining hyperparameters not activate in incumbent or "
                "default configuration)."
            )
            return False, None, None, None
