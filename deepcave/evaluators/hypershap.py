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
# HyperSHAP_Eval

This module uses HyperSHAP for explaining Hyperparameter Optimization (HPO).

HyperSHAP is a game-theoretic Python library that uses Shapley values and
interaction indices to provide local and global insights into how
individual hyper-parameters affect a model's performance.
This module includes the evaluation of tunabilty, mistuneability, and ablation.

## Classes:
    - HyperSHAP_Eval: Provide an evaluator used for HyperSHAP evaluation.
"""

from typing import Dict

from hypershap.hypershap import HyperSHAP
from hypershap.task import ExplanationTask

from deepcave.runs import AbstractRun, Status
from deepcave.utils.logs import get_logger


class HyperSHAP_Eval:
    """
    Provide an evaluator used for HyperSHAP evaluation.

    Properties
    ----------
    run : AbstractRun
        The run to analyze.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        The names of the hyperparameters of the configspace.
    hypershap : HyperSHAP
        HyperSHAP object used for evaluation.
    iv : InteractionValues
        An object containing evaluation results.
    tune : Dict[str: Any]
        The iv tuneability parameters converted as python dictionary.
    abl : Dict[str: Any]
        The iv ablation parameters converted as python dictionary.
    """

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = list(self.cs.keys())
        self.logger = get_logger(self.__class__.__name__)

    def hype_tune(self, tunability: str, objective_id: int, budget_id: int, seed: int = 42) -> Dict:
        """
        Calculate the tunability or mistunability.

        Parameters
        ----------
        tunability : str
            Whether to calculate tunability or mistunability.
        objective_id : int
            The id of the objective to evaluate on.
        budget_id : int
            The id of the budget to evaluate on.

        Returns
        -------
        Dict
            The dictionary with the evaluation results.
        """
        if budget_id is None:
            budget = self.run.get_highest_budget()

        objective = self.run.get_objective(objective_id)
        budget = self.run.get_budget(budget_id)

        df = self.run.get_encoded_data(
            objective,
            budget,
            statuses=Status.SUCCESS,
        )

        # Match the configuration data with the resulting performance
        configuration_list = self.cs.sample_configuration(size=1_000)
        data = list(zip(configuration_list, df[objective.name].to_numpy()))  # type: ignore

        explanation_task = ExplanationTask.from_data(config_space=self.cs, data=data, seed=seed)
        self.hypershap = HyperSHAP(explanation_task=explanation_task)

        baseline_config = self.cs.sample_configuration()
        if tunability == "Tunability":
            self.iv = self.hypershap.tunability(baseline_config=baseline_config)

        elif tunability == "Mistunability":
            self.iv = self.hypershap.mistunability(baseline_config=baseline_config)

        # Convert to python dictionary to avoid JSON serializability problems
        self.tune = {
            "index": str(self.iv.index),
            "max_order": self.iv.max_order,
            "min_order": self.iv.min_order,
            "estimated": self.iv.estimated,
            "estimation_budget": self.iv.estimation_budget,
            "n_players": self.iv.n_players,
            "baseline_value": self.iv.baseline_value,
            "interactions": {str(key): value for key, value in self.iv.interactions.items()},
        }

        return self.tune

    def hype_ablation(self, objective_id: int, budget_id: int, seed: int = 42) -> Dict:
        """
        Calculate the ablation.

        Parameters
        ----------
        objective_id : int
            The id of the objective to evaluate on.
        budget_id : int
            The id of the budget to evaluate on.

        Returns
        -------
        Dict
            The dictionary with the evaluation results.

        Raises
        ------
        ValueError
            If the Objective is None.
        """
        if budget_id is None:
            budget_id = self.run.get_highest_budget()

        objective = self.run.get_objective(objective_id)
        budget = self.run.get_budget(budget_id)

        if objective is None:
            raise ValueError(
                "No Objective has been chosen. Please select an "
                "Objective or try to select the Objective again."
            )

        df = self.run.get_encoded_data(
            objective,
            budget,
            statuses=Status.SUCCESS,
        )

        # Match the configuration data with the resulting performance
        configuration_list = self.cs.sample_configuration(size=1_000)
        data = list(zip(configuration_list, df[objective.name].to_numpy()))  # type: ignore

        explanation_task = ExplanationTask.from_data(config_space=self.cs, data=data, seed=seed)
        self.hypershap = HyperSHAP(explanation_task=explanation_task)

        baseline_config = self.cs.get_default_configuration()
        config_of_interest, _ = self.run.get_incumbent(budget=budget, objectives=objective)

        self.iv = self.hypershap.ablation(
            config_of_interest=config_of_interest, baseline_config=baseline_config
        )

        # Convert to python dictionary to avoid JSON serializability problems
        self.abl = {
            "index": str(self.iv.index),
            "max_order": self.iv.max_order,
            "min_order": self.iv.min_order,
            "estimated": self.iv.estimated,
            "estimation_budget": self.iv.estimation_budget,
            "n_players": self.iv.n_players,
            "baseline_value": self.iv.baseline_value,
            "interactions": {str(key): value for key, value in self.iv.interactions.items()},
        }

        return self.abl

    def get_tunability(self) -> Dict:
        """
        Get the tunability or mistunability evaluation values.

        Returns
        -------
        Dict
            A dictionary containing the evaluation values.
        """
        return self.tune

    def get_ablation(self) -> Dict:
        """
        Get the tunability or mistunability evaluation values.

        Returns
        -------
        Dict
            A dictionary containing the evaluation values.
        """
        return self.abl
