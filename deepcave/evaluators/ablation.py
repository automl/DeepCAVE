# noqa: D400
"""
# Ablation Paths

This module evaluates the ablation paths.

Starting from a default configuration, the ablation path method iteratively changes the default
configuration to the incumbent configuration by changing one hyperparameter at a time, choosing the
hyperparameter that leads to the largest improvement in the objective function at each step.

## Classes:
    - Ablation: Provide an evaluator of the ablation paths.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import copy
from collections import OrderedDict

import numpy as np

from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger


class Ablation:
    """
    Provide an evaluator of the ablation paths.

    Properties
    ----------
    run : AbstractRun
        The run to analyze.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        A list of the hyperparameter names.
    importances : Optional[Dict[Any, Any]]
        A dictionary containing the importances for each HP.
    objectives : Optional[Union[Objective, List[Objective]]]
        The objective(s) of the run.
    default_encode : List
        An encoding of the default configuration.
    default_config : Configurations
        The default configuration of this configuration space.
        Gets changed step by step to the incumbent.
    """

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = self.cs.get_hyperparameter_names()
        self.importances: Optional[Dict[Any, Any]] = None
        self.logger = get_logger(self.__class__.__name__)

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]],  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        n_trees: int = 50,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """
        Calculate the ablation path importances.

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
        if isinstance(objectives, list) and len(objectives) > 1:
            raise ValueError("Only one objective is supported for ablation importances.")
        objective = objectives[0] if isinstance(objectives, list) else objectives
        assert isinstance(objective, Objective)

        performances: OrderedDict = OrderedDict()
        improvements: OrderedDict = OrderedDict()

        df = self.run.get_encoded_data(objective, budget, specific=True)

        # Obtain all configurations with theirs costs
        df = df.dropna(subset=[objective.name])
        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        Y = df[objective.name].to_numpy()

        # A Random Forest Regressor is used as surrogate model
        self._model = RandomForestSurrogate(self.cs, seed=seed, n_trees=n_trees)
        self._model._fit(X, Y)

        # Get the incumbent configuration
        incumbent_config, _ = self.run.get_incumbent(budget=budget, objectives=objective)
        incumbent_encode = self.run.encode_config(incumbent_config)

        # Get the default configuration
        self.default_config = self.cs.get_default_configuration()
        default_encode = self.run.encode_config(self.default_config)

        # Obtain the predicted cost of the default and incumbent configuration
        def_cost, def_std = self._model.predict(np.array([default_encode]))
        def_cost, def_std = def_cost[0], def_std[0]
        inc_cost, _ = self._model.predict(np.array([incumbent_encode]))

        # For further calculations, assume that the objective is to be minimized
        if objective.optimize == "upper":
            def_cost = -def_cost
            inc_cost = -inc_cost

        if inc_cost > def_cost:
            self.logger.warning(
                "The predicted incumbent cost is smaller than the predicted default "
                f"cost for budget: {budget}. This could mean that the configuration space "
                "with which the surrogate model was trained contained too few examples."
            )
            performances = OrderedDict({hp_name: (0, 0) for hp_name in self.hp_names})
            improvements = OrderedDict({hp_name: (0, 0) for hp_name in self.hp_names})
        else:
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()
            for i in range(len(hp_it)):
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_cost, max_hp_std = self._ablation(
                    objective, budget, incumbent_config, def_cost, hp_it
                )

                if not continue_ablation:
                    break

                # As only one objective is allowed, there is only one objective in the list
                if objective.optimize == "upper":
                    # For returning the importance, flip back the objective if it was flipped before
                    performances[max_hp] = (-max_hp_cost, max_hp_std)
                else:
                    performances[max_hp] = (max_hp_cost, max_hp_std)
                impr_std = np.sqrt(def_std**2 + max_hp_std**2)
                improvements[max_hp] = ((def_cost - max_hp_cost), impr_std)
                # New 'default' cost and std
                def_cost = max_hp_cost
                def_std = max_hp_std
                # Remove the current best hp for keeping the order right
                hp_it.remove(max_hp)

        self.performances = performances
        self.improvements = improvements

    def get_ablation_performances(self) -> Optional[Dict[Any, Any]]:
        """
        Get the ablation performances.

        Returns
        -------
        Optional[Dict[Any, Any]]
            A dictionary containing the ablation performances.

        Raises
        ------
        RuntimeError
            If the ablation performances have not been calculated.
        """
        if self.performances is None:
            raise RuntimeError("Ablation performances scores must be calculated first.")
        return self.performances

    def get_ablation_improvements(self) -> Optional[Dict[Any, Any]]:
        """
        Get the ablation improvements.

        Returns
        -------
        Optional[Dict[Any, Any]]
            A dictionary containing the ablation improvements.

        Raises
        ------
        RuntimeError
            If the ablation improvements have not been calculated.
        """
        if self.improvements is None:
            raise RuntimeError("Ablation improvements must be calculated first.")

        return self.improvements

    def _ablation(
        self,
        objective: Objective,
        budget: Optional[Union[int, float]],
        incumbent_config: Any,
        def_cost: Any,
        hp_it: List[str],
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Calculate the ablation importance for each hyperparameter.

        Parameters
        ----------
        objective: Objective
            The objective to be considered.
        budget: Optional[Union[int, float]]
            The budget of the run.
        incumbent_config: Any
            The incumbent configuration.
        def_cost: Any
            The default cost.
        hp_it: List[str]
            A list of the HPs that still have to be looked at.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            continue_ablation, max_hp, max_hp_performance, max_hp_std
        """
        max_hp = ""
        max_hp_difference = -np.inf

        for hp in hp_it:
            if incumbent_config[hp] is not None and hp in self.default_config.keys():
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]

                new_cost, _ = self._model.predict(np.array([self.run.encode_config(config_copy)]))
                if objective.optimize == "upper":
                    new_cost = -new_cost

                difference = def_cost - new_cost

                # Check for the maximum difference hyperparameter in this round
                if difference >= max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
        hp_count = len(self.cs.get_hyperparameter_names())
        if max_hp != "":
            if max_hp_difference <= 0:
                self.logger.info(
                    "No improvement found in ablation step "
                    f"{hp_count - len(hp_it) + 1}/{hp_count} for budget {budget}, "
                    "choose hyperparameter with smallest increase in cost."
                )
            # For the maximum impact hyperparameter, switch the default with the incumbent value
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_cost, max_hp_std = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            if objective.optimize == "upper":
                max_hp_cost = -max_hp_cost
            return True, max_hp, max_hp_cost[0], max_hp_std[0]
        else:
            self.logger.info(
                f"End ablation at step {hp_count - len(hp_it) + 1}/{hp_count} "
                f"for budget {budget} (remaining hyperparameters not activate in incumbent or "
                "default configuration)."
            )
            return False, None, None, None
