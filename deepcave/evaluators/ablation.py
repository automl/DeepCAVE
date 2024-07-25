# noqa: D400
"""
# Ablation Importances

This module evaluates the ablation importance.

The ablation method determines the parameter importances between two configurations.
One being the default configuration and one the incumbent.

## Classes:
    - AblationImportances: Provide an evaluator of the ablation importances.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import copy
from collections import OrderedDict

import numpy as np

from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


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

        importances: OrderedDict = OrderedDict()

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
        def_cost, _ = self._model.predict(np.array([default_encode]))
        inc_cost, _ = self._model.predict(np.array([incumbent_encode]))

        # For further calculations, assume that the objective is to be minimized
        if objective.optimize == "upper":
            def_cost = -def_cost
            inc_cost = -inc_cost

        if inc_cost > def_cost:
            print(
                "The predicted incumbent cost is smaller than the predicted default "
                f"cost for budget: {budget}. This could mean that the configuration space "
                "with which the surrogate model was trained contained too few examples."
            )
            importances = OrderedDict({hp_name: (0, 0) for hp_name in self.hp_names})
        else:
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()
            for i in range(len(hp_it)):
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_cost, max_hp_var = self._ablation(
                    objective, budget, incumbent_config, def_cost, hp_it
                )

                if not continue_ablation:
                    break

                # As only one objective is allowed, there is only one objective in the list
                if objective.optimize == "upper":
                    # For returning the importance, flip back the objective if it was flipped before
                    importances[max_hp] = (-max_hp_cost[0], max_hp_var[0])
                else:
                    importances[max_hp] = (max_hp_cost[0], max_hp_var[0])
                # New 'default' cost
                def_cost = max_hp_cost
                # Remove the current best hp for keeping the order right
                hp_it.remove(max_hp)

        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Optional[Dict[Any, Any]]:
        """
        Get the importances.

        Parameters
        ----------
        hp_names : List[str]
            A list of the hp names.

        Returns
        -------
        Optional[Dict[Any, Any]]
            A dictionary containing the importance.

        Raises
        ------
        RuntimeError
            If the importance score have not been calculated.
        """
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")
        return self.importances

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
            The current cost.
        hp_it: List[str]
            A list of the HPs that still have to be looked at.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            continue_ablation, max_hp, max_hp_performance, max_hp_var
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
                print(
                    "Warning: No improvement found in ablation step "
                    f"{hp_count - len(hp_it) + 1}/{hp_count} for budget {budget}, "
                    "choose hyperparameter with smallest increase in cost."
                )
            # For the maximum impact hyperparameter, switch the default with the incumbent value
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_cost, max_hp_var = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            if objective.optimize == "upper":
                max_hp_cost = -max_hp_cost
            return True, max_hp, max_hp_cost, max_hp_var
        else:
            print(
                f"Warning: End ablation at step {hp_count - len(hp_it) + 1}/{hp_count} "
                f"for budget {budget} (remaining hyperparameters not activate in incumbent or "
                "default configuration)."
            )
            return False, None, None, None
