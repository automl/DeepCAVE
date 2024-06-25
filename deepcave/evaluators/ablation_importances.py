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

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


class AblationImportances:
    """Provide an evaluator of the ablation importances."""

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = self.cs.get_hyperparameter_names()
        self.importances: Optional[Dict[Any, Any]] = None

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        continous_neighbors: int = 500,  # noqa
        n_trees: int = 10,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """Prepare the data for processing and train a Random Forest surrogate model."""
        self.objectives = objectives

        # random_seed = random.randint(0, 100)  # TODO: 100 ok? Random ok?
        importances = dict()

        # A Random Forest Regressor is used as surrogate
        (incumbent_config, incumbent_encode) = self._train_surrogate(seed, budget)

        cost_mean_def, _ = self._model.predict(np.array([self.default_encode]))
        cost_mean_inc, _ = self._model.predict(np.array([incumbent_encode]))

        def_performance = 1 - cost_mean_def
        inc_performance = 1 - cost_mean_inc

        print("Default performance:", def_performance)
        print("Incumbent performance:", inc_performance)

        if inc_performance < def_performance:
            print("Inc is smaller than default for budget: ", budget)
            importances = {hp_name: (0, 0) for hp_name in self.hp_names}
            # TODO: Display warning here

        else:
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()
            for i in range(len(hp_it)):
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_performance, max_hp_var = self._ablation(
                    incumbent_config, def_performance, hp_it
                )

                if not continue_ablation:
                    print("end ablation")
                    break

                print(
                    "Hyperparameter with max impact", max_hp, "New performance:", max_hp_performance
                )
                # TODO: Change the variance
                importances[max_hp] = (max_hp_performance[0] - def_performance[0], max_hp_var[0])
                # New the 'default' performance
                def_performance = max_hp_performance
                # Remove the current max hp for keeping the order right
                hp_it.remove(max_hp)

        importances["sort"] = (-1, -1)
        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Optional[Dict[Any, Any]]:
        """I am a placeholder."""
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")
        # The ranks aka the order of the hps have to be determined
        # importances = dict(sorted(self.importances.items(), key=lambda x: x[1][1]))
        # self.importances = {key: value for key, value in sorted(self.importances.items())}
        return self.importances

    def _ablation(
        self, incumbent_config: Any, def_performance: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any, Any]:
        max_hp = ""
        max_hp_difference = -1

        for hp in hp_it:
            if (
                incumbent_config[hp] is not None and hp in self.default_config.keys()
            ):  # Why should it not be in default keys though?
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]
                cost_mean_new, _ = self._model.predict(
                    np.array([self.run.encode_config(config_copy)])
                )  # TODO: Change the variable names
                difference = def_performance - cost_mean_new
                if difference > max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
                # TODO: Maybe raise an error here? Does not seem ideal
        if max_hp != "":
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_mean, max_hp_var = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            return True, max_hp, 1 - max_hp_mean, max_hp_var
        else:
            print(
                "No hyperparameter to ablate: ", max_hp, max_hp_difference
            )  # TODO: This needs to be more clear what the problem is
            return False, None, None, None

    def _train_surrogate(self, seed: int, budget: Union[int, float, None]) -> Tuple[Any, Any]:
        # Collect the runs attributes for training the surrogate
        df = self.run.get_encoded_data(
            self.objectives, budget, specific=True, include_combined_cost=True
        )

        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives (the normalized cost)
        Y = df[COMBINED_COST_NAME].to_numpy()

        print("Budget: ", budget)
        print("Max and min performances of this cs: ", max(Y), min(Y))
        print("Objectives that are passed: ", self.objectives)
        # Only get first entry, the normalized cost is not needed
        incumbent_config = self.run.get_incumbent(budget=budget, objectives=self.objectives)[0]
        incumbent_encode = self.run.encode_config(incumbent_config)

        # If there is no default config, create one fitting to the cs
        self.default_config = self.cs.get_default_configuration()
        self.default_encode = self.run.encode_config(self.default_config)

        self._model = RandomForestSurrogate(self.cs, seed=seed, n_trees=50)
        self._model._fit(X, Y)

        return incumbent_config, incumbent_encode
