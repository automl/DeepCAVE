# noqa: D400
"""
# Ablation Importances

This module provides a plugin for calculating the ablation importance.

The ablation method determines the parameter importances between two configurations.

## Classes:
    - AblationImportances: Provide a plugin for the visualization of the ablation importances.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import copy

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective

# TODO: Fix documentation & type annotation


class AblationImportances:
    """Provide a plugin for the visualization of the ablation importances."""

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = self.cs.get_hyperparameter_names()
        self.importances: Optional[Dict[Any, Any]] = None
        # TODO: What makes sense to init here?

    def calculate(  # TODO: Change head
        # TODO: Iterate a few times more (how often?) and average?
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        continous_neighbors: int = 500,  # noqa
        n_trees: int = 10,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """Prepare the data for processing and train a Random Forest surrogate model."""
        self.objectives = objectives
        self.budget = budget
        self.rs = np.random.RandomState(seed)
        # A Random Forest Regressor is used as surrogate
        (
            self.default_config,
            incumbent_config,
        ) = self._train_surrogate(
            seed
        )  # TODO: Does it make sense to train on only one cs? Yes but average it at the end,
        # take the abweichung & use for the graphic as well
        cost_mean_def, _ = self._model.predict(np.array([self.default_encode]))
        cost_mean_inc, _ = self._model.predict(np.array([self.incumbent]))

        print("Default performance:", 1 - cost_mean_def)
        print("Incumbent performance:", 1 - cost_mean_inc)

        importances = {}
        # Copy the hps names as to not remove objects from the original list
        hp_it = self.hp_names.copy()
        for i in range(len(hp_it)):
            # Get the results of the current ablation iteration
            continue_ablation, max_hp, max_hp_performance, max_hp_variance = self._ablation(
                incumbent_config, cost_mean_def, hp_it
            )

            if not continue_ablation:
                print("end ablation")
                break

            print("Hyperparameter with max impact", max_hp, "New performance:", max_hp_performance)
            # Remove the current max hp for keeping the order right
            hp_it.remove(max_hp)
            importances[max_hp] = (max_hp_performance[0], max_hp_variance[0])
            print(importances)
        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Optional[Dict[Any, Any]]:
        """I am a placeholder."""
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")
        importances = {
            key: value for key, value in sorted(self.importances.items())
        }  # Why did i sort this?
        return importances

    def _ablation(
        self, incumbent_config: Any, cost_mean_def: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any, Any]:
        # TODO: Check objectives for lower or upper -> only use hps that make a positive impact
        max_hp = ""
        max_hp_difference = 0

        # It is important to check whether we want lower or upper
        # optimize = self.objectives.optimize
        for hp in hp_it:
            if (
                incumbent_config[hp] is not None and hp in self.default_config.keys()
            ):  # Why should it not be in default keys though?
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]
                cost_mean_new, _ = self._model.predict(
                    np.array([self.run.encode_config(config_copy)])
                )  # TODO: Change the variable names
                difference = cost_mean_def - cost_mean_new
                if difference > max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
                # TODO: Maybe raise an error here? Does not seem ideal
        if max_hp != "":
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_mean, max_hp_variance = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            return True, max_hp, 1 - max_hp_mean, max_hp_variance
        else:
            print(
                "No hyperparameter to ablate: ", max_hp, max_hp_difference
            )  # TODO: This needs to be more clear what the problem is
            return False, None, None, None

    def _train_surrogate(self, seed: int) -> Tuple[Any, Any]:
        # Collect the runs attributes for training the surrogate
        df = self.run.get_encoded_data(
            self.objectives, self.budget, specific=True, include_combined_cost=True
        )

        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives (the normalized cost)
        Y = df[COMBINED_COST_NAME].to_numpy()  # TODO: Change to wanted measure,
        # i think choosing diff objectives as user input, calculates the same

        # Only get first entry, the normalized cost is not needed
        incumbent_config = self.run.get_incumbent()[0]
        self.incumbent = self.run.encode_config(incumbent_config)

        default_config = (
            self.cs.get_default_configuration()
        )  # TODO: Find a better fit than a random sample
        self.default_encode = self.run.encode_config(default_config)

        # TODO: Change parameters?
        self._model = RandomForestSurrogate(self.cs, seed=seed)
        self._model._fit(X, Y)

        return default_config, incumbent_config
