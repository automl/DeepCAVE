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
import random

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective

# TODO: Fix documentation & type annotation
# TODO: All variables used sensibly?
# TODO: What if inc is larger than sample (due to the prediction being off)
# TODO: None Werte abfangen


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

        # Will later contain all results per iteration and is used for averaging
        self.it_dict: Dict[str, float] = {}
        # Average over ten iteratins
        for i in range(10):
            random_seed = random.randint(0, 100)  # TODO: 100 ok?
            # A Random Forest Regressor is used as surrogate
            (incumbent_config, incumbent_encode) = self._train_surrogate(random_seed, budget)
            # TODO: Does it make sense to train on only one cs? Yes but average it at the end,
            # take the abweichung & use for the graphic as well

            cost_mean_def, _ = self._model.predict(np.array([self.sample_encode]))
            cost_mean_inc, _ = self._model.predict(np.array([incumbent_encode]))

            print("Default performance:", 1 - cost_mean_def)
            print("Incumbent performance:", 1 - cost_mean_inc)

            importances = {}
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()

            for j in range(len(hp_it)):
                print("HP IT COPY: ", hp_it)
                print("HP LIST ORIGINAL: ", self.hp_names)
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_performance = self._ablation(
                    incumbent_config, cost_mean_def, hp_it
                )

                if not continue_ablation:
                    print("end ablation")
                    break

                print(
                    "Hyperparameter with max impact", max_hp, "New performance:", max_hp_performance
                )
                # Remove the current max hp for keeping the order right
                print("MAX BEFORE: ", hp_it, max_hp)
                hp_it.remove(max_hp)
                print("MAX AFTER: ", hp_it, max_hp)
                # TODO: Change to actual variance after averaging later
                importances[max_hp] = (max_hp_performance[0], 0)

            # Now average the results
            print("Round: ", i, " Importances dict: ", importances)
        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Optional[Dict[Any, Any]]:
        """I am a placeholder."""
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")
        importances = {key: value for key, value in sorted(self.importances.items())}
        return importances

    def _ablation(
        self, incumbent_config: Any, cost_mean_def: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any]:
        max_hp = ""
        max_hp_difference = 0

        for hp in hp_it:
            if (
                incumbent_config[hp] is not None and hp in self.sample_config.keys()
            ):  # Why should it not be in default keys though?
                config_copy = copy.copy(self.sample_config)
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
            self.sample_config[max_hp] = incumbent_config[max_hp]
            max_hp_mean, _ = self._model.predict(
                np.array([self.run.encode_config(self.sample_config)])
            )
            print("MAX HP ABL: ", max_hp)
            return True, max_hp, 1 - max_hp_mean
        else:
            print(
                "No hyperparameter to ablate: ", max_hp, max_hp_difference
            )  # TODO: This needs to be more clear what the problem is
            return False, None, None

    def _train_surrogate(self, seed: int, budget: Union[int, float, None]) -> Tuple[Any, Any]:
        # Collect the runs attributes for training the surrogate
        df = self.run.get_encoded_data(
            self.objectives, budget, specific=True, include_combined_cost=True
        )

        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives (the normalized cost)
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Only get first entry, the normalized cost is not needed
        incumbent_config = self.run.get_incumbent(budget=budget)[0]
        incumbent = self.run.encode_config(incumbent_config)
        print("ACTUAL INCUMBENT: ", 1 - self.run.get_incumbent(budget=budget)[1])

        # Sample a configuration from the cs to use as a starter
        self.sample_config = self.cs.sample_configuration()
        self.sample_encode = self.run.encode_config(self.sample_config)
        # sample_encode = self.run.encode_config(sample_config)

        # TODO: Change parameters?
        self._model = RandomForestSurrogate(self.cs, seed=seed)
        self._model._fit(X, Y)

        return incumbent_config, incumbent
