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

from sklearn.ensemble import (
    RandomForestRegressor,  # TODO: Does that make sense as model?
)
from sklearn.metrics import mean_squared_error

from deepcave.constants import COMBINED_COST_NAME
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
        # A Random Forest Regressor is used as surrogate
        (
            self.default_config,
            incumbent_config,
        ) = self._train_surrogate()  # TODO: Does it make sense to train on only one cs?

        res_default = self._model.predict([self.default])

        print("Default performance:", 1 - res_default)
        print("Incumbent performance:", 1 - self._model.predict([self.incumbent]))

        importances = {}

        # Copy the hps names as to not remove objects from the original list
        hp_it = self.hp_names.copy()
        for i in range(len(hp_it)):
            # Get the results of the current ablation iteration
            continue_ablation, max_hp, max_hp_performance, max_error = self._ablation(
                incumbent_config, res_default, hp_it
            )

            if not continue_ablation:
                print("end ablation")
                break

            print("Hyperparameter with max MSE", max_hp, "New performance:", max_hp_performance)
            # Remove the current max hp for keeping the order right
            hp_it.remove(max_hp)
            importances[max_hp] = (max_hp_performance[0], max_error)

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
        self, incumbent_config: Any, res_default: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any, Any]:
        max_hp = ""
        max_hp_error = 0
        for hp in hp_it:
            if (
                incumbent_config[hp] is not None and hp in self.default_config.keys()
            ):  # Why should it not be in default keys though?
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]
                res = self._model.predict(
                    [self.run.encode_config(config_copy)]
                )  # TODO: Change the variable names
                mse = mean_squared_error(res_default, res)
                if mse > max_hp_error:
                    max_hp = hp
                    max_hp_error = mse  # TODO: Is this really the right way?
            else:
                continue  # TODO: Maybe raise an error here? Does not seem ideal

        if max_hp != "":
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_performance = self._model.predict([self.run.encode_config(self.default_config)])
            return True, max_hp, 1 - max_hp_performance, max_hp_error
        else:
            print(
                "No hyperparameter to ablate: ", max_hp, max_hp_error
            )  # TODO: This needs to be more clear what the problem is
            return False, None, None, None

    def _train_surrogate(self) -> Tuple[Any, Any]:
        # Collect the runs attributes for training the surrogate
        objectives = self.run.get_objectives()
        budget = self.run.get_highest_budget()

        df = self.run.get_encoded_data(objectives, budget, specific=True)
        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df[
            COMBINED_COST_NAME
        ].to_numpy()  # TODO: Change to wanted measure? At least BOHB does not work this way

        # Only get first entry, the normalized cost is not needed
        incumbent_config = self.run.get_incumbent()[0]
        self.incumbent = self.run.encode_config(incumbent_config)

        default_config = (
            self.cs.get_default_configuration()
        )  # TODO: Find a better fit than a random sample?
        self.default = self.run.encode_config(default_config)

        self._model = RandomForestRegressor(
            max_depth=100, random_state=0
        )  # TODO: Change parameters?
        self._model.fit(X, Y)

        return default_config, incumbent_config
