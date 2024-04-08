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

from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective

# TODO: Fix documentation & type annotation


class AblationImportances:
    """Provide a plugin for the visualization of the ablation importances."""

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = self.cs.get_hyperparameter_names()
        print("INIRT HPS: ", self.hp_names)
        self.importances: Optional[Dict[Any, Any]] = None

    def calculate(  # TODO: Change head
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        continous_neighbors: int = 500,  # noqa
        n_trees: int = 10,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """Prepare the data for processing and train a Random Forest surrogate model."""
        (
            self.default_config,
            incumbent_config,
        ) = self._train_surrogate()  # TODO: Does it make sense to train on only one cs?
        res_default = self._model.predict([self.default])
        print(self.default)

        print("Default performance:", 1 - res_default)
        print("Incumbent performance:", 1 - self._model.predict([self.incumbent]))

        importances = {}
        print("HP NAMES ", self.hp_names)
        hp_it = self.hp_names.copy()
        for i in range(len(hp_it)):
            continue_ablation, max_hp, max_hp_performance, max_error = self._ablation(
                incumbent_config, res_default, hp_it
            )
            if not continue_ablation:
                print("end ablation")
                break
            print("Hyperparameter with max MSE", max_hp, "New performance:", max_hp_performance)
            print("HP IT: ", hp_it)
            hp_it.remove(max_hp)
            print("HP NAMES AFTER", self.hp_names)
            print("MAX PERGOR: ", max_hp_performance)
            importances[max_hp] = (max_hp_performance[0], max_error)

        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Optional[Dict[Any, Any]]:
        """I am a placeholder."""
        if self.importances is None:
            raise RuntimeError("Importance scores must be calculated first.")
        print("ABLI IMPOR: ", self.importances)
        importances = {key: value for key, value in sorted(self.importances.items())}
        print(importances)
        return importances

    def _ablation(
        self, incumbent_config: Any, res_default: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any, Any]:
        max_hp = ""
        max_hp_error = 0
        for hp in hp_it:
            print("HPS: ", hp)
            if incumbent_config[hp] is not None and hp in self.default_config.keys():
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]
                res = self._model.predict([self.run.encode_config(config_copy)])
                mse = mean_squared_error(res_default, res)
                if mse > max_hp_error:
                    max_hp = hp
                    max_hp_error = mse
            else:
                continue
            print("MAX HP: ", max_hp, "MAX HP ERROR: ", max_hp_error)
        if max_hp != "":
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_performance = self._model.predict([self.run.encode_config(self.default_config)])
            return True, max_hp, 1 - max_hp_performance, max_hp_error
        else:
            print("No hyperparameter to ablate: ", max_hp, max_hp_error)
            return False, None, None, None

    def _train_surrogate(self) -> Tuple[Any, Any]:
        objectives = self.run.get_objectives()
        budget = self.run.get_highest_budget()

        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df["Combined Cost"].to_numpy()  # TODO: Change to wanted measure

        incumbent_config = self.run.get_incumbent()[0]
        self.incumbent = self.run.encode_config(incumbent_config)

        default_config = (
            self.cs.get_default_configuration()
        )  # TODO: Find a better fit than a random sample
        print("ABLI DEFAULT CONFIG ", default_config)
        self.default = self.run.encode_config(default_config)

        self._model = RandomForestRegressor(max_depth=100, random_state=0)
        self._model.fit(X, Y)

        return default_config, incumbent_config
