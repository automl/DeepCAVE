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
    """
    Provide an evaluator of the ablation importances.

    Properties
    ----------
    run : AbstractRun
        The run(s) to analyze.
    cs : ConfigurationSpace
        The configuration space of the run(s).
    hp_names : List[str]
        A list of the hyperparameter names.
    importances : Optional[Dict[Any, Any]]
        A dictionary containing the importances for each HP.
    objectives : Optional[Union[Objective, List[Objective]]]
        The objective(s) of the run(s).
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
        objectives: Optional[Union[Objective, List[Objective]]] = None,  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        continous_neighbors: int = 500,  # noqa
        n_trees: int = 50,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """
        Prepare the data for processing and train a Random Forest surrogate model.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]]
            The objective(s) of the run(s).
            Default is None.
        budget : Optional[Union[int, float]]
            The budget of the run(s).
            Default is None
        n_trees : int
            The number of trees for the surrogate model.
            Default is 50.
        seed : int
            The seed for the surrogate model.
            Default is 0.

        Note
        ----
        continous_neighbors will not be used.
        """
        self.objectives = objectives

        importances = dict()

        # A Random Forest Regressor is used as surrogate
        (incumbent_config, incumbent_encode) = self._train_surrogate(seed, budget, n_trees)

        cost_mean_def, _ = self._model.predict(np.array([self.default_encode]))
        cost_mean_inc, _ = self._model.predict(np.array([incumbent_encode]))

        def_performance = 1 - cost_mean_def
        inc_performance = 1 - cost_mean_inc

        # This is for sorting purposes in the 'importance' plugin
        importances["sort"] = (0, 0)

        if inc_performance < def_performance:
            print(
                "The predicted incumbent performance is smaller than the predicted "
                "default performance for budget: ",
                budget,
                ". This could mean that the configuration space which with the surrogate "
                "model was trained is too small.",
            )
            importances = {hp_name: (0, 0) for hp_name in self.hp_names}
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

                importances[max_hp] = (max_hp_performance[0] - def_performance[0], max_hp_var[0])
                # New 'default' performance
                def_performance = max_hp_performance
                # Remove the current max hp for keeping the order right
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
        self.importances = {
            key: self.importances[key]
            for key in hp_names
            if key in self.importances or key == "sort"
        }
        return self.importances

    def _ablation(
        self, incumbent_config: Any, def_performance: Any, hp_it: List[str]
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Calculate the ablation importance for each hyperparameter.

        Parameters
        ----------
        incumbent_config: Any
            The incumbent configuration.
        def_performance: Any
            The current performance.
        hp_it: List[str]
            A list of the HPs that still have to be looked at.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            continue_ablation, max_hp, max_hp_performance, max_hp_var
        """
        max_hp = ""
        max_hp_difference = -1

        for hp in hp_it:
            if incumbent_config[hp] is not None and hp in self.default_config.keys():
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]
                cost_mean_new, _ = self._model.predict(
                    np.array([self.run.encode_config(config_copy)])
                )
                difference = def_performance - cost_mean_new
                # Check for the maximum difference hyperparameter in this round
                if difference >= max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
        if max_hp != "":
            # Switch the maximum impact hyperparameter with its default parameter
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_mean, max_hp_var = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            return True, max_hp, 1 - max_hp_mean, max_hp_var
        else:
            print("No maximum impact hyperparameter found: ", max_hp, max_hp_difference)
            return False, None, None, None

    def _train_surrogate(
        self, seed: int, budget: Union[int, float, None], n_trees: int
    ) -> Tuple[Any, Any]:
        """
        Get the data points from the cs and train the surrogate.

        Parameters
        ----------
        seed: int
            The seed for the model.
        budget: Union[int, float, None]
            The budget of the run(s).
        n_trees: int
            The number of trees for the Random Forest.

        Returns
        -------
        Tuple[Any, Any]
            incumbent_config, incumbent_encode
        """
        # Collect the runs attributes for training the surrogate
        df = self.run.get_encoded_data(
            self.objectives, budget, specific=True, include_combined_cost=True
        )

        X = df[self.run.configspace.get_hyperparameter_names()].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Only get first entry, the normalized cost is not needed
        incumbent_config = self.run.get_incumbent(budget=budget, objectives=self.objectives)[0]
        incumbent_encode = self.run.encode_config(incumbent_config)

        self.default_config = self.cs.get_default_configuration()
        self.default_encode = self.run.encode_config(self.default_config)

        self._model = RandomForestSurrogate(self.cs, seed=seed, n_trees=n_trees)
        self._model._fit(X, Y)

        return incumbent_config, incumbent_encode
