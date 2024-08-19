#  noqa: D400
"""
# LPI

This module provides utilities to calculate the local parameter importance (LPI).

## Classes
    - LPI: This class calculates the local parameter importance (LPI).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.c_util import change_hp_value
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    NumericalHyperparameter,
)
from ConfigSpace.types import Array, f64
from ConfigSpace.util import impute_inactive_values

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


# https://github.com/automl/ParameterImportance/blob/f4950593ee627093fc30c0847acc5d8bf63ef84b/pimp/evaluator/local_parameter_importance.py#L27
class LPI:
    """
    Calculate the local parameter importance (LPI).

    Properties
    ----------
    run : AbstractRun
        The AbstractRun to get the importance from.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        The names of the Hyperparameters.
    variances : Dict[Any, list]
        The overall variances per tree.
    importances : dict
        The importances of the Hyperparameters.
    continuous_neighbors : int
        The number of neighbors chosen for continuous Hyperparameters.
    incumbent : Configuration
        The incumbent of the run.
    default : Configuration
        A configuration containing Hyperparameters with default values.
    incumbent_array : numpy.ndarray
        The internal vector representation of the incumbent.
    seed : int
        The seed. If not provided it will be random.
    rs : RandomState
        A random state with a given seed value.
    """

    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace
        self.hp_names = list(self.cs.keys())
        self.variances: Optional[Dict[Any, List[Any]]] = None
        self.importances: Optional[Dict[Any, Any]] = None

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        continous_neighbors: int = 500,
        n_trees: int = 10,
        seed: int = 0,
    ) -> None:
        """
        Prepare the data and train a RandomForest model.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default, None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default, None. If None, the highest budget is chosen.
        continuous_neighbors : int, optional
            How many neighbors should be chosen for continuous hyperparameters (HPs).
            By default, 500.
        n_trees : int, optional
            The number of trees for the fanova forest.
            Default is 10.
        seed : Optional[int], optional
            The seed. By default None. If None, a random seed is chosen.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        # Set variables
        self.continous_neighbors = continous_neighbors
        self.incumbent, _ = self.run.get_incumbent(budget=budget, objectives=objectives)
        self.default = self.cs.get_default_configuration()
        self.incumbent_array = self.incumbent.get_array()

        self.seed = seed
        self.rs = np.random.RandomState(seed)

        # Get data
        df = self.run.get_encoded_data(
            objectives=objectives, budget=budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Get model and train it
        # Use same forest as for fanova
        self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
        self._model.train(X, Y)

        # Get neighborhood sampled on an unit-hypercube.
        neighborhood = self._get_neighborhood()

        # The delta performance is needed from the default configuration and the incumbent
        def_perf, def_var = self._predict_mean_var(self.default)
        inc_perf, inc_var = self._predict_mean_var(self.incumbent)
        delta = def_perf - inc_perf

        # These are used for plotting and hold the predictions for each neighbor of each parameter.
        # That means performances holds the mean, variances the variance of the forest.
        performances: Dict[str, List[np.ndarray]] = {}
        variances: Dict[str, List[np.ndarray]] = {}
        # These are used for importance and hold the corresponding importance/variance over
        # neighbors. Only import if NOT quantifying importance via performance-variance across
        # neighbors.
        importances = {}
        # Nested list of values per tree in random forest.
        predictions: Dict[str, List[List[np.ndarray]]] = {}

        # Iterate over parameters
        for hp_idx, hp_name in enumerate(self.incumbent.keys()):
            if hp_name not in neighborhood:
                continue

            performances[hp_name] = []
            variances[hp_name] = []
            predictions[hp_name] = []
            incumbent_added = False
            incumbent_idx = 0

            # Iterate over neighbors
            for unit_neighbor, neighbor in zip(neighborhood[hp_name][0], neighborhood[hp_name][1]):
                if not incumbent_added:
                    # Detect incumbent
                    if unit_neighbor > self.incumbent_array[hp_idx]:
                        performances[hp_name].append(inc_perf)
                        variances[hp_name].append(inc_var)
                        incumbent_added = True
                    else:
                        incumbent_idx += 1

                # Create the neighbor-Configuration object
                new_array = self.incumbent_array.copy()
                new_array = change_hp_value(
                    self.cs, new_array, hp_name, unit_neighbor, self.cs.index_of[hp_name]
                )
                new_config = impute_inactive_values(Configuration(self.cs, vector=new_array))

                # Get the leaf values
                x = np.array(new_config.get_array())
                leaf_values = self._model.get_leaf_values(x)

                # And the prediction/performance/variance
                predictions[hp_name].append([np.mean(tree_pred) for tree_pred in leaf_values])
                performances[hp_name].append(np.mean(predictions[hp_name][-1]))
                variances[hp_name].append(np.var(predictions[hp_name][-1]))

            if len(neighborhood[hp_name][0]) > 0:
                neighborhood[hp_name][0] = np.insert(
                    neighborhood[hp_name][0], incumbent_idx, self.incumbent_array[hp_idx]
                )
                neighborhood[hp_name][1] = np.insert(
                    neighborhood[hp_name][1], incumbent_idx, self.incumbent[hp_name]
                )
            else:
                neighborhood[hp_name][0] = np.array(self.incumbent_array[hp_idx])
                neighborhood[hp_name][1] = [self.incumbent[hp_name]]

            if not incumbent_added:
                performances[hp_name].append(inc_perf)
                variances[hp_name].append(inc_var)

            # After all neighbors are estimated, look at all performances except the incumbent
            perf_before = performances[hp_name][:incumbent_idx]
            perf_after = performances[hp_name][incumbent_idx + 1 :]
            tmp_perf = perf_before + perf_after

            # Avoid division by zero
            if delta == 0:
                delta = 1

            imp_over_mean = (np.mean(tmp_perf) - performances[hp_name][incumbent_idx]) / delta
            imp_over_median = (np.median(tmp_perf) - performances[hp_name][incumbent_idx]) / delta
            imp_over_max = (np.max(tmp_perf) - performances[hp_name][incumbent_idx]) / delta

            importances[hp_name] = np.array([imp_over_mean, imp_over_median, imp_over_max])

        # Creating actual importance value (by normalizing over sum of vars)
        num_trees = len(list(predictions.values())[0][0])
        hp_names = list(performances.keys())

        overall_var_per_tree = {}
        for hp_name in hp_names:
            hp_variances = []
            for tree_idx in range(num_trees):
                variance = np.var([neighbor[tree_idx] for neighbor in predictions[hp_name]])
                hp_variances += [variance]

            overall_var_per_tree[hp_name] = hp_variances

        # Sum up variances per tree across parameters
        sum_var_per_tree = [
            sum([overall_var_per_tree[hp_name][tree_idx] for hp_name in hp_names])
            for tree_idx in range(num_trees)
        ]

        # Normalize
        overall_var_per_tree = {
            p: [
                t / sum_var_per_tree[idx] if sum_var_per_tree[idx] != 0.0 else np.nan
                for idx, t in enumerate(trees)
            ]
            for p, trees in overall_var_per_tree.items()
        }
        self.variances = overall_var_per_tree
        self.importances = importances

    def get_importances(self, hp_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Return the importances.

        Parameters
        ----------
        hp_names : List[str]
            Selected Hyperparameter names to get the importance scores from.

        Returns
        -------
        importances : Dict[str, Tuple[float, float]]
            Hyperparameter name and mean+var importance.

        Raises
        ------
        RuntimeError
            If the important scores are not calculated.
        """
        if self.importances is None or self.variances is None:
            raise RuntimeError("Importance scores must be calculated first.")

        importances: Dict[str, Tuple[float, float]] = {}
        for hp_name in hp_names:
            mean = 0
            std = 0

            if hp_name in self.importances:
                mean = np.nanmean(self.variances[hp_name])
                std = np.nanvar(self.variances[hp_name])

            # Use this to quantify importance via importance over mean value (not normalized to 1)
            # mean = self.importances[hp_name][0]

            # Sometimes there is an ugly effect if default is better than
            # incumbent.
            if mean < 0:
                mean = 0
                std = 0

            importances[hp_name] = (mean, std)

        return importances

    def _get_neighborhood(self) -> Dict[str, List[Union[np.ndarray, List[np.ndarray]]]]:
        """
        Slight modification of ConfigSpace's get_one_exchange neighborhood.

        This orders the parameter values and samples more neighbors in one go.
        Further each and every neighbor needs to be rigorously checked if it is forbidden or not.

        Returns
        -------
        neighborhood : Dict[str, List[Union[np.ndarray, List[np.ndarray]]]]
            The neighborhood.
        """
        hp_names = list(self.cs.keys())

        neighborhood: Dict[str, List[Union[np.ndarray, List[np.ndarray]]]] = {}
        for hp_idx, hp_name in enumerate(hp_names):
            # Check if hyperparameter is active
            if not np.isfinite(self.incumbent_array[hp_idx]):
                continue

            hp_neighborhood = []
            checked_neighbors = []  # On unit cube
            checked_neighbors_non_unit_cube = []  # Not on unit cube
            hp = self.cs[hp_name]
            num_neighbors = hp.get_num_neighbors(self.incumbent[hp_name])

            neighbors: Union[List[Union[f64]], Array[Union[f64]]]

            if num_neighbors == 0:
                continue
            elif np.isinf(num_neighbors):
                assert isinstance(hp, NumericalHyperparameter)
                if hp.log:
                    base = np.e
                    log_lower = np.log(hp.lower) / np.log(base)
                    log_upper = np.log(hp.upper) / np.log(base)
                    neighbors_range = np.logspace(
                        start=log_lower,
                        stop=log_upper,
                        num=self.continous_neighbors,
                        endpoint=True,
                        base=base,
                    )
                else:
                    neighbors_range = np.linspace(hp.lower, hp.upper, self.continous_neighbors)
                neighbors = list(map(lambda x: hp.to_vector(x), neighbors_range))
            else:
                neighbors = hp.neighbors_vectorized(self.incumbent_array[hp_idx], n=4, seed=self.rs)

            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    continue

                new_array = self.incumbent_array.copy()
                new_array = change_hp_value(self.cs, new_array, hp_name, neighbor, hp_idx)

                try:
                    new_config = Configuration(self.cs, vector=new_array)
                    hp_neighborhood.append(new_config)
                    new_config.check_valid_configuration()
                    self.cs.check_configuration_vector_representation(new_array)

                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(new_config[hp_name])
                except (ForbiddenValueError, ValueError):
                    pass

            sort_idx = list(
                map(lambda x: x[0], sorted(enumerate(checked_neighbors), key=lambda y: y[1]))
            )
            if isinstance(self.cs[hp_name], CategoricalHyperparameter):
                checked_neighbors_non_unit_cube_categorical = list(
                    np.array(checked_neighbors_non_unit_cube)[sort_idx]
                )
                neighborhood[hp_name] = [
                    np.array(checked_neighbors)[sort_idx],
                    checked_neighbors_non_unit_cube_categorical,
                ]
            else:
                checked_neighbors_non_unit_cube_non_categorical = np.array(
                    checked_neighbors_non_unit_cube
                )[sort_idx]
                neighborhood[hp_name] = [
                    np.array(checked_neighbors)[sort_idx],
                    checked_neighbors_non_unit_cube_non_categorical,
                ]

        return neighborhood

    def _predict_mean_var(self, config: Configuration) -> Tuple[np.ndarray, np.ndarray]:
        """
        Small wrapper to predict marginalized over instances.

        Parameter
        ---------
        config:Configuration
            The self.incumbent of which the performance across the whole instance set is to be
            estimated.

        Returns
        -------
        mean: np.ndarray
            The mean performance over the instance set.
        var: np.ndarray
            The variance over the instance set. If logged values are used, the variance might not
            be able to be used.
        """
        config = impute_inactive_values(config)
        array = np.array([config.get_array()])
        mean, var = self._model.predict_marginalized(array)

        return mean.squeeze(), var.squeeze()
