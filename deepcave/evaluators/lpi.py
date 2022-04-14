from collections import OrderedDict
import numpy as np

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
    OrdinalHyperparameter,
    NumericalHyperparameter,
)
from ConfigSpace.util import (
    impute_inactive_values,
    get_random_neighbor,
    get_one_exchange_neighbourhood,
)
from ConfigSpace.c_util import change_hp_value, check_forbidden

from deepcave.runs import AbstractRun


# https://github.com/automl/ParameterImportance/blob/f4950593ee627093fc30c0847acc5d8bf63ef84b/pimp/evaluator/local_parameter_importance.py#L27
class LPI:
    def __init__(self, run: AbstractRun):
        self.run = run
        self.cs = run.configspace

    def run(self, budget, continous_neighbors=500, quantify_importance_via_variance=False) -> None:
        # TODO: Set and train model

        self.continous_neighbors = continous_neighbors
        self.incumbent, _ = self.run.get_incumbent(budget=budget)
        self.default = self.cs.get_default_configuration()
        self.incumbent_array = self.incumbent.get_array()

        # Get neighborhood sampled on an unit-hypercube.
        neighborhood = self._get_neighborhood()

        # We need the delta performance from the default configuration and the incumbent
        def_perf, def_var = self._predict_mean_var(self.default)
        inc_perf, inc_var = self._predict_mean_var(self.incumbent)
        delta = def_perf - inc_perf
        evaluated_parameter_importance = {}

        # These are used for plotting and hold the predictions for each neighbor of each parameter.
        # That means performances holds the mean, variances the variance of the forest.
        performances = {}
        variances = {}
        # This are used for importance and hold the corresponding importance/variance over neighbors.
        # Only import if NOT quantifying importance via performance-variance across neighbours.
        importances = {}
        # Nested list of values per tree in random forest.
        predictions = {}

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
                new_array = change_hp_value(self.cs, new_array, hp_name, unit_neighbor, hp_idx)
                new_config = impute_inactive_values(Configuration(self.cs, vector=new_array))

                # Get the leaf values
                x = np.array(new_config.get_array())
                leaf_values = self.model.rf.all_leaf_values(x)

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
            p: [t / sum_var_per_tree[idx] for idx, t in enumerate(trees)]
            for p, trees in overall_var_per_tree.items()
        }

        for hp_name in performances.keys():
            if self.quantify_importance_via_variance:
                evaluated_parameter_importance[hp_name] = np.mean(overall_var_per_tree[hp_name])
            else:
                evaluated_parameter_importance[hp_name] = importances[hp_name][0]

        only_show = sorted(
            list(evaluated_parameter_importance.keys()),
            key=lambda p: evaluated_parameter_importance[p],
        )[: min(self.to_evaluate, len(evaluated_parameter_importance.keys()))]

        self.neighborhood = neighborhood
        self.performances = performances
        self.variances = variances
        self.evaluated_parameter_importance = OrderedDict(
            [(p, evaluated_parameter_importance[p]) for p in only_show]
        )

        if self.quantify_importance_via_variance:
            self.evaluated_parameter_importance_uncertainty = OrderedDict(
                [(p, np.std(overall_var_per_tree[p])) for p in only_show]
            )

        all_res = {
            "imp": self.evaluated_parameter_importance,
            "order": list(self.evaluated_parameter_importance.keys()),
        }

        return all_res

    def _get_neighborhood(self):
        """
        Slight modification of ConfigSpace's get_one_exchange neighborhood. This orders the
        parameter values and samples more neighbors in one go. Further we need to rigorously
        check each and every neighbor if it is forbidden or not.
        """

        hp_names = self.cs.get_hyperparameter_names()

        neighborhood = {}
        for hp_idx, hp_name in enumerate(hp_names):
            # Check if hyperparameter is active
            if not np.isfinite(self.incumbent_array[hp_idx]):
                continue

            hp_neighborhood = []
            checked_neighbors = []  # On unit cube
            checked_neighbors_non_unit_cube = []  # Not on unit cube
            hp = self.configspace.get_hyperparameter(hp_name)
            num_neighbors = hp.get_num_neighbors(self.incumbent[hp_name])

            if num_neighbors == 0:
                continue
            elif np.isinf(num_neighbors):
                if hp.log:
                    base = np.e
                    log_lower = np.log(hp.lower) / np.log(base)
                    log_upper = np.log(hp.upper) / np.log(base)
                    neighbors = np.logspace(
                        start=log_lower,
                        stop=log_upper,
                        num=self.continous_neighbors,
                        endpoint=True,
                        base=base,
                    )
                else:
                    neighbors = np.linspace(hp.lower, hp.upper, self.continous_neighbors)
                neighbors = list(map(lambda x: hp._inverse_transform(x), neighbors))
            else:
                neighbors = hp.get_neighbors(self.incumbent_array[hp_name], self.seed)

            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    continue

                new_array = self.incumbent_array.copy()
                new_array = change_hp_value(self.cs, new_array, hp_name, neighbor, hp_idx)

                try:
                    new_config = Configuration(self.cs, vector=new_array)
                    hp_neighborhood.append(new_config)
                    new_config.is_valid_configuration()
                    check_forbidden(self.cs.forbidden_clauses, new_array)

                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(new_config[hp_name])
                except (ForbiddenValueError, ValueError):
                    pass

            sort_idx = list(
                map(lambda x: x[0], sorted(enumerate(checked_neighbors), key=lambda y: y[1]))
            )
            if isinstance(self.cs.get_hyperparameter(hp_name), CategoricalHyperparameter):
                checked_neighbors_non_unit_cube = list(
                    np.array(checked_neighbors_non_unit_cube)[sort_idx]
                )
            else:
                checked_neighbors_non_unit_cube = np.array(checked_neighbors_non_unit_cube)[
                    sort_idx
                ]

            neighborhood[hp_name] = [
                np.array(checked_neighbors)[sort_idx],
                checked_neighbors_non_unit_cube,
            ]

        return neighborhood

    def _predict_mean_var(self, config):
        """
        Small wrapper to predict marginalized over instances.

        Parameter
        ---------
        config:Configuration
            The self.incumbent of wich the performance across the whole instance set is to be estimated

        Returns
        -------
        mean
            the mean performance over the instance set
        var
            the variance over the instance set. If logged values are used, the variance might not be able to be used
        """

        array = np.array([config.get_array()])
        array = impute_inactive_values(array)
        mean, var = self.model.predict_marginalized_over_instances(array)

        return mean.squeeze(), var.squeeze()
