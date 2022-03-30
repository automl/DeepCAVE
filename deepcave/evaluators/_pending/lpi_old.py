import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from bokeh.models import Panel, Row, Tabs
from tqdm import tqdm


class LPI(AbstractEvaluator):

    """
    Implementation of Ablation via surrogates
    """

    def __init__(
        self,
        scenario,
        cs,
        model,
        to_evaluate: int,
        incumbent=None,
        continous_neighbors=500,
        quant_var=True,
        **kwargs,
    ):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)

        self.incumbent = incumbent
        self.incumbent_dict = self.incumbent.get_dictionary()
        self.continous_neighbors = continous_neighbors
        self.neighborhood_dict = None
        self.performance_dict = {}
        self.sampled_neighbors = 0
        self.variance_dict = {}
        self.quantify_importance_via_variance = quant_var
        self.evaluated_parameter_importance_uncertainty = OrderedDict()

    def _get_one_exchange_neighborhood_by_parameter(self):
        """
        Slight modification of ConfigSpace's get_one_exchange neighborhood. This orders the parameter values and samples
        more neighbors in one go. Further we need to rigorously check each and every neighbor if it is forbidden or not.
        """
        neighborhood_dict = {}
        params = list(self.incumbent.keys())
        self.logger.debug("params: " + str(params))
        for index, param in enumerate(params):
            self.logger.info("Sampling neighborhood of %s" % param)
            array = self.incumbent.get_array()

            if not np.isfinite(array[index]):
                self.logger.info(">".join(["-" * 50, " Not active!"]))
                continue

            neighbourhood = []
            checked_neighbors = []
            checked_neighbors_non_unit_cube = []
            hp = self.incumbent.configuration_space.get_hyperparameter(param)
            num_neighbors = hp.get_num_neighbors(self.incumbent.get(param))
            self.logger.debug("\t" + str(num_neighbors))
            if num_neighbors == 0:
                self.logger.debug("\tNo neighbors!")
                continue
            elif np.isinf(num_neighbors):  # Continous Parameters
                if hp.log:
                    base = np.e
                    log_lower = np.log(hp.lower) / np.log(base)
                    log_upper = np.log(hp.upper) / np.log(base)
                    neighbors = np.logspace(
                        log_lower, log_upper, self.continous_neighbors, endpoint=True, base=base
                    )
                else:
                    neighbors = np.linspace(hp.lower, hp.upper, self.continous_neighbors)
                neighbors = list(map(lambda x: hp._inverse_transform(x), neighbors))
            else:
                neighbors = hp.get_neighbors(array[index], self.rng)
            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    continue
                new_array = array.copy()
                new_array = change_hp_value(
                    self.incumbent.configuration_space, new_array, param, neighbor, index
                )
                try:
                    new_configuration = Configuration(
                        self.incumbent.configuration_space, vector=new_array
                    )
                    neighbourhood.append(new_configuration)
                    new_configuration.is_valid_configuration()
                    check_forbidden(self.cs.forbidden_clauses, new_array)
                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(new_configuration[param])
                except (ForbiddenValueError, ValueError) as e:
                    pass

            self.logger.info(
                ">".join(["-" * 50, " Found {:>3d} valid neighbors".format(len(checked_neighbors))])
            )
            self.sampled_neighbors += len(checked_neighbors) + 1
            sort_idx = list(
                map(lambda x: x[0], sorted(enumerate(checked_neighbors), key=lambda y: y[1]))
            )
            if isinstance(self.cs.get_hyperparameter(param), CategoricalHyperparameter):
                checked_neighbors_non_unit_cube = list(
                    np.array(checked_neighbors_non_unit_cube)[sort_idx]
                )
            else:
                checked_neighbors_non_unit_cube = np.array(checked_neighbors_non_unit_cube)[
                    sort_idx
                ]
            neighborhood_dict[param] = [
                np.array(checked_neighbors)[sort_idx],
                checked_neighbors_non_unit_cube,
            ]

        return neighborhood_dict

    def _predict_over_instance_set(self, config):
        """
        Small wrapper to predict marginalized over instances
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
        mean, var = self.model.predict_marginalized_over_instances(np.array([config.get_array()]))
        return mean.squeeze(), var.squeeze()

    def run(self) -> OrderedDict:
        """
        Main function.
        Returns
        -------
        evaluated_parameter_importance:OrderedDict
            Parameter -> importance. The order is important as smaller indices indicate higher importance
        """
        neighborhood_dict = (
            self._get_one_exchange_neighborhood_by_parameter()
        )  # sampled on a unit-hypercube!
        self.neighborhood_dict = neighborhood_dict
        incumbent_array = self.incumbent.get_array()
        def_perf, def_var = self._predict_over_instance_set(
            impute_inactive_values(self.cs.get_default_configuration())
        )
        inc_perf, inc_var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
        delta = def_perf - inc_perf
        evaluated_parameter_importance = {}

        # These are used for plotting and hold the predictions for each neighbor of each parameter
        # That means performance_dict holds the mean, variance_dict the variance of the forest
        performance_dict = {}
        variance_dict = {}
        # This are used for importance and hold the corresponding importance/variance over neighbors
        # Only import if NOT quantifying importance via performance-variance across neighbours
        overall_imp = {}
        # Nested list of values per tree in random forest
        pred_per_tree = {}

        # Iterate over parameters
        pbar = tqdm(range(self.sampled_neighbors), ascii=True, disable=not self.verbose)
        for index, param in enumerate(self.incumbent.keys()):
            if param not in neighborhood_dict:
                pbar.set_description("{: >.70s}".format("Parameter %s is inactive" % param))
                continue

            pbar.set_description("Predicting performances for neighbors of {: >.30s}".format(param))
            performance_dict[param] = []
            variance_dict[param] = []
            pred_per_tree[param] = []
            added_inc = False
            inc_at = 0
            # Iterate over neighbors
            for unit_neighbor, neighbor in zip(
                neighborhood_dict[param][0], neighborhood_dict[param][1]
            ):
                if not added_inc:
                    # Detect incumbent
                    if unit_neighbor > incumbent_array[index]:
                        performance_dict[param].append(inc_perf)
                        variance_dict[param].append(inc_var)
                        pbar.update(1)
                        added_inc = True
                    else:
                        inc_at += 1
                # self.logger.debug('%s -> %s' % (self.incumbent[param], neighbor))
                # Create the neighbor-Configuration object
                new_array = incumbent_array.copy()
                new_array = change_hp_value(
                    self.incumbent.configuration_space, new_array, param, unit_neighbor, index
                )
                new_configuration = impute_inactive_values(
                    Configuration(self.incumbent.configuration_space, vector=new_array)
                )
                # Predict performance
                x = np.array(new_configuration.get_array())
                pred_per_tree[param].append(
                    [np.mean(tree_pred) for tree_pred in self.model.rf.all_leaf_values(x)]
                )
                # self.logger.debug("Pred per tree: %s", str(pred_per_tree[param][-1]))
                performance_dict[param].append(np.mean(pred_per_tree[param][-1]))
                variance_dict[param].append(np.var(pred_per_tree[param][-1]))

                pbar.update(1)
            if len(neighborhood_dict[param][0]) > 0:
                neighborhood_dict[param][0] = np.insert(
                    neighborhood_dict[param][0], inc_at, incumbent_array[index]
                )
                neighborhood_dict[param][1] = np.insert(
                    neighborhood_dict[param][1], inc_at, self.incumbent[param]
                )
            else:
                neighborhood_dict[param][0] = np.array(incumbent_array[index])
                neighborhood_dict[param][1] = [self.incumbent[param]]
            if not added_inc:
                mean, var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
                performance_dict[param].append(mean)
                variance_dict[param].append(var)
                pbar.update(1)
            # After all neighbors are estimated, look at all performances except the incumbent
            tmp_perf = performance_dict[param][:inc_at] + performance_dict[param][inc_at + 1 :]
            if delta == 0:
                delta = 1  # To avoid division by zero
            imp_over_mea = (np.mean(tmp_perf) - performance_dict[param][inc_at]) / delta
            imp_over_med = (np.median(tmp_perf) - performance_dict[param][inc_at]) / delta
            try:
                imp_over_max = (np.max(tmp_perf) - performance_dict[param][inc_at]) / delta
            except ValueError:
                imp_over_max = np.nan  # Hacky fix as this is never used anyway
            overall_imp[param] = np.array([imp_over_mea, imp_over_med, imp_over_max])

        # Creating actual importance value (by normalizing over sum of vars)
        num_trees = len(list(pred_per_tree.values())[0][0])
        params = list(performance_dict.keys())
        overall_var_per_tree = {
            param: [
                np.var([neighbor[tree_idx] for neighbor in pred_per_tree[param]])
                for tree_idx in range(num_trees)
            ]
            for param in params
        }
        # Sum up variances per tree across parameters
        sum_var_per_tree = [
            sum([overall_var_per_tree[param][tree_idx] for param in params])
            for tree_idx in range(num_trees)
        ]
        # Normalize
        overall_var_per_tree = {
            p: [t / sum_var_per_tree[idx] for idx, t in enumerate(trees)]
            for p, trees in overall_var_per_tree.items()
        }
        self.logger.debug(
            "overall_var_per_tree %s (%d trees)",
            str(overall_var_per_tree),
            len(list(pred_per_tree.values())[0][0]),
        )
        self.logger.debug(
            "sum_var_per_tree %s (%d trees)",
            str(sum_var_per_tree),
            len(list(pred_per_tree.values())[0][0]),
        )
        for param in performance_dict.keys():
            if self.quantify_importance_via_variance:
                evaluated_parameter_importance[param] = np.mean(overall_var_per_tree[param])
            else:
                evaluated_parameter_importance[param] = overall_imp[param][0]

        only_show = sorted(
            list(evaluated_parameter_importance.keys()),
            key=lambda p: evaluated_parameter_importance[p],
        )[: min(self.to_evaluate, len(evaluated_parameter_importance.keys()))]

        self.neighborhood_dict = neighborhood_dict
        self.performance_dict = performance_dict
        self.variance_dict = variance_dict
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
