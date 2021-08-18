import numpy as np
import pandas as pd
import copy

import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, Constant, UniformFloatHyperparameter

from smac.runhistory.runhistory import RunHistory
from deep_cave.evaluators.epm.util_funcs import get_types
from smac.tae import StatusType

from deep_cave.utils.mapping import numerical_map_fn, categorical_map_fn


class Run:
    def __init__(self, meta: dict, runhistory: RunHistory, configspace: ConfigSpace):
        """
        meta: start_time, end_time, duration, 
        """

        self.meta = meta
        self.rh = runhistory
        self.cs = configspace

    def get_meta(self):
        return self.meta

    def get_runhistory(self, fidelity=None):
        # TODO: Get only relevant fidelities
        # TODO: merge multiple runs runhistory with update_from_json()

        return self.rh

    def get_configspace(self, hyperparameter_ids=None, remove_inactive=False):
        if hyperparameter_ids is None:
            return self.cs

        selected_hps = self.get_hyperparameters(
            hyperparameter_ids, remove_inactive)

        # Create a new configspace if only specific hyperparameters are selected
        new_cs = copy.deepcopy(self.cs)
        new_cs._hyperparameter_idx = {}
        new_cs._idx_to_hyperparameter = {}
        new_id = 0
        for hp_name, hp in self.cs.get_hyperparameters_dict().items():
            if hp not in selected_hps:
                del new_cs._hyperparameters[hp_name]
                del new_cs._parents[hp_name]
                del new_cs._children[hp_name]

                try:
                    del new_cs._children["__HPOlib_configuration_space_root__"][hp_name]
                except:
                    pass
            else:
                new_cs._hyperparameter_idx[hp_name] = new_id
                new_cs._idx_to_hyperparameter[new_id] = hp_name
                new_id += 1

        new_conditionals = set()
        for hp_name in new_cs._conditionals:
            hp = self.cs.get_hyperparameter(hp_name)
            if hp in selected_hps:
                new_conditionals.add(hp_name)
        new_cs._conditionals = new_conditionals

        new_forbidden_clauses = []
        for clause in new_cs.forbidden_clauses:
            if clause.hyperparameter in selected_hps:
                new_forbidden_clauses.append(clause)

        for hp_name, d in new_cs._children.copy().items():
            for child_hp_name in d.copy().keys():
                child_hp = self.cs.get_hyperparameter(child_hp_name)
                if child_hp not in selected_hps:
                    del new_cs._children[hp_name][child_hp_name]

        new_cs._update_cache()
        new_cs._sort_hyperparameters()
        new_cs._check_default_configuration()

        return new_cs

    def get_incumbent(self, fidelity=None, hyperparameter_ids=None):
        costs = self.costs(fidelity)
        return min(costs, key=costs.get)

    def get_costs(self, fidelities=None, statuses=[StatusType.SUCCESS]):
        """
        Input:
            fidelities (list or float or None)
        """
        results = {}

        configs = []
        for (config_id, _, _, budget), (_, _, status, _, _, _) in self.rh.data.items():
            if fidelities is not None:
                if isinstance(fidelities, list) and budget not in fidelities:
                    continue
                elif fidelities != budget:
                    continue

            if status not in statuses:
                continue

            configs.append(self.rh.ids_config[config_id])

        for config in configs:
            cost = self.rh.get_instance_costs_for_config(config)
            results[config] = list(cost.values())[0]

        return results

    def get_hyperparameters(self, ids, remove_inactive=True):
        """
        Retrieve hyperparameter name/s by id/s.

        Parameters:
            remove_inactive (bool): Childs of not used parents are removed from the list.
        """

        if ids is None:
            return []

        if isinstance(ids, list):
            hps = []
            for i in ids:
                hp = self.cs.get_hyperparameter(
                    self.cs.get_hyperparameter_by_idx(int(i)))
                hps.append(hp)
        else:
            hps = [self.cs.get_hyperparameter(
                self.cs.get_hyperparameter_by_idx(int(ids)))]

        if remove_inactive:
            def check_parents_active(hp, selected_hps: list):
                """
                Recursive method to get if all the parents of a given hyperparameter
                are active.
                """

                parents = self.cs.get_parents_of(hp)
                if len(parents) == 0:
                    if hp in selected_hps:
                        return True
                    else:
                        return False
                else:
                    active = True

                    for parent in parents:
                        parent_active = check_parents_active(
                            parent, selected_hps)
                        if not parent_active:
                            active = False
                            break

                    return active

            new_hps = []

            # Remove all childs of unused parents
            for hp in hps:
                # Can be a higher hierarchy, so we have to check recursively.
                if check_parents_active(hp, hps):
                    new_hps += [hp]

            return new_hps
        else:
            return hps

    def transform_config(self, config, mapping):
        print("---")

        array = []
        print(mapping.keys())
        for hp_name in self.cs.get_hyperparameter_names():

            if hp_name in mapping:
                hp_value = config.get(hp_name)
                array.append(mapping[hp_name](hp_value))

        print(array)
        print(config.get_array())

        return array

    def transform_configs(self,
                          fidelities=None,
                          statuses=[StatusType.SUCCESS],
                          hyperparameter_ids=None,
                          remove_inactive=False):
        """
        Return all configurations numerical encoded and imputed.

        Parameters:
            fidelity (str)
            hyperparameter_selection (list, optional): list of HyperParameter objects.

        Returns:
            X (np.array): 
            y (float): cost
            mapping: for new configs
            id_mapping: 
            types:
            bounds: 
        """

        new_configspace = self.cs
        selected_hps = None
        if hyperparameter_ids is not None:
            selected_hps = self.get_hyperparameters(
                hyperparameter_ids, remove_inactive)
            new_configspace = self.get_configspace(
                hyperparameter_ids, remove_inactive)

        hyperparameters = self.cs.get_hyperparameters_dict()
        cs = self.cs

        types, bounds = get_types(cs)

        X, y = {}, []
        mapping = {}
        categorical_mapping = {}

        results = self.get_costs(fidelities, statuses)
        for config, cost in results.items():
            if cost > 1:
                cost = 1

            y += [cost]

            for hp_id, (hp_name, hp) in enumerate(hyperparameters.items()):
                # Skip hyperparameter if we haven't selected it
                if selected_hps is not None and hp not in selected_hps:
                    continue

                # Create a new list for the hyperparameter
                if hp_name not in X:
                    X[hp_name] = []

                if hp_name in config:
                    value = config[hp_name]

                    # Do the categorical mapping here
                    if isinstance(hp, CategoricalHyperparameter) or isinstance(hp, OrdinalHyperparameter):
                        if isinstance(hp, CategoricalHyperparameter):
                            choices = hp.choices
                        elif isinstance(hp, OrdinalHyperparameter):
                            choices = hp.sequence

                        if hp_name not in categorical_mapping:
                            categorical_mapping[hp_name] = {}

                            # It seems like it starts from 1 here
                            # https://github.com/automl/SMAC3/blob/3df3a749f3050d971af4110d051dae0cd795d615/smac/epm/util_funcs.py#L31
                            cat_id = 0
                            for choice in choices:
                                categorical_mapping[hp_name][choice] = cat_id
                                cat_id += 1

                            # Also add nan here
                            if len(cs.get_parents_of(hp_name)) > 0:
                                categorical_mapping[hp_name][None] = -1

                    elif isinstance(hp, Constant):
                        raise NotImplementedError()

                    # Add value to X
                    X[hp_name].append(value)

                else:
                    # Add nan otherwise
                    X[hp_name].append(np.nan)

        # We map the values between 0..1 now
        for hp_name, values in X.copy().items():
            hp = hyperparameters[hp_name]

            if hp_name not in mapping:
                if isinstance(hp, CategoricalHyperparameter) or isinstance(hp, OrdinalHyperparameter):
                    mapping[hp_name] = lambda v, reverse=False, nan=np.nan, mapping=categorical_mapping[hp_name]: \
                        categorical_map_fn(v, mapping, reverse, nan)
                else:
                    values = np.array(values)

                    try:
                        mn = hp.lower
                        mx = hp.upper
                    except:
                        mn = np.nanmin(values)
                        mx = np.nanmax(values)

                    mapping[hp_name] = lambda v, reverse=False, nan=np.nan, mn=mn, mx=mx, log=hp.log: \
                        numerical_map_fn(v, mn, mx, log, reverse, nan)

            # Use the mapping to create new_values now
            new_values = []
            for value in values:
                new_values.append(mapping[hp_name](value))

            # Update X
            X[hp_name] = new_values

        # Moreover, we need a id mapping because
        # the hyperparameter ids have changed
        id_mapping = {}
        new_id = 0
        for hp_id, (hp_name, hp) in enumerate(hyperparameters.items()):
            if selected_hps is not None and hp not in selected_hps:
                id_mapping[hp_id] = None
            else:
                id_mapping[hp_id] = new_id
                new_id += 1

        # Types and bonds have to be addressed now
        # Simply remove the not used ids
        types = [elem for i, elem in enumerate(
            types) if id_mapping[i] is not None]
        bounds = [elem for i, elem in enumerate(
            bounds) if id_mapping[i] is not None]

        return X, y, mapping, id_mapping, types, bounds, new_configspace

    def get_fidelities(self):
        budgets = []

        runkeys = self.rh.data
        for runkey in runkeys:
            budget = runkey.budget

            if budget not in budgets:
                budgets.append(budget)

        return budgets

    def get_fidelity(self, id):
        fidelities = self.get_fidelities()
        return fidelities[id]

    def has_fidelities(self):
        fidelities = self.get_fidelities()
        return len(fidelities) > 0

    def get_trajectory(self, fidelity=None):
        """
        Trajectory can be returned based on which fidelity should be
        considered.
        """

        wallclock_times = [0]
        costs = [1]
        additional = [{}]

        # How to get trajectory for all fidelities?
        # Have to consider instance, seed, fidelity ...

        current_cost = 1
        starttime = None

        # Create new runhistory to have full control
        rh = RunHistory()

        for run_key, run_value in self.rh.data.items():
            config = self.rh.ids_config[run_key.config_id]
            origin = config.origin
            budget = run_key.budget

            # If a specific fidelity is desired,
            # then we simply exclude all runs with a different one
            if fidelity is not None:
                if budget != fidelity:
                    continue

            rh.add(
                config=config,
                cost=run_value.cost,
                time=run_value.time,
                status=run_value.status,
                instance_id=run_key.instance_id,
                seed=run_key.seed,
                budget=budget,
                starttime=run_value.starttime,
                endtime=run_value.endtime,
                additional_info=run_value.additional_info,
                force_update=True
            )

            # wallclock time is not saved within the runhistory
            # therefore, we calculate it based on the start time
            if starttime is None:
                starttime = run_value.starttime

            wallclock_time = (-1) * (starttime - run_value.endtime)

            cost = list(rh.get_instance_costs_for_config(config).values())[0]
            if cost < current_cost:
                current_cost = cost

                # Update trajectory here
                wallclock_times += [wallclock_time]
                costs += [cost]
                additional += [{
                    "config": config,
                    "origin": origin,
                    "budget": budget
                }]

        return wallclock_times, costs, additional

    def __repr__(self):
        import hashlib
        import json

        def encode(json_serializable):
            return hashlib.sha1(json.dumps(json_serializable).encode()).hexdigest()

        rh_data_dict = {}
        for k, v in self.rh.data.items():
            rh_data_dict[str(k)] = str(v)

        meta_encoded = encode(self.meta)
        rh_encoded = encode(rh_data_dict)
        cs_encoded = encode(ConfigSpace.read_and_write.json.write(self.cs))

        return str(meta_encoded) + str(rh_encoded) + str(cs_encoded)
