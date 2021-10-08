import numpy as np
import pandas as pd
import copy

import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
#from deep_cave.evaluators.epm.util_funcs import get_types
from smac.tae import StatusType


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

    def get_configspace(self):
        return self.cs

    def get_incumbent(self, fidelity=None):
        costs = self.costs(fidelity)
        return min(costs, key=costs.get)

    def get_costs(self, fidelities=None, statuses=[StatusType.SUCCESS]):
        """
        How's the behaviour here?

        ???
        If multiple fidelities are given, only the configs which are evaluated on all given fidelities are
        returned.
        ???


        Input:
            fidelities (list or float or None)
        """

        config_ids = []
        for (config_id, _, _, budget), (_, _, status, _, _, _) in self.rh.data.items():
            if fidelities is not None:
                if isinstance(fidelities, list) and budget not in fidelities:
                    continue
                elif fidelities != budget:
                    continue

            if status not in statuses:
                continue

            config_ids.append(config_id)

        configs = []
        for config_id in config_ids:
            configs.append(self.rh.ids_config[config_id])

        results = {}
        for config in configs:
            cost = self.rh.get_instance_costs_for_config(config)
            results[config] = list(cost.values())[0]

        return results

    def get_encoded_data(self,
                         fidelities=None,
                         statuses=[StatusType.SUCCESS],
                         for_tree=False):
        """
        Inputs:
            `for_tree`: Inactives are treated differently.
        """

        X = []
        Y = []

        results = self.get_costs(fidelities, statuses)
        for config, cost in results.items():
            encoded = config.get_array()

            X.append(encoded)
            Y.append(cost)

        X = np.array(X)
        Y = np.array(Y)

        # Imputation: Easiest case is to replace all nans with -1
        # However, since Stefan used different values for inactives
        # we also have to use different inactives to be compatible
        # with the random forests.
        # https://github.com/automl/SMAC3/blob/a0c89502f240c1205f83983c8f7c904902ba416d/smac/epm/base_rf.py#L45

        if not for_tree:
            X[np.isnan(X)] = -1
        else:
            conditional = {}
            impute_values = {}

            for idx, hp in enumerate(self.cs.get_hyperparameters()):
                if idx not in conditional:
                    parents = self.cs.get_parents_of(hp.name)
                    if len(parents) == 0:
                        conditional[idx] = False
                    else:
                        conditional[idx] = True
                        if isinstance(hp, CategoricalHyperparameter):
                            impute_values[idx] = len(hp.choices)
                        elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                            impute_values[idx] = -1
                        elif isinstance(hp, Constant):
                            impute_values[idx] = 1
                        else:
                            raise ValueError

                if conditional[idx] is True:
                    nonfinite_mask = ~np.isfinite(X[:, idx])
                    X[nonfinite_mask, idx] = impute_values[idx]

        return X, Y

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
