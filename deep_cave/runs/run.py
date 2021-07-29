import numpy as np
import pandas as pd
import ConfigSpace
from smac.runhistory.runhistory import RunHistory
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter


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
    
    def get_runhistory(self):
        return self.rh

    def get_configspace(self):
        return self.cs

    def get_config_costs(self, fidelity=None):
        results = {}

        if fidelity is None:
            configs = self.rh.get_all_configs()
        else:
            configs = self.rh.get_all_configs_per_budget([fidelity])

        for config in configs:
            cost = self.rh.get_instance_costs_for_config(config)
            results[config] = list(cost.values())[0]

        return results

    def get_encoded_hyperparameters(self, fidelity, hp_ids=None):
        hyperparameters = self.cs.get_hyperparameters_dict()
        hyperparameter_names = list(hyperparameters.keys())

        X, y = {}, []
        mapping = {}
        categorical_mapping = {}

        results = self.get_config_costs(fidelity)
        for config, cost in results.items():
            if cost > 1:
                cost = 1

            y += [cost]

            for hp_id, (hp_name, hp) in enumerate(hyperparameters.items()):
                # Skip hyperparameter if we haven't selected it
                if isinstance(hp_ids, list) and hp_id not in hp_ids:
                    continue

                # Create a new list for the hyperparameter
                if hp_name not in X:
                    X[hp_name] = []

                if hp_name in config:
                    value = config[hp_name]

                    # Do the mapping here
                    if isinstance(hp, CategoricalHyperparameter):

                        if hp_name not in categorical_mapping:
                            categorical_mapping[hp_name] = {}

                        if value not in categorical_mapping[hp_name]:
                            categorical_mapping[hp_name][value] = len(categorical_mapping[hp_name])

                        # Mapping tells us which value we have in the end
                        value = categorical_mapping[hp_name][value]

                    # Add value to X
                    X[hp_name].append(value)
                        
                else:
                    # Add nan otherwise
                    X[hp_name].append(np.nan)

        categorical_mapping_reversed = {}
        for hp_name, d in categorical_mapping.items():
            categorical_mapping_reversed[hp_name] = {v:k for k,v in d.items()}
        
        # We map the values between 0..1 now
        for hp_name, values in X.copy().items():
            hp = hyperparameters[hp_name]
            values = np.array(values)

            try:
                mn = hp.lower
                mx = hp.upper
            except:
                mn = np.nanmin(values)
                mx = np.nanmax(values)

            # Min-Max scaling
            new_values = (values-mn)/(mx-mn)

            # Replace nans
            new_values = np.nan_to_num(new_values, copy=False, nan=-1.0)

            if hp_name not in mapping:
                mapping[hp_name] = {}

            for v1, v2 in zip(values, new_values):
                if np.isnan(v1):
                    value = "Missing Value"
                elif isinstance(hp, CategoricalHyperparameter):
                    # We have to take the categorical mapping here
                    value = categorical_mapping_reversed[hp_name][v1]
                else:
                    value = v1

                mapping[hp_name][v2] = value

            # Update X
            X[hp_name] = new_values

        # Since we might not add all hyperparameters
        # We get a different hyperparameter mapping later on
        hp_id_mapping = {}

        config_space = ConfigSpace.ConfigurationSpace()
        new_hp_id = 0
        for hp_id, hp_name in enumerate(hyperparameter_names):
            if isinstance(hp_ids, list) and hp_id not in hp_ids:
                continue

            config_space.add_hyperparameter(UniformFloatHyperparameter(hp_name, -1, 1))
            
            hp_id_mapping[hp_id] = new_hp_id
            new_hp_id += 1
        
        return X, y, mapping, config_space, hp_id_mapping


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


            
            




    
