from typing import Any, List, Optional, Union

from copy import deepcopy

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration

from deepcave.runs import AbstractRun
from deepcave.runs.run import Run
from deepcave.utils.hash import string_to_hash


class NotMergeableError(Exception):
    """Raised if two or more runs are not mergeable"""

    pass


class GroupedRun(AbstractRun):
    prefix = "group"

    def __init__(self, name: str, runs: list[Run]):
        super(GroupedRun, self).__init__(name)
        self.runs = [run for run in runs if run is not None]  # Filter for Nones

        self.configs: dict[str, Configuration] = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

        self.merged_history = False

        try:
            # Make sure the same configspace is used
            # Otherwise it does not make sense to merge
            # the histories
            self.configspace

            # Also check if budgets are the same
            self.get_budgets()

            # And if objectives are the same
            self.get_objectives()

            # We need new config ids
            current_config_id = 0

            # Combine runs here
            for run in self.runs:
                config_mapping = {}  # Maps old ones to the new ones

                # Update configs + origins
                for config_id in run.configs.keys():
                    config = run.configs[config_id]
                    origin = run.origins[config_id]

                    for added_config_id, added_config in self.configs.items():
                        if config == added_config:
                            config_mapping[config_id] = added_config_id
                            break

                    if config_id not in config_mapping:
                        self.configs[current_config_id] = config
                        self.origins[current_config_id] = origin
                        config_mapping[config_id] = current_config_id
                        current_config_id += 1

                # Update history + trial_keys
                for trial in run.history:
                    # Deep copy trial
                    trial = deepcopy(trial)

                    (config_id, budget) = trial.get_key()

                    # Config id might have changed
                    config_id = config_mapping[config_id]

                    # Update config id
                    trial.config_id = config_id

                    # Now we add it to the history
                    trial_key = trial.get_key()
                    if trial_key not in self.trial_keys:
                        self.trial_keys[trial_key] = len(self.history)
                        self.history.append(trial)
                    else:
                        self.history[self.trial_keys[trial_key]] = trial

            self.merged_history = True
        except:
            pass

    def __iter__(self):
        for run in self.runs:
            yield run.name

    @property
    def hash(self) -> str:
        total_hash_str = ""
        for run in self.runs:
            total_hash_str += run.hash

        return string_to_hash(total_hash_str)

    @property
    def run_names(self) -> list[str]:
        return [run.name for run in self.runs]

    @property
    def configspace(self) -> Optional[ConfigSpace.ConfigurationSpace]:
        cs = self.runs[0].configspace
        for run in self.runs:
            if cs != run.configspace:
                raise NotMergeableError("Configspace of runs are not equal.")

        return cs

    def get_meta(self) -> dict[str, str]:
        """
        Returns the meta data if all runs have the same meta data.
        Otherwise raise an error.
        """

        meta = self.runs[0].get_meta()
        for run in self.runs:
            meta2 = run.get_meta()

            for k, v in meta.items():
                # Don't check on objectives or budgets
                if k == "objectives" or k == "budgets":
                    continue

                if k not in meta2 or meta2[k] != v:
                    raise NotMergeableError("Meta data of runs are not equal.")

        return meta

    def get_objectives(self) -> dict[str, Any]:
        """
        Returns the objectives if all runs have the same objectives.
        Otherwise raise an error.
        """

        objectives = None
        for run in self.runs:
            objectives2 = run.get_objectives()

            if objectives is None:
                objectives = objectives2
                continue

            if len(objectives) != len(objectives2):
                raise NotMergeableError("Objectives of runs are not equal.")

            for o1, o2 in zip(objectives, objectives2):
                o1.merge(o2)

        return objectives

    def get_objective_names(self) -> list:
        """
        Returns the objective names if all runs have the same objective names
        Otherwise raise an error.
        """

        return [obj["name"] for obj in self.get_objectives()]

    def get_config(self, id):
        """
        Returns a config if runs are mergeable.
        Otherwise raise an error.
        """

        if self.merged_history:
            return self.configs[id]
        else:
            raise NotMergeableError("Run data are not mergeable.")

    def get_config_id(self, config: dict):
        """
        Returns a config id if runs are mergeable.
        Otherwise raise an error.
        """

        if self.merged_history:
            # Find out config id
            for id, c in self.configs.items():
                if c == config:
                    return id

            return None
        else:
            raise NotMergeableError("Run data are not mergeable.")

    def get_configs(self, budget=None):
        """
        Return all configs if runs are mergeable.
        Otherwise raise an error.
        """

        if self.merged_history:
            configs = []
            for trial in self.history:
                if budget is not None:
                    if budget != trial.budget:
                        continue

                config = self.configs[trial.config_id]
                configs += [config]

            return configs
        else:
            raise NotMergeableError("Run data are not mergeable.")

    def get_budgets(self, human=False) -> Union[List[str], NotMergeableError]:
        """
        Returns all budgets if all runs have the same budgets.
        Otherwise raise an error.
        """

        budgets = self.runs[0].get_budgets(human=human)
        for run in self.runs:
            if budgets != run.get_budgets(human=human):
                raise NotMergeableError("Budgets of runs are not equal.")

        return budgets

    def get_budget(self, idx: int) -> Union[float, NotMergeableError]:
        """
        Returns the budget of index `idx` if all runs have the same budget.
        Otherwise raise an error.
        """

        budgets = self.get_budgets()
        return budgets[idx]

    def get_highest_budget(self) -> Union[float, NotMergeableError]:
        """
        Returns highest budget if all runs have the same budgets.
        Otherwise raise an error.
        """

        self.get_budgets()
        return self.runs[0].get_highest_budget()

    def get_costs(self, *args, **kwargs):
        """
        Return costs if runs are mergeable.
        Otherwise raise an error.
        If no budget is given, the highest budget is chosen.
        """

        if self.merged_history:
            return super().get_costs(*args, **kwargs)
        else:
            raise NotMergeableError("Run data are not mergeable.")

    def get_min_cost(self, *args, **kwargs):
        """
        Return costs if runs are mergeable.
        Otherwise raise an error.
        If no budget is given, the highest budget is chosen.
        """

        if self.merged_history:
            return super().get_min_cost(*args, **kwargs)
        else:
            raise NotMergeableError("Run data are not mergeable.")

    def get_trajectory(self, *args, **kwargs):

        # Cache costs
        run_costs = []
        run_times = []

        # All x values on which we need y values
        all_times = []

        for run in self.runs:
            times, costs_mean, _, _ = run.get_trajectory(*args, **kwargs)

            # Cache st we don't calculate it multiple times
            run_costs.append(costs_mean)
            run_times.append(times)

            # Add all times
            for time in times:
                if time not in all_times:
                    all_times.append(time)

        all_times.sort()

        # Now look for corresponding y values
        all_costs = []

        for time in all_times:
            y = []

            # Iterate over all runs
            for costs, times in zip(run_costs, run_times):
                # Find closest x value
                idx = min(range(len(times)), key=lambda i: abs(times[i] - time))
                y.append(costs[idx])

            all_costs.append(y)

        # Make numpy arrays
        all_costs = np.array(all_costs)

        times = all_times
        costs_mean = np.mean(all_costs, axis=1)
        costs_std = np.std(all_costs, axis=1)

        return times, list(costs_mean), list(costs_std), []
