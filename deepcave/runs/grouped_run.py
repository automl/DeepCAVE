from typing import Any, Dict, List, Optional, Union

from copy import deepcopy

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration

from deepcave import Objective
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
        self.reset()

        if len(self.runs) == 0:
            return

        try:
            # Merge meta
            self.meta = self.runs[0].get_meta()
            for run in self.runs:
                meta = run.get_meta()

                for k, v in self.meta.items():
                    # Don't check on objectives or budgets
                    if k == "objectives" or k == "budgets":
                        continue

                    if k not in meta or meta[k] != v:
                        raise NotMergeableError("Meta data of runs are not equal.")

            # Make sure the same configspace is used
            # Otherwise it does not make sense to merge
            # the histories
            self.configspace = self.runs[0].configspace
            for run in self.runs:
                if self.configspace != run.configspace:
                    raise NotMergeableError("Configspace of runs are not equal.")

            # Also check if budgets are the same
            budgets = self.runs[0].get_budgets()
            for run in self.runs:
                if budgets != run.get_budgets():
                    raise NotMergeableError("Budgets of runs are not equal.")

            self.meta["budgets"] = budgets

            # And if objectives are the same
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
            self.meta["objectives"] = objectives

            # We need new config ids
            current_config_id = 0

            # Combine runs here
            for run in self.runs:
                config_mapping: dict[str, str] = {}  # Maps old ones to the new ones

                # Update configs + origins
                for config_id in run.configs.keys():
                    config = run.configs[config_id]
                    origin = run.origins[config_id]

                    for added_config_id, added_config in self.configs.items():
                        if config == added_config:
                            config_mapping[config_id] = added_config_id
                            break

                    if config_id not in config_mapping:
                        self.configs[str(current_config_id)] = config
                        self.origins[current_config_id] = origin
                        config_mapping[config_id] = str(current_config_id)
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
        except:
            raise NotMergeableError("Runs can not be merged.")

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
