from typing import Dict, List, Tuple

from copy import deepcopy

import numpy as np

from deepcave.runs import AbstractRun, NotMergeableError, check_equality
from deepcave.utils.hash import string_to_hash


class Group(AbstractRun):
    prefix = "group"

    def __init__(self, name: str, runs: List[AbstractRun]):
        super(Group, self).__init__(name)
        self.runs = [run for run in runs if run is not None]  # Filter for Nones
        self.reset()

        if len(self.runs) == 0:
            return

        try:
            attributes = check_equality(self.runs)
            # abstract run requires meta to contain budgets / objectives
            self.meta = {"budgets": attributes["budgets"], "objectives": attributes["objectives"]}
            self.configspace = attributes["configspace"]
            self.objectives = attributes["objectives"]
            self.budgets = attributes["budgets"]

            # We need new config ids
            current_config_id = 0

            # Key: new_config_id; Value: (run_id, config_id)
            self._original_config_mapping: Dict[int, Tuple[int, int]] = {}

            # Key: (run_id, config_id); Value: new_config_id
            self._new_config_mapping: Dict[Tuple[int, int], int] = {}

            # Combine runs here
            for run_id, run in enumerate(self.runs):
                config_mapping: Dict[int, int] = {}  # Maps old ones to the new ones

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
                    new_config_id = config_mapping[config_id]

                    # Update config id
                    trial.config_id = new_config_id

                    # Now we add it to the history
                    trial_key = trial.get_key()
                    if trial_key not in self.trial_keys:
                        self.trial_keys[trial_key] = len(self.history)
                        self.history += [trial]
                    else:
                        self.history[self.trial_keys[trial_key]] = trial

                    # Get model mapping done
                    self._original_config_mapping[new_config_id] = (run_id, config_id)
                    self._new_config_mapping[(run_id, config_id)] = new_config_id

                    # And update highest budget
                    self._update_highest_budget(new_config_id, trial.budget, trial.status)
        except Exception as e:
            raise NotMergeableError(f"Runs can not be merged: {e}")

    def __iter__(self):
        for run in self.runs:
            yield run.name

    @property
    def hash(self) -> str:
        hashes = []
        for run in self.runs:
            hashes += [run.hash]

        # We sort hashes now because we don't want to be dependent on the order
        hashes = sorted(hashes)
        return string_to_hash("-".join(hashes))

    @property
    def id(self) -> str:
        # Groups do not have a path, therefore we use the name.
        return string_to_hash(f"{self.prefix}:{self.name}")

    @property
    def latest_change(self) -> int:
        latest_change = 0
        for run in self.runs:
            if run.latest_change > latest_change:
                latest_change = run.latest_change

        return latest_change

    @property
    def run_paths(self) -> List[str]:
        return [str(run.path) for run in self.runs]

    @property
    def run_names(self) -> List[str]:
        return [run.name for run in self.runs]

    def get_runs(self) -> List[AbstractRun]:
        return self.runs

    def get_new_config_id(self, run_id: int, original_config_id: int) -> int:
        return self._new_config_mapping[(run_id, original_config_id)]

    def get_original_config_id(self, config_id: int) -> id:
        return self._original_config_mapping[config_id][1]

    def get_original_run(self, config_id: int) -> AbstractRun:
        run_id = self._original_config_mapping[config_id][0]
        return self.runs[run_id]

    def get_model(self, config_id):
        run_id, config_id = self._original_config_mapping[config_id]
        return self.runs[run_id].get_model(config_id)

    def get_trajectory(self, *args, **kwargs):
        # Cache costs
        run_costs = []
        run_times = []

        # All x values on which we need y values
        all_times = []

        for _, run in enumerate(self.runs):
            times, costs_mean, _, _, _ = run.get_trajectory(*args, **kwargs)

            # Cache s.t. we don't calculate it multiple times
            run_costs.append(costs_mean)
            run_times.append(times)

            # Add all times
            # We want to calculate standard deviation on all times
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

        return times, list(costs_mean), list(costs_std), [], []
