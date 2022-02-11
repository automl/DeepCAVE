from copy import deepcopy
from typing import List
import numpy as np
from deepcave.runs import AbstractRun, NotMergeableError, check_equality
from deepcave.utils.hash import string_to_hash


class GroupedRun(AbstractRun):
    prefix = "group"

    def __init__(self, name: str, runs: List[AbstractRun]):
        super(GroupedRun, self).__init__(name)
        self.runs = [run for run in runs if run is not None]  # Filter for Nones
        self.reset()

        if len(self.runs) == 0:
            return

        try:
            attributes = check_equality(self.runs)
            self.meta = attributes["meta"]
            self.configspace = attributes["configspace"]
            self.objectives = attributes["objectives"]
            self.budgets = attributes["budgets"]

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
