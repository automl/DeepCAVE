from copy import deepcopy
from typing import Optional, Any

import ConfigSpace
from ConfigSpace import Configuration

from deepcave.runs import AbstractRun
from deepcave.runs.run import Run
from deepcave.utils.hash import string_to_hash


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

            # We need new config ids
            current_config_id = 0

            # Combine runs here
            for runs in self.runs:
                config_mapping = {}  # Maps old ones to the new ones

                # Update configs + origins
                for config_id in runs.configs.keys():
                    config = runs.configs[config_id]
                    origin = runs.origins[config_id]

                    if config not in self.configs.values():
                        self.configs[current_config_id] = config
                        self.origins[current_config_id] = origin
                        current_config_id += 1
                        config_mapping[config_id] = current_config_id

                # Update history + trial_keys
                for trial in self.history:
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
                raise RuntimeError("Configspace of runs are not equal.")

        return cs

    def get_meta(self) -> dict[str, str]:
        """
        Returns the meta data if all runs have the same meta data.
        Otherwise raise an error.
        """

        meta = self.runs[0].get_meta()
        for run in self.runs:
            if meta != run.get_meta():
                raise RuntimeError("Meta data of runs are not equal.")

        return meta

    def get_objectives(self) -> dict[str, Any]:
        """
        Returns the objectives if all runs have the same objectives.
        Otherwise raise an error.
        """

        objectives = self.runs[0].get_objectives()
        for run in self.runs:
            if objectives != run.get_objectives():
                raise RuntimeError("Meta data of runs is not equal.")

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
            raise RuntimeError("Run data are not mergeable.")

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
            raise RuntimeError("Run data are not mergeable.")

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
            raise RuntimeError("Run data are not mergeable.")

    def get_budgets(self, human=False) -> list[str]:
        """
        Returns all budgets if all runs have the same budgets.
        Otherwise raise an error.
        """

        budgets = self.runs[0].get_budgets(human=human)
        for run in self.runs:
            if budgets != run.get_budget(human=human):
                raise RuntimeError("Budgets of runs are not equal.")

        return budgets

    def get_budget(self, idx: int) -> float:
        """
        Returns the budget of index `idx` if all runs have the same budget.
        Otherwise raise an error.
        """

        try:
            budgets = self.get_budgets()
            return budgets[idx]
        except:
            raise RuntimeError("Budgets of runs are not equal.")


    def get_highest_budget(self):
        """
        Returns highest budget if all runs have the same budgets.
        Otherwise raise an error.
        """

        try:
            self.get_budgets()
            return super().get_highest_budget()
        except:
            raise RuntimeError("Budgets of runs are not equal.")



    def get_costs(self, *args, **kwargs):
        """
        Return costs if runs are mergeable.
        Otherwise raise an error.
        If no budget is given, the highest budget is chosen.
        """

        if self.merged_history:
            return self.get_costs(*args, **kwargs)
        else:
            raise RuntimeError("Run data are not mergeable.")


# TODO(dwoiwode): Folgender Code sollte auch die Trial-Klasse ersetzen können. Ist vielleicht lesbarer als ein vererbter Tuple