import os
import numpy as np
import jsonlines
from enum import IntEnum
import json
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as cs_json

from deep_cave.utils.files import make_dirs


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORTED = 4
    STOPPED = 5
    RUNNING = 6


class Trials:
    """
    Creates
    - meta.json
    - configspace.json
    - configs.json
    - history.jsonl
    - models/1.blub
    """

    def __init__(self,
                 save_path,
                 configspace=None,
                 objectives="cost",
                 meta={},
                 load=False):
        """
        Inputs:
            meta (dict): Could be `ram`, `cores`, ...
        """

        if save_path[-1] != "/":
            save_path += "/"

        make_dirs(save_path)

        # Filenames
        self.save_path = save_path
        self.meta_fn = os.path.join(save_path, "meta.json")
        self.configspace_fn = os.path.join(save_path, "configspace.json")
        self.configs_fn = os.path.join(save_path, "configs.json")
        self.history_fn = os.path.join(save_path, "history.jsonl")

        run_exists = os.path.isfile(self.meta_fn) and \
            os.path.isfile(self.configspace_fn) and \
            os.path.isfile(self.configs_fn) and \
            os.path.isfile(self.history_fn) and \

        if load:
            if run_exists:
                return self._load()
            else:
                raise RuntimeError("Trials could not be loaded.")
        else:
            # Check if files are already there
            if run_exists:
                raise RuntimeError("Files already exists.")

        if configspace is None:
            raise RuntimeError("Please provide a configspace.")

        self.meta = {
            "objectives": objectives,
            "budgets": []
        }
        self.meta.update(meta)
        self.configspace = configspace
        self.history = []
        self.configs = {}
        self.models = {}

        self._save(initial=True)

    def add(self,
            costs,
            config,  # either dict or Configuration
            budget=None,
            start_time=None,
            end_time=None,
            status=Status.SUCCESS,
            model=None,
            additional={}):
        """
        Inputs:
            additional (dict): What's supported by DeepCAVE? Like `ram`, 
        """

        self._check_cost_dimension(costs)

        if isinstance(config, Configuration):
            config = config.get_dictionary()

        if config not in self.configs.values():
            config_id = len(self.configs)
            self.configs[config_id] = config
        else:
            config_id = self.get_config_id(config)

        if budget not in self.meta["budgets"]:
            self.meta["budgets"].append(budget)

        config_id = self.get_config_id(config)

        entry = (
            config_id,
            budget,
            costs,
            np.round(start_time, 2),
            np.round(end_time, 2),
            status,
            additional
        )

        self.models[config_id] = model
        self.history.append(entry)

        self._save()

    def _check_cost_dimension(self, costs):
        if type(costs) != type(self.meta["objectives"]):
            raise RuntimeError(
                "Costs must have the same format as objectives.")
        else:
            if isinstance(costs, list):
                if len(costs) != len(self.meta["objectives"]):
                    raise RuntimeError(
                        "Costs does not match the size of objectives.")

    def get_config_id(self, config: dict):
        # Find out config id
        for id, c in self.configs.items():
            if c == config:
                return id

        return None

    def get_trajectory(self, budget=None, objective_weights=None):
        """
        If no budget is chosen, only the highest is considered.
        """

        if budget is None:
            budget = self.meta["budgets"][-1]

        costs = []
        times = []

        if objective_weights is None:
            objectives = self.meta["objectives"]
            if isinstance(objectives, list):
                # Give the same weight to all objectives
                objective_weights = [
                    1 / len(objectives) for _ in range(len(objectives))
                ]
            else:
                objective_weights = [1]
        else:
            # Make sure objective_weights has same length than objectives
            self._check_cost_dimension(objective_weights)

        # TODO: Sort self.history by end_time

        current_cost = np.inf
        for entry in self.history:
            # Only consider selected/last budget
            if entry[1] != budget:
                continue

            cost = entry[0]
            if not isinstance(cost, list):
                cost = [cost]

            cost = [u*v for u, v in zip(cost, objective_weights)]
            cost = np.mean(cost)
            if cost < current_cost:
                current_cost = cost

                costs.append(cost)
                times.append(entry[4])  # Use end_time as time

        return costs, times

    def empty(self):
        return len(self.history) == 0

    def _save(self, initial=False):
        if initial:
            # Save configspace
            with open(self.configspace_fn, 'w') as f:
                f.write(cs_json.write(self.configspace))

        # Save meta data (could be changed)
        with open(self.meta_fn, 'w') as f:
            json.dump(self.meta, f, indent=4)

        with open(self.configs_fn, 'w') as f:
            json.dump(self.configs, f, indent=4)

        # Save history
        with jsonlines.open(self.history_fn, mode='w') as f:
            for trial in self.history:
                f.write(trial)

    def _load(self):
        # Load meta data
        with open(self.meta_fn) as f:
            self.meta = json.load(f)

        # Load configspace
        with open(self.configspace_fn, 'r') as f:
            self.configspace = cs_json.read(f.read())

        # Load configs
        with open(self.configs_fn) as f:
            configs = json.load(f)
            self.configs = list(configs.values())

        # Load history
        with jsonlines.open(self.history_fn) as f:
            self.history = []
            for obj in f:
                self.history.append(obj)
