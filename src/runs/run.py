import os
import numpy as np
import jsonlines
from enum import IntEnum
import json

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter

from src.utils.files import make_dirs
from src.utils.logs import get_logger

logger = get_logger(__name__)


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORTED = 4
    STOPPED = 5
    RUNNING = 6


class Run:
    """
    Creates
    - meta.json
    - configspace.json
    - configs.json
    - history.jsonl
    - origins.json
    - models/1.blub
    """

    def __init__(self,
                 configspace=None,
                 objectives="accuracy",
                 objective_weights=None,
                 meta={},
                 path=None):
        """
        If path is given, trials are loaded from the path.

        Inputs:
            meta (dict): Could be `ram`, `cores`, ...
        """

        self.reset()
        self.configspace = configspace
        self.path = path
        if self.path is not None:
            return self.load()

        if configspace is None and path is None:
            raise RuntimeError(
                "Please provide a configspace or specify a path to load existing trials.")

        self.meta = {
            "objectives": objectives,
            "objective_weights": objective_weights,
            "budgets": []
        }
        self.meta.update(meta)

        # Objectives and objective weights must be compatible
        if objective_weights is not None:
            self._check_objective_compatibility(objective_weights)

    def reset(self):
        self.meta = {}
        self.configspace = None
        self.configs = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        """
        If path is changed, also change the filenames of all created files.
        """

        if value is None:
            self._path = None
            return

        if value[-1] != "/":
            value += "/"

        make_dirs(value)
        self._path = value

        self.meta_fn = os.path.join(value, "meta.json")
        self.configspace_fn = os.path.join(value, "configspace.json")
        self.configs_fn = os.path.join(value, "configs.json")
        self.origins_fn = os.path.join(value, "origins.json")
        self.history_fn = os.path.join(value, "history.jsonl")

    def exists(self):
        if self._path is None:
            return False

        return os.path.isfile(self.meta_fn) and \
            os.path.isfile(self.configspace_fn) and \
            os.path.isfile(self.configs_fn) and \
            os.path.isfile(self.origins_fn) and \
            os.path.isfile(self.history_fn)

    def add(self,
            costs,
            config,  # either dict or Configuration
            budget=None,
            start_time=None,
            end_time=None,
            status=Status.SUCCESS,
            origin=None,
            model=None,
            additional={}):
        """

        if combination of config and budget already exists, it will be overwritten.

        Inputs:
            additional (dict): What's supported by DeepCAVE? Like `ram`, 
        """

        self._check_objective_compatibility(costs)

        if isinstance(config, Configuration):
            config = config.get_dictionary()

        if config not in self.configs.values():
            config_id = len(self.configs)
            self.configs[config_id] = config
            self.origins[config_id] = origin
        else:
            config_id = self.get_config_id(config)

        config_id = self.get_config_id(config)
        trial = Trial(
            config_id=config_id,
            budget=budget,
            costs=costs,
            start_time=np.round(start_time, 2),
            end_time=np.round(end_time, 2),
            status=status,
            additional=additional
        )

        trial_key = trial.get_key()
        if trial_key not in self.trial_keys:
            self.trial_keys[trial_key] = len(self.history)
            self.history.append(trial)
        else:
            self.history[self.trial_keys[trial_key]] = trial

        # Update budgets
        if budget not in self.meta["budgets"]:
            self.meta["budgets"].append(budget)
            self.meta["budgets"].sort()

        # Update models
        self.models[trial_key] = model

    def _check_objective_compatibility(self, o):
        """
        Costs or weights must be the same format as the defined objectives.
        Otherwise, this function will return an error.
        """

        if o is None:
            return

        if isinstance(o, list):
            if len(o) == len(self.meta["objectives"]):
                return
        elif isinstance(o, int) or isinstance(o, float):
            if isinstance(self.meta["objectives"], str):
                return

        raise RuntimeError(
            "Object does not match the size of objectives.")

    def get_config_id(self, config: dict):
        # Find out config id
        for id, c in self.configs.items():
            if c == config:
                return id

        return None

    def get_configs(self, budget=None):
        """"""

        configs = []
        for trial in self.history:
            if budget is not None:
                if budget != trial.budget:
                    continue

            config = self.configs[trial.config_id]
            configs += [config]

        return configs

    def get_budget(self, id):
        return self.meta["budgets"][id]

    def get_budgets(self):
        return self.meta["budgets"]

    def get_highest_budget(self):
        budgets = self.meta["budgets"]
        if len(budgets) == 0:
            return None

        return budgets[-1]

    def get_costs(self, budget=None, statuses=[Status.SUCCESS]):
        """
        If no budget is given, the highest budget is chosen.
        """

        if budget is None:
            budget = self.get_highest_budget()

        results = {}
        for trial in self.history:
            if trial.budget is not None:
                if trial.budget != budget:
                    continue

            if trial.status not in statuses:
                continue

            results[trial.config_id] = trial.costs

        return results

    def get_trajectory(self, budget=None):
        """
        If no budget is chosen, only the highest is considered.
        """

        if budget is None:
            budget = self.get_budgets()[-1]

        costs = []
        times = []

        # TODO: Sort self.history by start_time

        current_cost = np.inf
        for trial in self.history:
            # Only consider selected/last budget
            if trial.budget != budget:
                continue

            cost = self.calculate_cost(trial.costs)
            if cost < current_cost:
                current_cost = cost

                costs.append(cost)
                times.append(trial.end_time)  # Use end_time as time

        return costs, times

    def calculate_cost(self, costs):
        """
        Calculates cost from multiple costs given objective weights.
        If no weights are given, the mean is used.
        """

        if not isinstance(costs, list):
            costs = [costs]

        objective_weights = self.meta["objective_weights"]
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
            self._check_objective_compatibility(objective_weights)

        costs = [u*v for u, v in zip(costs, objective_weights)]
        cost = np.mean(costs)

        return cost

    def empty(self):
        return len(self.history) == 0

    def get_encoded_configs(self,
                            budget=None,
                            statuses=[Status.SUCCESS],
                            for_tree=False):
        """
        Inputs:
            `for_tree`: Inactives are treated differently.
        """

        X = []
        Y = []

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():

            config = self.configs[config_id]
            config = Configuration(self.configspace, config)

            encoded = config.get_array()
            cost = self.calculate_cost(costs)

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

            for idx, hp in enumerate(self.configspace.get_hyperparameters()):
                if idx not in conditional:
                    parents = self.configspace.get_parents_of(hp.name)
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

    def save(self, path=None):
        """
        If path is none, self.path will be chosen.
        """

        if path is not None:
            self.path = path

        if self.path is None:
            raise RuntimeError("Please specify a path to save the trials.")

        # Save configspace
        with open(self.configspace_fn, 'w') as f:
            f.write(cs_json.write(self.configspace))

        # Save meta data (could be changed)
        with open(self.meta_fn, 'w') as f:
            json.dump(self.meta, f, indent=4)

        with open(self.configs_fn, 'w') as f:
            json.dump(self.configs, f, indent=4)

        with open(self.origins_fn, 'w') as f:
            json.dump(self.origins, f, indent=4)

        # Save history
        with jsonlines.open(self.history_fn, mode='w') as f:
            for trial in self.history:
                f.write(trial)

        # TODO: Update general cache file and tell him that self.path was used
        # to save the run.
        # Then, DeepCAVE can show direct suggestions in the select path dialog.

    def load(self, path=None):
        self.reset()

        if path is not None:
            self.path = path

        if not self.exists():
            raise RuntimeError(
                "Could not load trials because trials were not found.")

        # Load meta data
        with open(self.meta_fn) as f:
            self.meta = json.load(f)

        # Load configspace
        with open(self.configspace_fn, 'r') as f:
            self.configspace = cs_json.read(f.read())

        # Load configs
        with open(self.configs_fn) as f:
            configs = json.load(f)
            self.configs = {int(k): v for k, v in configs.items()}
            # Make sure all keys are integers

        # Load origins
        with open(self.origins_fn) as f:
            self.origins = json.load(f)

        # Load history
        with jsonlines.open(self.history_fn) as f:
            self.history = []
            for obj in f:
                # Create trial object here
                trial = Trial(*obj)
                self.history.append(trial)

                # Also create trial_keys
                self.trial_keys[trial.get_key()] = len(self.history) - 1

        # Load models
        # TODO


class Trial(tuple):
    def __new__(cls, *args, **kwargs):

        if len(kwargs) > 0:
            return super(Trial, cls).__new__(cls, tuple(kwargs.values()))
        else:
            return super(Trial, cls).__new__(cls, tuple(args))

    def __init__(self,
                 config_id,
                 budget,
                 costs,
                 start_time,
                 end_time,
                 status,
                 additional):

        data = {
            "config_id": config_id,
            "budget": budget,
            "costs": costs,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
            "additional": additional
        }

        # Make dict available as member variables
        for k, v in data.items():
            setattr(self, k, v)

    def get_key(self):
        return (self.config_id, self.budget)
