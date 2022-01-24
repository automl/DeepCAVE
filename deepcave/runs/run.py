from abc import ABC, abstractmethod
from functools import cached_property
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Union, Any, Optional, Iterable, Iterator
from copy import deepcopy

import ConfigSpace
import jsonlines
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.read_and_write import json as cs_json

from deepcave.runs.objective import Objective
from deepcave.utils.files import make_dirs
from deepcave.utils.logs import get_logger
from deepcave.utils.hash import string_to_hash

logger = get_logger(__name__)


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    MEMORYOUT = 3
    CRASHED = 4
    ABORTED = 5
    RUNNING = 6


class AbstractRun(ABC):
    prefix: str

    def __init__(self, name: str):
        self.name = name

    @property
    def run_cache_id(self) -> str:
        return string_to_hash(f"{self.prefix}:{self.name}")

    @property
    @abstractmethod
    def configspace(self) -> Optional[ConfigSpace.ConfigurationSpace]:
        return None

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Hash of current run. If hash changes, cache has to be cleared, as something has changed
        """
        pass
    
    def empty(self):
        return len(self.history) == 0

    def get_objectives(self):
        return self.meta["objectives"]

    def get_objective_name(self, objective_names=None):
        """
        Get the cost name of given objective names.
        Returns "Combined Cost" if multiple objective names were involved.
        """

        given_objective_names = self.get_objective_names()

        if objective_names is None:
            if len(given_objective_names) == 1:
                return given_objective_names[0]
        else:
            if isinstance(objective_names, str):
                objective_names = [objective_names]

            if len(objective_names) == 1:
                return objective_names[0]

        return "Combined Cost"

    def get_objective_names(self) -> list:
        return [obj["name"] for obj in self.get_objectives()]

    def get_configs(self, budget=None):
        configs = []
        for trial in self.history:
            if budget is not None:
                if budget != trial.budget:
                    continue

            config = self.configs[trial.config_id]
            configs += [config]

        return configs

    def get_config(self, id):
        return self.configs[id]

    def get_config_id(self, config: dict):
        # Find out config id
        for id, c in self.configs.items():
            if c == config:
                return id

        return None

    def get_budget(self, id: int) -> float:
        return self.meta["budgets"][id]

    def get_budgets(self, human=False) -> list[str]:
        """
        There's at least one budget with None included.

        Args:

        """
        budgets = self.meta["budgets"]
        assert len(budgets) > 0

        if human:
            readable_budgets = []
            for b in budgets:
                if b is None:
                    readable_budgets += ["None"]
                else:
                    readable_budgets += [str(np.round(float(b), 2))]

            return readable_budgets

        return budgets

    def get_highest_budget(self):
        budgets = self.meta["budgets"]
        if len(budgets) == 0:
            return None

        return budgets[-1]
    
    def _process_costs(self, costs: Iterable[float]) -> list[float]:
        """
        Get rid of none costs.
        """

        new_costs = []
        for cost, obj in zip(costs, self.get_objectives()):
            # Replace with the worst cost
            if cost is None:
                if obj["optimize"] == "lower":
                    cost = obj["upper"]
                else:
                    cost = obj["lower"]

            new_costs += [cost]

        return new_costs

    def get_costs(self, budget=None, statuses=None):
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

            if statuses is not None:
                if trial.status not in statuses:
                    continue

            results[trial.config_id] = self._process_costs(trial.costs)

        return results

    def get_min_cost(self, objective_names=None, budget=None, statuses=None):
        min_cost = np.inf
        best_config = None

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():
            cost = self.calculate_cost(costs, objective_names, normalize=True)

            if cost < min_cost:
                min_cost = cost
                best_config = self.get_config(config_id)

        return min_cost, best_config

    def calculate_cost(self, costs, objective_names=None, normalize=False) -> float:
        """
        Calculates cost from multiple costs.
        Normalizes the cost first and weight every cost the same.
        """

        costs = self._process_costs(costs)

        if objective_names is None:
            objective_names = self.get_objective_names()

        assert len(objective_names) > 0

        # No normalization needed
        if len(objective_names) == 1 and not normalize:
            return costs[0]

        objectives = self.get_objectives()

        # First normalize
        filtered_objectives = []
        normalized_costs = []
        for cost, objective in zip(costs, objectives):
            if objective["name"] not in objective_names:
                continue

            a = cost - objective["lower"]
            b = objective["upper"] - objective["lower"]
            normalized_cost = a / b

            # We optimize the lower
            # So we need to flip the normalized cost
            if objective["optimize"] == "upper":
                normalized_cost = 1 - normalized_cost

            normalized_costs.append(normalized_cost)
            filtered_objectives.append(objective)

        # Give the same weight to all objectives (for now)
        objective_weights = [
            1 / len(objectives) for _ in range(len(filtered_objectives))
        ]

        costs = [u * v for u, v in zip(normalized_costs, objective_weights)]
        cost = np.mean(costs).item()

        return cost
    
    def get_trajectory(self, objective_names=None, budget=None):
        if budget is None:
            budget = self.get_highest_budget()

        costs = []
        ids = []
        times = []

        order = []
        # Sort self.history by end_time
        for id, trial in enumerate(self.history):
            order.append((id, trial.end_time))

        order.sort(key=lambda tup: tup[1])

        current_cost = np.inf
        for id, cost in order:
            trial = self.history[id]
            # Only consider selected/last budget
            if trial.budget != budget:
                continue

            cost = self.calculate_cost(trial.costs, objective_names)
            if cost < current_cost:
                current_cost = cost

                costs.append(cost)
                times.append(trial.end_time)
                ids.append(id)

        return costs, times, ids

    def get_encoded_configs(self,
                            objective_names=None,
                            budget=None,
                            statuses=None,
                            for_tree=False,
                            pandas=False) -> Union[tuple[np.ndarray, np.ndarray], pd.DataFrame]:
        """
        Args:
            for_tree (bool): Inactives are treated differently.
            pandas (bool): Return pandas DataFrame instead of X and Y.
        """

        X = []
        Y = []

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():
            config = self.configs[config_id]
            config = Configuration(self.configspace, config)

            encoded = config.get_array()
            cost = self.calculate_cost(costs, objective_names)

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

        if pandas:
            cost_column = self.get_objective_name(objective_names)

            Y = Y.reshape(-1, 1)
            data = np.concatenate((X, Y), axis=1)
            df = pd.DataFrame(
                data=data,
                # Combined Cost
                columns=[
                            name for name in self.configspace.get_hyperparameter_names()] + [cost_column])

            return df

        return X, Y


class Run(AbstractRun, ABC):
    """
    TODO(dwoiwode): Docstring is outdated?
    Creates
    - meta.json
    - configspace.json
    - configs.json
    - history.jsonl
    - origins.json
    - models/1.blub
    """
    prefix = "run"
    _initial_order: int

    def __init__(self,
                 name: str,
                 configspace=None,
                 objectives: Union[Objective, list[Objective]] = None,
                 meta: dict[str, Any] = None,
                 path: Optional[Union[str, Path]] = None):
        """
        If path is given, runs are loaded from the path.

        Inputs:
            objectives (Objective or list of Objective): ...
            meta (dict): Could be `ram`, `cores`, ...
        """
        super(Run, self).__init__(name)
        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        # objects created by reset
        self.configs: dict[str, Configuration] = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

        # Reset and load configspace/path
        self.reset()
        self._configspace = configspace
        self.path = path
        if self.path is not None:
            self.load()
            return

        if configspace is None and path is None:
            raise RuntimeError(
                "Please provide a configspace or specify a path to load existing trials.")

        # Objectives
        if not isinstance(objectives, list):
            objectives = [objectives]

        for objective in objectives:
            assert isinstance(objective, Objective)

        # Meta
        self.meta = {
            "objectives": objectives,
            "budgets": []
        }
        self.meta.update(meta)

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> "Run":
        """
        Based on a path, return a new Run object.
        """
        pass

    def reset(self):
        self.meta = {}
        self._configspace = None
        self.configs = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def configspace(self) -> ConfigSpace.ConfigurationSpace:
        return self._configspace

    @path.setter
    def path(self, value: Optional[Union[str, Path]]):
        """
        If path is changed, also change the filenames of all created files.
        """

        if value is None:
            self._path = None
            return

        self._path = Path(value)

        make_dirs(self._path)

        self.meta_fn = self._path / "meta.json"
        self.configspace_fn = self._path / "configspace.json"
        self.configs_fn = self._path / "configs.json"
        self.origins_fn = self._path / "origins.json"
        self.history_fn = self._path / "history.jsonl"

    def exists(self) -> bool:
        if self._path is None:
            return False

        return all(f.is_file() for f in (self.meta_fn, self.configspace_fn, self.configs_fn,
                                         self.origins_fn, self.history_fn))

    def add(self,
            costs: Union[list[float], float],
            config: Union[dict, Configuration],  # either dict or Configuration
            budget: float = np.inf,
            start_time: float = 0.,
            end_time: float = 0.,
            status: Status = Status.SUCCESS,
            origin=None,
            model=None,
            additional: Optional[dict] = None):
        """

        If combination of config and budget already exists, it will be overwritten.
        Not successful runs are added with None costs.
        The cost will be calculated on the worst result later on.

        Inputs:
            additional (dict): What's supported by DeepCAVE? Like `ram`, 
            costs (float or list of floats)
        """
        if additional is None:
            additional = {}

        if not isinstance(costs, list):
            costs = [costs]

        assert len(costs) == len(self.get_objectives())

        for i in range(len(costs)):
            cost = costs[i]
            objective = self.get_objectives()[i]

            # Update time objective here
            if objective["name"] == "time" and cost is None:
                costs[i] = end_time - start_time
                cost = costs[i]

            # If cost is none, replace it later with the highest cost
            if cost is None:
                continue

            # Update bounds here
            if not objective["lock_lower"]:
                if cost < objective["lower"]:
                    self.get_objectives()[i]["lower"] = cost

            if not objective["lock_upper"]:
                if cost > objective["upper"]:
                    self.get_objectives()[i]["upper"] = cost

        if isinstance(config, Configuration):
            config = config.get_dictionary()

        if config not in self.configs.values():
            config_id = len(self.configs)
            self.configs[config_id] = config
            self.origins[config_id] = origin

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

    def save(self, path: Optional[Union[str, Path]] = None):
        """
        If path is none, self.path will be chosen.
        """

        if path is None:
            raise RuntimeError("Please specify a path to save the trials.")

        self.path = Path(path)

        # Save configspace
        self.configspace_fn.write_text(cs_json.write(self.configspace))

        # Save meta data (could be changed)
        self.meta_fn.write_text(json.dumps(self.meta, indent=4))
        self.configs_fn.write_text(json.dumps(self.configs, indent=4))
        self.origins_fn.write_text(json.dumps(self.origins, indent=4))

        # Save history
        with jsonlines.open(self.history_fn, mode='w') as f:
            for trial in self.history:
                f.write(trial)

        # TODO: Update general cache file and tell him that self.path was used
        # to save the run.
        # Then, DeepCAVE can show direct suggestions in the select path dialog.

    def load(self, path: Optional[Union[str, Path]] = None):
        self.reset()

        if path is None and self.path is None:
            raise RuntimeError("Could not load trials because path is None.")
        if path is not None:
            self.path = Path(path)

        if not self.exists():
            raise RuntimeError("Could not load trials because trials were not found.")

        # Load meta data
        self.meta = json.loads(self.meta_fn.read_text())

        # Load configspace
        self._configspace = cs_json.read(self.configspace_fn.read_text())

        # Load configs
        configs = json.loads(self.configs_fn.read_text())
        # Make sure all keys are integers
        self.configs = {int(k): v for k, v in configs.items()}

        # Load origins
        self.origins = json.loads(self.origins_fn.read_text())

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
        raise:
            raise RuntimeError("Budgets of runs are not equal.")
        
    
    def get_highest_budget(self):
        """
        Returns highest budget if all runs have the same budgets.
        Otherwise raise an error.
        """
        
        try:
            self.get_budgets()
            return super.get_highest_budget()
        raise:
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
# @dataclass
# class Trial:
#     config_id: str
#     budget: int
#     costs: float
#     start_time: float
#     end_time: float
#     status: Status
#     additional: dict[str, Any]
#
#     def __post_init__(self):
#         if isinstance(self.status, int):
#             self.status = Status(self.status)
#
#         assert isinstance(self.status, Status)
#     def get_key(self) -> tuple[str, int]:
#         return self.config_id, self.budget  # noqa
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

        if isinstance(status, int):
            status = Status(status)

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

    def get_key(self) -> tuple[str, int]:
        return self.config_id, self.budget  # noqa
