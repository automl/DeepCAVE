from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
from enum import IntEnum
from typing import Optional, Iterator, Iterable, Union, Any

import ConfigSpace
import numpy as np
import pandas as pd
from ConfigSpace import (
    Configuration,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from deepcave.runs.objective import Objective

from deepcave.utils.hash import string_to_hash
from deepcave.utils.logs import get_logger


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
        self.logger = get_logger(self.__class__.__name__)

        # objects created by reset
        self.configs: dict[int, Configuration] = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

        self.meta = {}

    def reset(self):
        self.meta = {}
        self._configspace = None
        self.configs = {}
        self.origins = {}
        self.models = {}

        self.history = []
        self.trial_keys = {}

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

    def get_meta(self):
        return self.meta

    def empty(self):
        return len(self.history) == 0

    def get_objectives(self):
        objectives = []
        for d in self.meta["objectives"]:
            objective = Objective(name=d["name"],
                                  lower=d["lower"],
                                  upper=d["upper"],
                                  optimize=d["optimize"])

            objective["lock_lower"] = d["lock_lower"]
            objective["lock_upper"] = d["lock_upper"]

            objectives.append(objective)

        return objectives

    def get_trials(self) -> Iterator["Trial"]:
        yield from self.history

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

    def get_encoded_configs(
        self,
        objective_names=None,
        budget=None,
        statuses=None,
        for_tree=False,
        pandas=False,
    ) -> Union[tuple[np.ndarray, np.ndarray], pd.DataFrame]:
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
                        elif isinstance(
                            hp,
                            (UniformFloatHyperparameter, UniformIntegerHyperparameter),
                        ):
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
                columns=[name for name in self.configspace.get_hyperparameter_names()]
                + [cost_column]
            )

            return df

        return X, Y


@dataclass
class Trial:
    config_id: str
    budget: int
    costs: float
    start_time: float
    end_time: float
    status: Status
    additional: dict[str, Any]

    def __post_init__(self):
        if isinstance(self.status, int):
            self.status = Status(self.status)

        assert isinstance(self.status, Status)

    def get_key(self) -> tuple[str, int]:
        return self.config_id, self.budget  # noqa
