from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import copy
from dataclasses import dataclass
from enum import IntEnum

import ConfigSpace
import numpy as np
import pandas as pd
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.runs.objective import Objective
from deepcave.utils.hash import string_to_hash
from deepcave.utils.logs import get_logger


class NotValidRunError(Exception):
    """Raised if directory is not a valid run."""

    pass


class NotMergeableError(Exception):
    """Raised if two or more runs are not mergeable"""

    pass


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    MEMORYOUT = 3
    CRASHED = 4
    ABORTED = 5
    RUNNING = 6


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

    def get_key(self) -> Tuple[str, int]:
        return self.config_id, self.budget  # noqa


class AbstractRun(ABC):
    prefix: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(self.__class__.__name__)

        # objects created by reset
        self.reset()

    def reset(self) -> None:
        self.meta: Dict[str, Any] = {}
        self.configspace: Optional[ConfigSpace.ConfigurationSpace] = None
        self.configs: Dict[int, Configuration] = {}
        self.origins: Dict[int, str] = {}
        self.models: Dict[int, Any] = {}

        self.history: List[Trial] = []
        self.trial_keys: Dict[Tuple[str, int], int] = {}

    @property
    def run_cache_id(self) -> str:
        return string_to_hash(f"{self.prefix}:{self.name}")

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Hash of current run. If hash changes, cache has to be cleared, as something has changed
        """
        pass

    def get_meta(self) -> None:
        return self.meta

    def empty(self) -> None:
        return len(self.history) == 0

    def get_objectives(self) -> None:
        objectives = []
        for d in self.meta["objectives"]:
            objective = Objective(
                name=d["name"],
                lower=d["lower"],
                upper=d["upper"],
                optimize=d["optimize"],
            )

            objective["lock_lower"] = d["lock_lower"]
            objective["lock_upper"] = d["lock_upper"]

            objectives.append(objective)

        return objectives

    def get_trials(self) -> Iterator[Trial]:
        yield from self.history

    def get_objective_name(self, objective_names=None) -> str:
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

    def get_objective_names(self) -> List:
        return [obj["name"] for obj in self.get_objectives()]

    def get_configs(self, budget=None) -> List:
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

    def get_budgets(self, human=False) -> List[str]:
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

        costs_mean = []
        costs_std = []
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

                costs_mean.append(cost)
                costs_std.append(0)
                times.append(trial.end_time)
                ids.append(id)

        return times, costs_mean, costs_std, ids

    def get_encoded_configs(
        self,
        objective_names=None,
        budget=None,
        statuses=None,
        for_tree=False,
        pandas=False,
    ):
        """
        Args:
            for_tree (bool): Inactives are treated differently. If false, all inactives are set to
            -1.
            normalize (bool): Normalize the configuration between 0 and 1.
            pandas (bool): Return pandas DataFrame instead of X and Y.
        Returns:
            X, Y (np.array): Encoded configurations OR
            df, df_labels (pd.DataFrame): Encoded dataframes if pandas equals True.
        """

        hp_names = self.configspace.get_hyperparameter_names()

        X, Y = [], []
        labels = []

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():
            config = self.configs[config_id]
            config = Configuration(self.configspace, config)

            y = self.calculate_cost(costs, objective_names)
            x = config.get_array()

            X.append(x)
            Y.append(y)

            labels_ = []
            for hp_name in hp_names:
                # hyperparameter name may not be in config
                if hp_name in config:
                    label = config[hp_name]

                    # Scientific notation
                    if type(label) == float:
                        if str(label).startswith("0.000") or "e-" in str(label):
                            label = np.format_float_scientific(label, precision=2)
                        else:
                            # Round to 2 decimals
                            label = np.round(label, 2)

                    labels_ += [label]
                else:
                    labels_ += ["NaN"]

            # We append y here directly
            labels_ += [y]
            labels.append(labels_)

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
            columns = [name for name in self.configspace.get_hyperparameter_names()] + [
                cost_column
            ]

            Y = Y.reshape(-1, 1)
            data = np.concatenate((X, Y), axis=1)

            df = pd.DataFrame(data=data, columns=columns)
            df_labels = pd.DataFrame(data=labels, columns=columns)

            return df, df_labels

        return X, Y


def check_equality(
    runs: List[AbstractRun],
    meta: bool = True,
    configspace: bool = True,
    objectives: bool = True,
    budgets: bool = True,
) -> Dict[str, Any]:
    """
    Checks the passed runs on equality based on the selected runs and returns the requested
    attributes.

    Parameters
    ----------
    runs : list[AbstractRun]
        Runs to check for equality.
    meta : bool, optional
        Meta-Data excluding objectives and budgets, by default True
    configspace : bool, optional
        ConfigSpace, by default True
    objectives : bool, optional
        Objectives, by default True
    budgets : bool, optional
        Budgets, by default True

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the checked attributes.
    """

    result = {}

    if len(runs) == 0:
        return result

    # Check meta
    if meta:
        m1 = runs[0].get_meta()
        for run in runs:
            m2 = run.get_meta()

            for k, v in m1.items():
                # Don't check on objectives or budgets
                if k == "objectives" or k == "budgets":
                    continue

                if k not in m2 or m2[k] != v:
                    raise NotMergeableError("Meta data of runs are not equal.")

        result["meta"] = m1

    # Make sure the same configspace is used
    # Otherwise it does not make sense to merge
    # the histories
    if configspace:
        cs1 = runs[0].configspace
        for run in runs:
            cs2 = run.configspace
            if cs1 != cs2:
                raise NotMergeableError("Configspace of runs are not equal.")

        result["configspace"] = cs1

    # Also check if budgets are the same
    if budgets:
        b1 = runs[0].get_budgets()
        for run in runs:
            b2 = run.get_budgets()
            if b1 != b2:
                raise NotMergeableError("Budgets of runs are not equal.")

        result["budgets"] = b1
        if meta:
            result["meta"]["budgets"] = b1

    # And if objectives are the same
    if objectives:
        o1 = None
        for run in runs:
            o2 = run.get_objectives()

            if o1 is None:
                o1 = o2
                continue

            if len(o1) != len(o2):
                raise NotMergeableError("Objectives of runs are not equal.")

            for o1_, o2_ in zip(o1, o2):
                o1_.merge(o2_)

        result["objectives"] = o1
        if meta:
            result["meta"]["objectives"] = o1

    return result
