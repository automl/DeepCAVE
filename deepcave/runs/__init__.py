from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from pathlib import Path
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
from deepcave.constants import CONSTANT_VALUE, NAN_LABEL, NAN_VALUE

from deepcave.runs.objective import Objective
from deepcave.utils.hash import string_to_hash
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import prettify_label


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
    NOTFOUND = 7


@dataclass
class Trial:
    config_id: int
    budget: Union[int, float]
    costs: List[float]
    start_time: float
    end_time: float
    status: Status
    additional: Dict[str, Any]

    def __post_init__(self):
        if isinstance(self.status, int):
            self.status = Status(self.status)

        assert isinstance(self.status, Status)

    def get_key(self) -> Tuple[int, int]:
        return AbstractRun.get_trial_key(self.config_id, self.budget)

    def to_json(self) -> List[Any]:
        return [
            self.config_id,
            self.budget,
            self.costs,
            self.start_time,
            self.end_time,
            self.status,
            self.additional,
        ]


class AbstractRun(ABC):
    prefix: str

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.path: Optional[Path] = None
        self.logger = get_logger(self.__class__.__name__)

        # objects created by reset
        self.reset()

    def reset(self) -> None:
        self.meta: Dict[str, Any] = {}
        self.configspace: Optional[ConfigSpace.ConfigurationSpace] = None
        self.configs: Dict[int, Configuration] = {}
        self.origins: Dict[int, str] = {}
        self.models: Dict[int, Optional[Union[str, "torch.nn.Module"]]] = {}

        self.history: List[Trial] = []
        self.trial_keys: Dict[Tuple[str, int], int] = {}

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Hash of the current run. If hash changes, cache has to be cleared. This ensures that
        the cache always holds the latest results of the run.

        Returns
        -------
        str
            Hash of the run.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Hash of the file. This is used to identify the file.
        In contrast to `hash`, this hash should not be changed throughout the run.

        Returns
        -------
        str
            Hash of the run.
        """
        pass

    @staticmethod
    def get_trial_key(config_id: int, budget: Union[int, float]):
        return (config_id, budget)

    def get_trial(self, trial_key) -> Optional[Trial]:
        if trial_key not in self.trial_keys:
            return None

        return self.history[self.trial_keys[trial_key]]

    def get_trials(self) -> Iterator[Trial]:
        yield from self.history

    def get_meta(self) -> None:
        return self.meta

    def empty(self) -> None:
        return len(self.history) == 0

    def get_objectives(self) -> List[Objective]:
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

    def get_objective(self, id: Union[str, int]) -> Optional[Objective]:
        """Returns the objective based on the id or the name.

        Parameters
        ----------
        id : Union[str, int]
            The id or name of the objective.

        Returns
        -------
        Objective
            The objective object.
        """

        objectives = self.get_objectives()
        if type(id) == int:
            return objectives[id]

        # Otherwise, iterate till the name is found
        for objective in objectives:
            if objective["name"] == id:
                return objective

        return None

    def get_objective_id(self, objective: Union[Objective, str]) -> Optional[int]:
        """
        Returns the id of the objective if it is found.

        Parameters
        ----------
        objective : Union[Objective, str]
            The objective for which the id is returned.

        Returns
        -------
        Optional[int]
            Objective id or None if not found.
        """
        objectives = self.get_objectives()
        for id, objective2 in enumerate(objectives):
            if isinstance(objective, Objective):
                if objective == objective2:
                    return id
            else:
                if objective == objective2["name"]:
                    return id

        return None

    def get_objective_name(self, objectives: Optional[List[Objective]] = None) -> str:
        """
        Get the cost name of given objective names.
        Returns "Combined Cost" if multiple objective names were involved.
        """

        available_objective_names = self.get_objective_names()

        if objectives is None:
            if len(available_objective_names) == 1:
                return available_objective_names[0]
        else:
            if len(objectives) == 1:
                return objectives[0]["name"]

        return "Combined Cost"

    def get_objective_names(self) -> List[str]:
        return [obj["name"] for obj in self.get_objectives()]

    def get_configs(self, budget: Union[int, float] = None) -> Dict[int, Configuration]:
        configs = {}
        for trial in self.history:
            if budget is not None:
                if budget != trial.budget:
                    continue

            if (config_id := trial.config_id) not in configs:
                config = self.get_config(config_id)
                configs[config_id] = config

        return configs

    def get_config(self, id: int) -> Configuration:
        config = Configuration(self.configspace, self.configs[id])
        return config

    def get_config_id(self, config: Union[Configuration, Dict]) -> Optional[int]:
        if isinstance(config, Configuration):
            config = config.get_dictionary()

        # Find out config id
        for id, c in self.configs.items():
            if c == config:
                return id

        return None

    def get_num_configs(self, budget: Union[int, float] = None) -> int:
        return len(self.get_configs(budget=budget))

    def get_budget(self, id: Union[int, str]) -> float:
        """
        Gets the budget given an id.

        Parameters
        ----------
        id : Union[int, str]
            Id of the wanted budget. If id is a string, it is converted to an integer.

        Returns
        -------
        float
            Budget.
        """
        if type(id) == str:
            id = int(id)

        return self.meta["budgets"][id]

    def get_budgets(self, human: bool = False) -> List[Union[int, float]]:
        """
        Returns the budgets from the meta data.

        Parameters
        ----------
        human : bool, optional
            Make the output better readable. By default False.

        Returns
        -------
        List[Union[int, float]]
            List of budgets.
        """
        budgets = self.meta["budgets"]

        if human:
            readable_budgets = []
            for b in budgets:
                if b is not None:
                    readable_budgets += [float(np.round(float(b), 2))]

            return readable_budgets

        return budgets

    def get_highest_budget(self) -> Optional[Union[int, float]]:
        """
        Returns the highest budget. If no budget is available, None is returned.

        Returns
        -------
        Optional[Union[int, float]]
            The highest budget or None if no budget was specified.
        """
        budgets = self.meta["budgets"]
        if len(budgets) == 0:
            return None

        return budgets[-1]

    def _process_costs(self, costs: Iterable[float]) -> List[float]:
        """
        Processes the costs to get rid of NaNs. NaNs are replaced by the worst value of the
        objective.

        Parameters
        ----------
        costs : Iterable[float]
            Costs, which should be processed. Must be the same length as the number of objectives.

        Returns
        -------
        List[float]
            Processed costs without NaN values.
        """
        new_costs = []
        for cost, objective in zip(costs, self.get_objectives()):
            # Replace with the worst cost
            if cost is None:
                cost = objective.get_worst_value()
            new_costs += [cost]

        return new_costs

    def get_cost(self, config_id: int, budget: Union[int, float] = None) -> Optional[List[float]]:
        """
        If no budget is given, the highest budget is chosen.
        """
        costs = self.get_costs(budget)
        if config_id not in costs:
            return None

        return costs[config_id]

    def get_costs(
        self, budget: Optional[Union[int, float]] = None, statuses: Optional[List[Status]] = None
    ) -> Dict[int, List[float]]:
        """
        Get costs with their config ids.

        Parameters
        ----------
        budget : Optional[Union[int, float]], optional
            Budget to select the costs. If no budget is given, the highest budget is chosen.
            By default None.
        statuses : Optional[List[Status]], optional
            Only selected stati are considered. If no status is given, all stati are considered.
            By default None.

        Returns
        -------
        Dict[int, List[float]]
            Costs with their config ids.
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

    def get_incumbent(
        self,
        objectives: Optional[List[Objective]] = None,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[List[Status]] = None,
    ) -> Tuple[float, Configuration]:
        """
        Returns the incumbent with its normalized cost.

        Parameters
        ----------
        objectives : Optional[List[Objective]], optional
            Considerd objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        statuses : Optional[List[Status]], optional
            Considered statuses. By default None. If None, all stati are considered.

        Returns
        -------
        Tuple[Configuration, float]
            Incumbent with its normalized cost.

        Raises
        ------
        RuntimeError
            If no incumbent was found.
        """
        min_cost = np.inf
        best_config_id = None

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():
            cost = self.merge_costs(costs, objectives)

            if cost < min_cost:
                min_cost = cost
                best_config_id = config_id

        if best_config_id is None:
            raise RuntimeError("No incumbent found.")

        config = self.get_config(best_config_id)
        config = Configuration(self.configspace, config)
        normalized_cost = min_cost

        return config, normalized_cost

    def merge_costs(
        self, costs: Iterable[float], objectives: Optional[List[Objective]] = None
    ) -> float:
        """
        Calculates one cost value from multiple costs.
        Normalizes the cost first and weight every cost the same.
        The lower the normalized cost, the better.

        Parameters
        ----------
        costs : Iterable[float]
            The costs, which should be merged. Must be the same length as the original number of objectives.
        objectives : Optional[List[Objective]], optional
            The considered objectives to the costs. By default None.
            If None, all objectives are considered. The passed objectives can differ from the
            original number objectives.

        Returns
        -------
        float
            Merged costs.
        """
        # Get rid of NaN values
        costs = self._process_costs(costs)

        if objectives is None:
            objectives = self.get_objectives()

        if len(costs) != len(self.get_objectives()):
            raise RuntimeError(
                "The number of costs must be the same as the original number of objectives."
            )

        # First normalize
        filtered_objectives = []
        normalized_costs = []
        for objective in objectives:
            objective_id = self.get_objective_id(objective)

            if objective_id is None:
                raise RuntimeError("The objective was not found.")
            cost = costs[objective_id]

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
        objective_weights = [1 / len(objectives) for _ in range(len(objectives))]

        costs = [u * v for u, v in zip(normalized_costs, objective_weights)]
        cost = np.mean(costs).item()

        return cost

    def get_model(self, config_id: int) -> Optional["torch.nn.Module"]:
        import torch

        filename = self.models_dir / f"{str(config_id)}.pth"
        if not filename.exists():
            return None

        return torch.load(filename)

    def get_trajectory(
        self, objective: Objective, budget: Optional[Union[int, float]] = None
    ) -> Tuple[List[float], List[float], List[float], List[int]]:
        if budget is None:
            budget = self.get_highest_budget()

        objective_id = self.get_objective_id(objective)
        if objective_id is None:
            raise RuntimeError("The passed objective is invalid.")

        costs_mean = []
        costs_std = []
        ids = []
        times = []

        order = []
        # Sort self.history by end_time
        for id, trial in enumerate(self.history):
            order.append((id, trial.end_time))

        order.sort(key=lambda tup: tup[1])

        # Important: Objective can be minimized or maximized
        if objective["optimize"] == "lower":
            current_cost = np.inf
        else:
            current_cost = -np.inf

        for id, _ in order:
            trial = self.history[id]

            # Only consider selected/last budget
            if trial.budget != budget:
                continue

            cost = trial.costs[objective_id]
            if cost is None:
                continue

            # Now it's important to check whether the cost was minimized or maximized
            if objective["optimize"] == "lower":
                improvement = cost < current_cost
            else:
                improvement = cost > current_cost

            if improvement:
                current_cost = cost

                costs_mean.append(cost)
                costs_std.append(0)
                times.append(trial.end_time)
                ids.append(id)

        return times, costs_mean, costs_std, ids

    def encode_config(
        self, config: Union[int, Dict[Any, Any], Configuration], specific: bool = False
    ) -> List:
        """
        Encodes a given config (id) to a normalized list.
        If a config is passed, no look-up has to be done.

        Parameters
        ----------
        config : Union[int, Dict[Any, Any], Configuration]
            Either the config id, config as dict, or Configuration itself.
        specific : bool, optional
            Use specific encoding for fanova tree, by default False.

        Returns
        -------
        List
            The encoded config as list.
        """
        if not isinstance(config, Configuration):
            if isinstance(config, int):
                config = self.configs[config]

            config = Configuration(self.configspace, config)

        hps = self.configspace.get_hyperparameters()
        values = list(config.get_array())

        if specific:
            return values

        x = []
        for value, hp in zip(values, hps):
            # NaNs should be encoded as -0.5
            if np.isnan(value):
                value = NAN_VALUE
            # Categorical values should be between 0..1
            elif isinstance(hp, CategoricalHyperparameter):
                value = value / (len(hp.choices) - 1)
            # Constants should be encoded as 1.0 (from 0)
            elif isinstance(hp, Constant):
                value = CONSTANT_VALUE

            x += [value]

        return x

    def get_encoded_configs(
        self,
        objectives: Optional[List[Objective]] = None,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[List[Status]] = None,
        encode_y: bool = False,
        specific: bool = False,
        pandas: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, List[int]], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Encodes configurations to process them further. After the configurations are encoded,
        they can be used in model prediction.

        Parameters
        ----------
        objectives : Optional[List[Objective]], optional
            Which objectives should be considered. By default None. If None, all objectives are
            considered.
        budget : Optional[List[Status]], optional
            Which budget should be considered. By default None. If None, only the highest budget
            is considered.
        statuses : Optional[List[Status]], optional
            Which statuses should be considered. By default None. If None, all statuses are
            considered.
        encode_y : bool, optional
            Whether y should be normalized too. By default False.
        specific : bool, optional
            Whether a specific encoding should be used. This encoding is compatible with fANOVA
            forest implementation. By default False.
        pandas : bool, optional
            Whether the data should be returned as pandas or numpy array. By default False.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray, List[int]], Tuple[pd.DataFrame, pd.DataFrame]]
            Encoded configurations or encoded dataframes if `pandas` equals true.

        Raises
        ------
        ValueError
            If a hyperparameter is not supported.
        """

        X, Y = [], []
        Labels = []
        config_ids = []

        if objectives is None:
            objectives = self.get_objectives()

        results = self.get_costs(budget, statuses)
        for config_id, costs in results.items():
            config = self.configs[config_id]
            x = self.encode_config(config, specific=specific)

            if encode_y:
                y = [self.merge_costs(costs, objectives)]
            else:
                y = []
                # Iterate over the objectives
                for i in range(len(objectives)):
                    y.append(costs[i])

            labels = []
            for hp_name in self.configspace.get_hyperparameter_names():
                # `hp_name` might not be in config (e.g. if hp is inactive)
                if hp_name not in config:
                    label = NAN_LABEL
                else:
                    label = prettify_label(config[hp_name])

                labels += [label]

            # Also prettify y values
            labels += [prettify_label(y_) for y_ in y]

            X.append(x)
            Y.append(y)
            Labels.append(labels)
            config_ids.append(config_id)

        X = np.array(X)  # type: ignore
        Y = np.array(Y)  # type: ignore

        # Imputation: Easiest case is to replace all nans with -1
        # However, since Stefan used different values for inactives
        # we also have to use different inactives to be compatible
        # with the random forests.
        # https://github.com/automl/SMAC3/blob/a0c89502f240c1205f83983c8f7c904902ba416d/smac/epm/base_rf.py#L45
        if specific:
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
            if encode_y:
                cost_columns = [self.get_objective_name(objectives)]
            else:
                cost_columns = [objective["name"] for objective in objectives]

            columns = [name for name in self.configspace.get_hyperparameter_names()] + cost_columns
            data = np.concatenate((X, Y), axis=1)

            df_data = pd.DataFrame(data=data, columns=columns)
            df_labels = pd.DataFrame(data=Labels, columns=columns)

            return df_data, df_labels
        else:
            return X, Y, config_ids


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
