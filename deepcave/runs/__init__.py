from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from pathlib import Path

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

from deepcave.constants import (
    COMBINED_BUDGET,
    COMBINED_COST_NAME,
    CONSTANT_VALUE,
    NAN_VALUE,
)
from deepcave.runs.exceptions import NotMergeableError
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial
from deepcave.utils.logs import get_logger


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
        self.configspace: ConfigSpace.ConfigurationSpace
        self.configs: Dict[int, Configuration] = {}
        self.origins: Dict[int, str] = {}
        self.models: Dict[int, Optional[Union[str, "torch.nn.Module"]]] = {}

        self.history: List[Trial] = []
        self.trial_keys: Dict[Tuple[str, int], int] = {}  # (config_id, budget) -> trial_id

        # Cached data
        self._highest_budget: Dict[int, Union[int, float]] = {}  # config_id -> budget

    def _update_highest_budget(
        self, config_id: int, budget: Union[int, float], status: Status
    ) -> None:
        if status == Status.SUCCESS:
            # Update highest budget
            if config_id not in self._highest_budget:
                self._highest_budget[config_id] = budget
            else:
                if budget > self._highest_budget[config_id]:
                    self._highest_budget[config_id] = budget

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Hash of the current run. If hash changes, cache has to be cleared. This ensures that
        the cache always holds the latest results of the run.

        Returns
        -------
        hash : str
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

    @property
    def latest_change(self) -> float:
        return 0

    @staticmethod
    def get_trial_key(config_id: int, budget: Union[int, float]):
        return (config_id, budget)

    def get_trial(self, trial_key) -> Optional[Trial]:
        if trial_key not in self.trial_keys:
            return None

        return self.history[self.trial_keys[trial_key]]

    def get_trials(self) -> Iterator[Trial]:
        yield from self.history

    def get_meta(self) -> Dict[str, Any]:
        return self.meta.copy()

    def empty(self) -> bool:
        return len(self.history) == 0

    def get_origin(self, config_id: int) -> str:
        return self.origins[config_id]

    def get_objectives(self) -> List[Objective]:
        objectives = []
        for d in self.meta["objectives"].copy():
            objective = Objective.from_json(d)
            objectives += [objective]

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
            if objective.name == id:
                return objective

        return None

    def get_objective_id(self, objective: Union[Objective, str]) -> int:
        """
        Returns the id of the objective if it is found.

        Parameters
        ----------
        objective : Union[Objective, str]
            The objective or objective name for which the id is returned.

        Returns
        -------
        objective_id : int
            Objective id from the passed objective.

        Raises
        ------
        RuntimeError
            If objective was not found.
        """
        objectives = self.get_objectives()
        for id, objective2 in enumerate(objectives):
            if isinstance(objective, Objective):
                if objective == objective2:
                    return id
            else:
                if objective == objective2.name:
                    return id

        raise RuntimeError("Objective was not found.")

    def get_objective_ids(self) -> List[int]:
        return list(range(len(self.get_objectives())))

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
                return objectives[0].name

        return COMBINED_COST_NAME

    def get_objective_names(self) -> List[str]:
        return [obj.name for obj in self.get_objectives()]

    def get_configs(self, budget: Union[int, float] = None) -> Dict[int, Configuration]:
        """
        Get configurations of the run. Optionally, only configurations which were evaluated
        on the passed budget are considered.

        Parameters
        ----------
        budget : Union[int, float], optional
            Considered budget. By default None (all configurations are included).

        Returns
        -------
        Dict[int, Configuration]
            Configuration id and the configuration.
        """
        # Include all configs if we have combined budget
        if budget == COMBINED_BUDGET:
            budget = None

        configs = {}
        for trial in self.history:
            if budget is not None:
                if budget != trial.budget:
                    continue

            if (config_id := trial.config_id) not in configs:
                config = self.get_config(config_id)
                configs[config_id] = config

        # Sort dictionary
        configs = dict(sorted(configs.items()))

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

    def get_budget(self, id: Union[int, str], human=False) -> float:
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
        budgets = self.get_budgets(human=human)
        return budgets[int(id)]

    def get_budget_ids(self, include_combined: bool = True) -> List[int]:
        budget_ids = list(range(len(self.get_budgets())))
        if not include_combined:
            budget_ids = budget_ids[:-1]

        return budget_ids

    def get_budgets(
        self, human: bool = False, include_combined: bool = True
    ) -> List[Union[int, float]]:
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
        budgets = self.meta["budgets"].copy()
        if include_combined and len(budgets) > 1 and COMBINED_BUDGET not in budgets:
            budgets += [COMBINED_BUDGET]

        if human:
            readable_budgets = []
            for b in budgets:
                if b == COMBINED_BUDGET:
                    readable_budgets += ["Combined"]
                elif b is not None:
                    readable_budgets += [float(np.round(float(b), 2))]

            return readable_budgets

        return budgets

    def get_highest_budget(self, config_id: Optional[int] = None) -> Optional[Union[int, float]]:
        """
        Returns the highest found budget for a config id. If no config id is specified then
        the highest available budget is returned.
        Moreover, if no budget is available None is returned.

        Returns
        -------
        Optional[Union[int, float]]
            The highest budget or None if no budget was specified.
        """
        if config_id is None:
            budgets = self.meta["budgets"]
            if len(budgets) == 0:
                return None

            return budgets[-1]
        else:
            return self._highest_budget[config_id]

    def _process_costs(self, costs: List[float]) -> List[float]:
        """
        Processes the costs to get rid of NaNs. NaNs are replaced by the worst value of the
        objective.

        Parameters
        ----------
        costs : List[float]
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

    def get_costs(self, config_id: int, budget: Optional[Union[int, float]] = None) -> List[float]:
        """
        Returns the costs of a configuration. In case of multi-objective, multiple costs are
        returned.

        Parameters
        ----------
        config_id : int
            Configuration id to get the costs for.
        budget : Optional[Union[int, float]], optional
            Budget to get the costs from the configuration id for. By default None. If budget is
            None, the highest budget is chosen.

        Raises
        ------
        ValueError
            If the configuration id is not found.
        RuntimeError
            If the budget was not evaluated for the passed config id.

        Returns
        -------
        List[float]
            List of costs from the associated configuration.
        """
        if budget is None:
            budget = self.get_highest_budget()

        if config_id not in self.configs:
            raise ValueError("Configuration id was not found.")

        costs = self.get_all_costs(budget)
        if config_id not in costs:
            raise RuntimeError(f"Budget {budget} was not evaluated for config id {config_id}.")

        return costs[config_id]

    def get_all_costs(
        self,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
    ) -> Dict[int, List[float]]:
        """
        Get all costs in the history with their config ids. Only configs from the given budget
        and statuses are returned.

        Parameters
        ----------
        budget : Optional[Union[int, float]], optional
            Budget to select the costs. If no budget is given, the highest budget is chosen.
            By default None.
        statuses : Optional[Union[Status, List[Status]]], optional
            Only selected stati are considered. If no status is given, all stati are considered.
            By default None.

        Returns
        -------
        Dict[int, List[float]]
            Costs with their config ids.
        """
        if budget is None:
            budget = self.get_highest_budget()

        # In case of COMBINED_BUDGET, we only keep the costs of the highest found budget
        highest_evaluated_budget = {}

        results = {}
        for trial in self.history:
            if statuses is not None:
                if isinstance(statuses, Status):
                    statuses = [statuses]

                if trial.status not in statuses:
                    continue

            if budget == COMBINED_BUDGET:
                if trial.config_id not in highest_evaluated_budget:
                    highest_evaluated_budget[trial.config_id] = trial.budget

                latest_budget = highest_evaluated_budget[trial.config_id]
                # We only keep the highest budget
                if trial.budget >= latest_budget:
                    results[trial.config_id] = trial.costs
            else:
                if trial.budget is not None:
                    if trial.budget != budget:
                        continue

                results[trial.config_id] = trial.costs  # self._process_costs(trial.costs)

        return results

    def get_status(self, config_id: int, budget: Optional[Union[int, float]] = None) -> Status:
        """
        Returns the status of a configuration.

        Parameters
        ----------
        config_id : int
            Configuration id to get the status for.
        budget : Optional[Union[int, float]], optional
            Budget to get the status from the configuration id for. By default None. If budget is
            None, the highest budget is chosen.

        Raises
        ------
        ValueError
            If the configuration id is not found.

        Returns
        -------
        Status
            Status of the configuration.
        """
        if budget == COMBINED_BUDGET:
            return Status.NOT_EVALUATED

        if budget is None:
            budget = self.get_highest_budget()

        if config_id not in self.configs:
            raise ValueError("Configuration id was not found.")

        trial_key = self.get_trial_key(config_id, budget)

        # Unfortunately, we have to iterate through the history to find the status
        # TODO: Cache the stati
        for trial in self.history:
            if trial_key == trial.get_key():
                return trial.status

        return Status.NOT_EVALUATED

    def get_incumbent(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
    ) -> Tuple[Configuration, float]:
        """
        Returns the incumbent with its normalized cost.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considerd objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        statuses : Optional[Union[Status, List[Status]]], optional
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

        results = self.get_all_costs(budget, statuses)
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
        self, costs: List[float], objectives: Optional[Union[Objective, List[Objective]]] = None
    ) -> float:
        """
        Calculates one cost value from multiple costs.
        Normalizes the costs first and weight every cost the same.
        The lower the normalized cost, the better.

        Parameters
        ----------
        costs : List[float]
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

        if isinstance(objectives, Objective):
            objectives = [objectives]

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

            assert objective.lower is not None
            assert objective.upper is not None

            # TODO: What to do if we deal with infinity here?
            assert objective.lower != np.inf
            assert objective.upper != -np.inf

            a = cost - objective.lower
            b = objective.upper - objective.lower
            normalized_cost = a / b

            # We optimize the lower
            # So we need to flip the normalized cost
            if objective.optimize == "upper":
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
    ) -> Tuple[List[float], List[float], List[float], List[int], List[int]]:
        """
        Calculates the trajectory of the given objective and budget.

        Parameters
        ----------
        objective : Objective
            Objective to calculate the trajectory for.
        budget : Optional[Union[int, float]], optional
            Budget to calculate the trajectory for. If no budget is given, then the highest budget
            is chosen. By default None.

        Returns
        -------
        times : List[float]
            Times of the trajectory.
        costs_mean : List[float]
            Costs of the trajectory.
        costs_std : List[float]
            Standard deviation of the costs of the trajectory. This is particularly useful for
            grouped runs.
        ids : List[int]
            The "global" ids of the selected trials.
        config_ids : List[int]
            Config ids of the selected trials.
        """
        if budget is None:
            budget = self.get_highest_budget()

        objective_id = self.get_objective_id(objective)

        costs_mean = []
        costs_std = []
        ids = []
        config_ids = []
        times = []

        order = []
        # Sort self.history by end_time
        for id, trial in enumerate(self.history):
            order.append((id, trial.end_time))

        order.sort(key=lambda tup: tup[1])

        # Important: Objective can be minimized or maximized
        if objective.optimize == "lower":
            current_cost = np.inf
        else:
            current_cost = -np.inf

        for id, _ in order:
            trial = self.history[id]

            # We want to use all budgets
            if budget != COMBINED_BUDGET:
                # Only consider selected/last budget
                if trial.budget != budget:
                    continue

            cost = trial.costs[objective_id]
            if cost is None:
                continue

            # Now it's important to check whether the cost was minimized or maximized
            if objective.optimize == "lower":
                improvement = cost < current_cost
            else:
                improvement = cost > current_cost

            if improvement:
                current_cost = cost

                costs_mean.append(cost)
                costs_std.append(0)
                times.append(trial.end_time)
                ids.append(id)
                config_ids.append(trial.config_id)

        return times, costs_mean, costs_std, ids, config_ids

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

    def encode_configs(self, configs: List[Configuration]) -> np.ndarray:
        X = []
        for config in configs:
            x = self.encode_config(config)
            X.append(x)

        return np.array(X)

    def get_encoded_data(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
        specific: bool = False,
        include_config_ids: bool = False,
        include_combined_cost: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encodes configurations to process them further. After the configurations are encoded,
        they can be used in model prediction.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Which objectives should be considered. If None, all objectives are
            considered. By default None.
        budget : Optional[List[Status]], optional
            Which budget should be considered. By default None. If None, only the highest budget
            is considered.
        statuses : Optional[Union[Status, List[Status]]], optional
            Which statuses should be considered. By default None. If None, all statuses are
            considered.
        encode_y : bool, optional
            Whether y should be normalized too. By default False.
        specific : bool, optional
            Whether a specific encoding should be used. This encoding is compatible with pyrfr.
            A wrapper for pyrfr is implemented in ``deepcave.evaluators.epm``.
            By default False.
        include_config_ids : bool, optional
            Whether to include config ids. By default False.
        include_combined_cost : bool, optional
            Whether to include combined cost. Note that the combined cost is calculated by the
            passed objectives only. By default False.

        Returns
        -------
        df : pd.DataFrame
            Encoded dataframe with the following columns (depending on the parameters):
            [CONFIG_ID, HP1, HP2, ..., HPn, OBJ1, OBJ2, ..., OBJm, COMBINED_COST]

        Raises
        ------
        ValueError
            If a hyperparameter is not supported.
        """

        if objectives is None:
            objectives = self.get_objectives()

        if isinstance(objectives, Objective):
            objectives = [objectives]

        X, Y = [], []
        config_ids = []

        results = self.get_all_costs(budget, statuses)
        for config_id, costs in results.items():
            config = self.configs[config_id]
            x = self.encode_config(config, specific=specific)
            y = []

            # Add all objectives
            for objective in objectives:
                objective_id = self.get_objective_id(objective)
                y += [costs[objective_id]]

            # Add combined cost
            if include_combined_cost:
                y += [self.merge_costs(costs, objectives)]

            X.append(x)
            Y.append(y)
            config_ids.append(config_id)

        X = np.array(X)  # type: ignore
        Y = np.array(Y)  # type: ignore
        config_ids = np.array(config_ids).reshape(-1, 1)  # type: ignore

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
                            raise ValueError("Hyperparameter not supported.")

                if conditional[idx] is True:
                    nonfinite_mask = ~np.isfinite(X[:, idx])
                    X[nonfinite_mask, idx] = impute_values[idx]

        # Now we create dataframes for both values and labels
        # [CONFIG_ID, HP1, HP2, ..., HPn, OBJ1, OBJ2, ..., OBJm, COMBINED_COST]
        if include_config_ids:
            columns = ["config_id"]
        else:
            columns = []

        columns += [name for name in self.configspace.get_hyperparameter_names()]
        columns += [objective.name for objective in objectives]

        if include_combined_cost:
            columns += [COMBINED_COST_NAME]

        if include_config_ids:
            data = np.concatenate((config_ids, X, Y), axis=1)
        else:
            data = np.concatenate((X, Y), axis=1)

        data = pd.DataFrame(data=data, columns=columns)

        return data


def check_equality(
    runs: List[AbstractRun],
    meta: bool = False,
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
        ignore = ["objectives", "budgets", "wallclock_limit"]

        m1 = runs[0].get_meta()
        for run in runs:
            m2 = run.get_meta()

            for k, v in m1.items():
                # Don't check on objectives or budgets
                if k in ignore:
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
        b1 = runs[0].get_budgets(include_combined=False)
        for run in runs:
            b2 = run.get_budgets(include_combined=False)
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

        serialized_objectives = [o.to_json() for o in o1]
        result["objectives"] = serialized_objectives
        if meta:
            result["meta"]["objectives"] = serialized_objectives

    return result
