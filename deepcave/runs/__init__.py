#  noqa: D400
"""
# AbstractRun

This module provides utilities to create and handle an abstract run.

It provides functions to get information of the run, as well as the used objectives.

## Classes
    - AbstractRun: Create a new run.
"""

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
    COMBINED_SEED,
    CONSTANT_VALUE,
    NAN_VALUE,
)
from deepcave.runs.exceptions import NotMergeableError, RunInequality
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial
from deepcave.utils.logs import get_logger


class AbstractRun(ABC):
    """
    Create a new run.

    Provide functions to get information of the run, as well as the used objectives.

    Properties
    ----------
    name : str
        The name of the run.
    path : Optional[Path]
        The path to the run.
    logger : Logger
        The logger for the run.
    meta: Dict[str, Any]
        Contains the run's meta information.
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space of the run.
    configs: Dict[int, Configuration]
        Contains the configurations.
    origins: Dict[int, str]
        The origin of the configuration.
    models: Dict[int, Optional[Union[str, "torch.nn.Module"]]]
        Contains the modules.
    history: List[Trial]
        The history of Trials.
    trial_keys: Dict[Tuple[str, int], int]
        Contains config_id, budget and the corresponding trial_id.
    models_dir : Path
        The directory of the model.
    """

    prefix: str

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.path: Optional[Path] = None
        self.logger = get_logger(self.__class__.__name__)

        # objects created by reset
        self.reset()

    def reset(self) -> None:
        """
        Reset the run to default values / empties.

        Clear the initial data and configurations of the object.
        """
        self.meta: Dict[str, Any] = {}
        self.configspace: ConfigSpace.ConfigurationSpace
        self.configs: Dict[int, Union[Configuration, Dict[Any, Any]]] = {}
        self.origins: Dict[int, Optional[str]] = {}
        self.models: Dict[  # type: ignore
            int, Optional[Union[str, "torch.nn.Module"]]  # noqa: F821
        ] = {}

        self.history: List[Trial] = []
        self.trial_keys: Dict[
            Tuple[int, Optional[Union[int, float]], Optional[int]], int
        ] = {}  # (config_id, budget, seed) -> trial_id

        # Cached data
        self._highest_budget: Dict[int, Union[int, float]] = {}  # config_id -> budget

    def _update_highest_budget(
        self, config_id: int, budget: Union[int, float], status: Status
    ) -> None:
        """
        Update the highest budget.

        Parameters
        ----------
        config_id : int
            The identificator of the configuration.
        budget : Union[int, float]
            The new highest budget.
        status : Status
            The status of the run.
        """
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
        Hash of the current run.

        If hash changes, cache has to be cleared. This ensures that
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
        Hash of the file.

        This is used to identify the file.
        In contrast to `hash`, this hash should not be changed throughout the run.

        Returns
        -------
        str
            Hash of the run.
        """
        pass

    @property
    def latest_change(self) -> float:
        """
        Get the latest change.

        Returns
        -------
        float
            The latest change.
        """
        return 0

    @staticmethod
    def get_trial_key(
        config_id: int, budget: Optional[Union[int, float]], seed: Optional[int]
    ) -> Tuple[int, Optional[Union[int, float]], Optional[int]]:
        """
        Get the trial key.

        It is obtained through the assembly of configuration and budget.

        Parameters
        ----------
        config_id : int
            The identificator of the configuration.
        budget : Optional[Union[int, float]]
            The budget of the Trial.
        seed: Optional[int]
            The seed used for the Trial.

        Returns
        -------
        Tuple[int, Optional[Union[int, float]], Optional[int]]
            Tuple representing the trial key, consisting of configuration id, budget, and seed.
        """
        return (config_id, budget, seed)

    def get_trial(self, trial_key: tuple[int, Union[int, float], int]) -> Optional[Trial]:
        """
        Get the trial with the responding key if existing.

        Parameters
        ----------
        trial_key : Tuple[int, Union[int, float], int]
            The key for the desired trial.

        Returns
        -------
        Optional[Trial]
            The trial object.
        """
        if trial_key not in self.trial_keys:
            return None

        return self.history[self.trial_keys[trial_key]]

    def get_trials(self) -> Iterator[Trial]:
        """
        Get an iterator of all stored trials.

        Returns
        -------
        Iterator[Trial]
            An iterator over all stored trials.
        """
        yield from self.history

    def get_meta(self) -> Dict[str, Any]:
        """
        Get a shallow copy of the meta information.

        Returns
        -------
        Dict[str, Any]
            A shallow copy of the meta information dictionary.
        """
        return self.meta.copy()

    def empty(self) -> bool:
        """
        Check if the run's history is empty.

        Returns
        -------
        bool
            True if run history is empty, False otherwise.
        """
        return len(self.history) == 0

    def get_origin(self, config_id: int) -> Optional[str]:
        """
        Get the origin, given a config id.

        Parameters
        ----------
        config_id : int
            The identificator of the configuration.

        Returns
        -------
        Optional[str]
            An origin string corresponding to the given configuration id.
        """
        return self.origins[config_id]

    def get_objectives(self) -> List[Objective]:
        """
        Get a list of all objectives corresponding to the run.

        Returns
        -------
        List[Objective]
            A list containing all objectives associated with the run.
        """
        objectives = []
        for d in self.meta["objectives"].copy():
            objective = Objective.from_json(d)
            objectives += [objective]

        return objectives

    def get_objective(self, id: Union[str, int]) -> Optional[Objective]:
        """
        Return the objective based on the id or the name.

        Parameters
        ----------
        id : Union[str, int]
            The id or name of the objective.

        Returns
        -------
        Optional[Objective]
            The objective.
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
        Return the id of the objective if it is found.

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
        """
        Get the ids of the objectives.

        Returns
        -------
        List[int]
            A list of the ids of the objectives.
        """
        return list(range(len(self.get_objectives())))

    def get_objective_name(self, objectives: Optional[List[Objective]] = None) -> str:
        """
        Get the cost name of given objective names.

        Returns "Combined Cost" if multiple objective names were involved.

        Parameters
        ----------
        objectives : Optional[List[Objective]]
            A list of the objectives.
            By default None.

        Returns
        -------
        str
            The name of the objective.
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
        """
        Get the names of the objectives.

        Returns
        -------
        List[str]
            A list containing the names of the objectives.
        """
        return [obj.name for obj in self.get_objectives()]

    def get_configs(
        self,
        budget: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
    ) -> Dict[int, Configuration]:
        """
        Get configurations of the run.

        Optionally, only configurations which were evaluated on the passed budget, seed,
        and stati are considered.

        Parameters
        ----------
        budget : Optional[Union[int, float]]
            Budget to select the configs. If no budget is given, all seeds are considered.
            By default None.
        seed: Optional[int]
            Seed to select the configs. If no seed is given, all seeds are considered.
            By default None.
        statuses : Optional[Union[Status, List[Status]]]
            Only selected stati are considered. If no status is given, all stati are considered.
            By default None.


        Returns
        -------
        Dict[int, Configuration]
            Configuration id and the configuration.
        """
        # Include all configs if budget is a combined budget
        if budget == COMBINED_BUDGET:
            budget = None

        # Include all configs if we have combined seed
        if seed == COMBINED_SEED:
            seed = None

        configs = {}
        for trial in self.history:
            if budget is not None:
                if budget != trial.budget:
                    continue

            if seed is not None:
                if seed != trial.seed:
                    continue

            if statuses is not None:
                if isinstance(statuses, Status):
                    statuses = [statuses]

                if trial.status not in statuses:
                    continue

            if (config_id := trial.config_id) not in configs:
                config = self.get_config(config_id)
                configs[config_id] = config

        # Sort dictionary
        configs = dict(sorted(configs.items()))

        return configs

    def get_config(self, id: int) -> Configuration:
        """
        Retrieve the configuration with the corresponding id.

        Parameters
        ----------
        id : int
            The id of the configuration.

        Returns
        -------
        Configuration
            The corresponding Configuration.
        """
        config = Configuration(self.configspace, self.configs[id])
        return config

    def get_config_id(self, config: Union[Configuration, Dict]) -> Optional[int]:
        """
        Get the id of the configuration.

        Parameters
        ----------
        config : Union[Configuration, Dict]
            The configuration for which to find the id.

        Returns
        -------
        Optional[int]
            The configuration id.
        """
        if isinstance(config, Configuration):
            config = config.get_dictionary()

        # Find out config id
        for id, c in self.configs.items():
            if c == config:
                return id

        return None

    def get_num_configs(
        self, budget: Optional[Union[int, float]] = None, seed: Optional[int] = None
    ) -> int:
        """
        Count the number of configurations stored in this run with a specific budget.

        Parameters
        ----------
        budget : Optional[Union[int, float]]
            The budget for which to count the configurations.
            If not provided, counts all configurations.
            Default is None.
        seed : Optional[int]
            The seed for which to count the configurations.
            If not provided, counts all configurations.
            Default is None.

        Returns
        -------
        int
            The number of all configurations with a given budget.
            If budget is None, counts all configurations.
        """
        return len(self.get_configs(budget=budget, seed=seed))

    def get_budget(self, id: Union[int, str], human: bool = False) -> Union[int, float]:
        """
        Get the budget given an id.

        Parameters
        ----------
        id : Union[int, str]
            The id of the wanted budget.
            If id is a string, it is converted to an integer.
        human : bool, optional
            Make the output more readable.
            By default False.

        Returns
        -------
        float, int
            The budget.

        Raises
        ------
        TypeError
            If the budget with this id is invalid.
        """
        budgets = self.get_budgets(human=human)
        return budgets[int(id)]  # type: ignore

    def get_budget_ids(self, include_combined: bool = True) -> List[int]:
        """
        Get the corresponding ids for the budgets.

        Parameters
        ----------
        include_combined : bool, optional
            If False, cut last id of budget ids.
            By default True.

        Returns
        -------
        List[int]
            A list of the budget ids.
        """
        budget_ids = list(range(len(self.get_budgets())))
        if not include_combined:
            budget_ids = budget_ids[:-1]

        return budget_ids

    def get_budgets(
        self, human: bool = False, include_combined: bool = True
    ) -> List[Union[int, float, str]]:
        """
        Return the budgets from the meta data.

        Parameters
        ----------
        human : bool, optional
            Make the output more readable.
            By default False.
        include_combined : bool, optional
            If True, include the combined budget.
            By default True.

        Returns
        -------
        List[Union[int, float, str]]
            List of budgets. In a readable form, if human is True.
        """
        budgets = self.meta["budgets"].copy()
        if include_combined and len(budgets) > 1 and COMBINED_BUDGET not in budgets:
            budgets += [COMBINED_BUDGET]

        if human:
            readable_budgets: List[Union[str, float]] = []
            for b in budgets:
                if b == COMBINED_BUDGET:
                    readable_budgets += ["Combined"]
                elif b is not None:
                    readable_budgets += [float(np.round(float(b), 2))]

            return readable_budgets

        return budgets

    def get_highest_budget(self, config_id: Optional[int] = None) -> Optional[Union[int, float]]:
        """
        Return the highest found budget for a config id.

        If no config id is specified then
        the highest available budget is returned.
        Moreover, if no budget is available None is returned.

        Parameters
        ----------
        config_id : Optional[int]
            The config id for which the highest budget is returned.

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

    def get_seeds(
        self, human: bool = False, include_combined: bool = True
    ) -> List[Union[int, str]]:
        """
        Return the seeds from the meta data.

        Parameters
        ----------
        human : bool, optional
            Make the output better readable. By default False.
        include_combined : bool, optional
            If true, return combined seed as well. By default True.

        Returns
        -------
        List[Union[int, str]]
            List of seeds.
        """
        seeds = self.meta["seeds"].copy()
        if include_combined and len(seeds) > 1 and COMBINED_SEED not in seeds:
            seeds += [COMBINED_SEED]

        if human:
            readable_seeds: List[Union[int, str]] = []
            for s in seeds:
                if s == COMBINED_SEED:
                    readable_seeds += ["Combined"]
                elif s is not None:
                    readable_seeds += [int(s)]

            return readable_seeds

        return seeds

    def _process_costs(self, costs: List[float]) -> List[float]:
        """
        Process the costs to get rid of NaNs.

        NaNs are replaced by the worst value of the
        objective.

        Parameters
        ----------
        costs : List[float]
            Costs, which should be processed.
            Must be the same length as the number of objectives.

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

    def get_avg_costs(
        self,
        config_id: int,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Get average costs over all seeds for a config.

        Optionally, only configurations which were evaluated on the passed budget and stati
        are considered.

        In case of multi-objective, multiple costs are returned in the form of a list.

        Parameters
        ----------
        config_id : int
            Configuration id to get the costs for.
        budget : Optional[Union[int, float]]
            Budget to get the costs from the configuration id for. If budget is
            None, the highest budget is chosen. By default None.
        statuses : Optional[Union[Status, List[Status]]]
            Only selected stati are considered. If no status is given, all stati are considered.
            By default None.

        Returns
        -------
        List[float]
            List of average cost values for the given config_id and budget.
        List[float]
            List of std cost values for the given config_id and budget.
        """
        objectives = self.get_objectives()

        # Budget might not be evaluated
        all_costs = self.get_all_costs(budget=budget, statuses=statuses)
        if config_id in all_costs:
            config_costs = all_costs[config_id]
        else:
            raise ValueError(f"No costs available for config_id {config_id}.")

        avg_costs, std_costs = [], []
        for idx in range(len(objectives)):
            costs = [values[idx] for values in config_costs.values() if values[idx] is not None]
            avg_costs.append(float(np.mean(costs)))
            std_costs.append(float(np.std(costs)))
        return avg_costs, std_costs

    def get_all_costs(
        self,
        budget: Optional[Union[int, float]] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
        seed: Optional[int] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> Dict[int, Dict[int, List[float]]]:
        """
        Get all costs in the history with their config ids and seeds.

        Optionally, only configurations which were evaluated on the passed budget, seed, and stati
        are considered.

        In case of multi-objective, multiple costs are returned in the form of a list.

        Parameters
        ----------
        budget : Optional[Union[int, float]]
            Budget to select the costs. If no budget is given, the highest budget is chosen.
            By default, None.
        statuses : Optional[Union[Status, List[Status]]]
            Only selected stati are considered. If no status is given, all stati are considered.
            By default, None.
        seed : Optional[int], optional
            Seed to select the costs. If no seed is given, all seeds are considered.
            By default, None.
        selected_ids: Optional[List[int]], optional
            If set, only history ids in the list will be considered. By default, None.

        Returns
        -------
        Dict[int, Dict[int, List[float]]]
            Config ids and seeds with their corresponding list of costs.
        """
        if budget is None:
            budget = self.get_highest_budget()

        # In case of COMBINED_BUDGET, only the costs of the highest found budget are kept
        highest_evaluated_budget = {}

        results: Dict[int, Dict[int, List[float]]] = {}
        if selected_ids is not None:
            history = [self.history[i] for i in selected_ids]
        else:
            history = self.history
        for trial in history:
            if statuses is not None:
                if isinstance(statuses, Status):
                    statuses = [statuses]

                if trial.status not in statuses:
                    continue

            if seed is not None:
                if trial.seed != seed:
                    continue

            if budget == COMBINED_BUDGET:
                if trial.config_id not in highest_evaluated_budget:
                    highest_evaluated_budget[trial.config_id] = trial.budget

                latest_budget = highest_evaluated_budget[trial.config_id]
                # Only the highest budget is kept
                if trial.budget >= latest_budget:
                    if trial.config_id not in results:
                        results[trial.config_id] = {}
                    results[trial.config_id][trial.seed] = trial.costs
            else:
                if trial.budget is not None:
                    if trial.budget != budget:
                        continue
                if trial.config_id not in results:
                    results[trial.config_id] = {}
                results[trial.config_id][trial.seed] = trial.costs
        return results

    def get_status(
        self,
        config_id: int,
        seed: int,
        budget: Optional[Union[int, float]] = None,
    ) -> Status:
        """
        Return the status of a trial (i.e. configuration, budget and seed).

        Parameters
        ----------
        config_id : int
            Configuration id to get the status for.
        seed : Optional[int]
            Seed to get the status from the configuration id for.
        budget : Optional[Union[int, float]]
            Budget to get the status from the configuration id for. If budget is
            None, the highest budget is chosen. By default None.

        Returns
        -------
        Status
            Status of the configuration.

        Raises
        ------
        ValueError
            If the configuration id is not found.
        """
        if budget == COMBINED_BUDGET:
            return Status.NOT_EVALUATED

        if budget is None:
            budget = self.get_highest_budget()

        if config_id not in self.configs:
            raise ValueError("Configuration id was not found.")

        trial_key = self.get_trial_key(config_id, budget, seed)

        # Unfortunately, it is necessary to iterate through the history to find the status
        # TODO: Cache the stati
        for trial in self.history:
            if trial_key == trial.get_key():
                return trial.status

        return Status.NOT_EVALUATED

    def get_incumbent(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> Tuple[Configuration, float]:
        """
        Return the incumbent with its normalized objective value.

        The incumbent is the configuration with the lowest normalized objective value.

        Optionally, only configurations which were evaluated on the passed budget, seed,
        and stati are considered.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. If None, all objectives are considered. By default None.
        budget : Optional[Union[int, float]], optional
            Considered budget. If None, the highest budget is chosen. By default None.
        seed : Optional[int], optional
            Considered seed. If no seed is given, all seeds are considered. By default None.
        statuses : Optional[Union[Status, List[Status]]], optional
            Considered stati. If None, all stati are considered. By default None.
        selected_ids: Optional[List[int]], optional
            If set, only ids in selected_ids will be considered. This can for example be
            useful if only ids up to a certain end-time shall be considered. By default None.

        Returns
        -------
        Tuple[Configuration, float]
            Incumbent with its normalized cost.

        Raises
        ------
        RuntimeError
            If no incumbent was found.
        """
        if isinstance(objectives, Objective):
            single_objective = objectives
        elif isinstance(objectives, list) and len(objectives) == 1:
            single_objective = objectives[0]
        else:
            single_objective = None

        if single_objective is not None and single_objective.optimize == "upper":
            best_cost = -np.inf
        else:
            best_cost = np.inf
        best_config_id = None

        results = self.get_all_costs(
            budget=budget, statuses=statuses, seed=seed, selected_ids=selected_ids
        )

        if len(results) == 0:
            raise RuntimeError("No data available, thus no incumbent found.")

        seed_count = {}
        for config_id, seed_costs_dict in results.items():
            seed_count[config_id] = len(seed_costs_dict)
        max_seed_count = max(seed_count.values())

        for config_id, seed_costs_dict in results.items():
            # If there are multiple seeds, only configurations evaluated on all seeds are
            # considered. From these configurations, the one with the highest average objective
            # value over the seeds is considered as the incumbent.
            if max_seed_count > 1:
                if len(seed_costs_dict) < max_seed_count:
                    continue

                # Get average over all seeds
                config_costs = np.zeros([max_seed_count, len(self.get_objectives())])
                for i, (_, seed_costs) in enumerate(seed_costs_dict.items()):
                    config_costs[i] = seed_costs
                avg_cost = np.mean(config_costs, axis=0)

            # If there is only one seed, the costs can be used directly
            else:
                avg_cost = [*seed_costs_dict.values()][0]

            # If there are multiple objectives, the costs are merged to one cost value
            if single_objective is None:
                cost = self.merge_costs(avg_cost, objectives)
            else:
                cost = avg_cost[self.get_objective_id(single_objective)]

            if cost is None:
                continue

            if single_objective is not None and single_objective.optimize == "upper":
                if cost > best_cost:
                    best_cost = cost
                    best_config_id = config_id
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_config_id = config_id

        if best_config_id is None:
            raise RuntimeError("No incumbent found.")

        config = self.get_config(best_config_id)
        config = Configuration(self.configspace, config)
        normalized_cost = best_cost

        return config, normalized_cost

    def merge_costs(
        self, costs: List[float], objectives: Optional[Union[Objective, List[Objective]]] = None
    ) -> float:
        """
        Calculate one cost value from multiple costs.

        Normalizes the costs first and weigh every cost the same.
        The lower the normalized cost, the better.

        Parameters
        ----------
        costs : List[float]
            The costs, which should be merged. Must be the same length as the original number of
            objectives.
        objectives : Optional[List[Objective]]
            The considered objectives to the costs. By default None.
            If None, all objectives are considered. The passed objectives can differ from the
            original number objectives.

        Returns
        -------
        float
            Merged costs.

        Raises
        ------
        RuntimeError
            If the number of costs is different from the original number of objectives.
            If the objective was not found.
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

            # The lower is optimized
            # So the normalized cost needs to be flipped
            if objective.optimize == "upper":
                normalized_cost = 1 - normalized_cost

            normalized_costs.append(normalized_cost)
            filtered_objectives.append(objective)

        # Give the same weight to all objectives (for now)
        objective_weights = [1 / len(objectives) for _ in range(len(objectives))]

        costs = [u * v for u, v in zip(normalized_costs, objective_weights)]
        cost = np.mean(costs).item()

        return cost

    def get_model(self, config_id: int) -> Optional["torch.nn.Module"]:  # type: ignore # noqa: F821
        """
        Get a model associated with the configuration id.

        Parameters
        ----------
        config_id : int
            The configuration id.

        Returns
        -------
        Optional["torch.nn.Module]
            A model for the provided configuration id.
        """
        import torch

        filename = self.models_dir / f"{str(config_id)}.pth"  # type: ignore
        if not filename.exists():
            return None

        return torch.load(filename)

    def get_trajectory(
        self,
        objective: Objective,
        budget: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[float], List[float], List[float], List[int], List[int]]:
        """
        Calculate the trajectory of the given objective, budget, and seed.

        Parameters
        ----------
        objective : Objective
            Objective to calculate the trajectory for.
        budget : Optional[Union[int, float]]
            Budget to calculate the trajectory for. If no budget is given, then the highest budget
            is chosen. By default None.
        seed : Optional[int], optional
            Seed to calculate the trajectory for. If no seed is given, then all seeds are
            considered. By default None.

        Returns
        -------
        Tuple[List[float], List[float], List[float], List[int], List[int]]
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

        costs_mean = []
        costs_std = []
        ids = []
        config_ids = []
        times = []

        order = []

        # Sort self.history by end-time
        for id, trial in enumerate(self.history):
            order.append((id, trial.end_time))
        order.sort(key=lambda tup: tup[1])

        # Important: Objective can be minimized or maximized
        if objective.optimize == "lower":
            current_cost = np.inf
        else:
            current_cost = -np.inf

        # Iterate over the history ordered by end-time and calculate the current incumbent
        for i, (id, _) in enumerate(order):
            trial = self.history[id]

            # Get the incumbent over all trials up to this point
            try:
                _, cost = self.get_incumbent(
                    objectives=objective,
                    budget=budget,
                    seed=seed,
                    selected_ids=[selected_id for selected_id, _ in order[: i + 1]],
                )
            except RuntimeError:
                continue

            # Now it's important to check whether the cost was minimized or maximized
            if objective.optimize == "lower":
                improvement = cost < current_cost
            else:
                improvement = cost > current_cost

            if improvement:
                current_cost = cost

                costs_mean.append(cost)
                costs_std.append(0.0)
                times.append(trial.end_time)
                ids.append(id)
                config_ids.append(trial.config_id)

        return times, costs_mean, costs_std, ids, config_ids

    def encode_config(
        self, config: Union[int, Dict[Any, Any], Configuration], specific: bool = False
    ) -> List:
        """
        Encode a given configuration (id) to a normalized list.

        If a configuration is passed, no look-up has to be done.

        Parameters
        ----------
        config : Union[int, Dict[Any, Any], Configuration]
            Either the configuration id, as configuration as dict,
            or a Configuration itself.
        specific : bool
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
        """
        Encode a list of configurations into a corresponding numpy array.

        Parameters
        ----------
        configs : List[Configuration]
            A list containing the configurations to be encoded.

        Returns
        -------
        np.ndarray
            A numpy array with the encoded configurations.
        """
        x_set = []
        for config in configs:
            x = self.encode_config(config)
            x_set.append(x)

        return np.array(x_set)

    def get_encoded_data(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        statuses: Optional[Union[Status, List[Status]]] = None,
        specific: bool = False,
        include_config_ids: bool = False,
        include_combined_cost: bool = False,
    ) -> pd.DataFrame:
        """
        Encode configurations to process them further.

        After the configurations are encoded,
        they can be used in model prediction.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]]
            Which objectives should be considered. If None, all objectives are considered.
            By default None.
        budget : Optional[List[Status]]
            Which budget should be considered. If None, only the highest budget is considered.
            By default None.
        seed: Optional[int]
            Which seed should be considered. If None, all seeds are considered.
            By default None.
        statuses : Optional[Union[Status, List[Status]]]
            Which stati should be considered. If None, all stati are considered.
            By default None.
        specific : bool
            Whether a specific encoding should be used. This encoding is compatible with pyrfr.
            A wrapper for pyrfr is implemented in ``deepcave.evaluators.epm``.
            By default False.
        include_config_ids : bool
            Whether to include configuration ids. By default False.
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
            If a hyperparameter (HP) is not supported.
        """
        if objectives is None:
            objectives = self.get_objectives()

        if isinstance(objectives, Objective):
            objectives = [objectives]

        x_set, y_set = [], []
        config_ids = []

        results = self.get_all_costs(budget, statuses, seed)
        for config_id, config_costs in results.items():
            config = self.configs[config_id]
            for seed, costs in config_costs.items():
                x = self.encode_config(config, specific=specific)
                y = []

                # Add all objectives
                for objective in objectives:
                    objective_id = self.get_objective_id(objective)
                    y += [costs[objective_id]]

                # Add combined cost
                if include_combined_cost:
                    y += [self.merge_costs(costs, objectives)]

                x_set.append(x)
                y_set.append(y)
                config_ids.append(config_id)

        x_set_array = np.array(x_set)
        y_set_array = np.array(y_set)
        config_ids_array = np.array(config_ids).reshape(-1, 1)

        # Imputation: Easiest case is to replace all nans with -1
        # However, since Stefan used different values for inactive hyperparameters,
        # Also different inactive hyperparameters have to be used, to be compatible
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
                    non_finite_mask = ~np.isfinite(x_set_array[:, idx])
                    x_set_array[non_finite_mask, idx] = impute_values[idx]

        # Now dataframes are created for both values and labels
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
            data: np.ndarray = np.concatenate((config_ids_array, x_set_array, y_set_array), axis=1)
        else:
            data = np.concatenate((x_set_array, y_set_array), axis=1)

        data = pd.DataFrame(data=data, columns=columns)

        return data


def check_equality(
    runs: List[AbstractRun],
    meta: bool = False,
    configspace: bool = True,
    objectives: bool = True,
    budgets: bool = True,
    seeds: bool = False,
) -> Dict[str, Any]:
    """
    Check the passed runs on equality based on the selected runs.

    Return the requested attributes.

    Parameters
    ----------
    runs : list[AbstractRun]
        Runs to check for equality.
    meta : bool, optional
        Meta-Data excluding objectives and budgets, by default False.
    configspace : bool, optional
        Wheter to include the configuration space, by default True.
    objectives : bool, optional
        Wheter to include the objectives, by default True.
    budgets : bool, optional
        Whether to include the budgets, by default True.
    seeds : bool, optional
        Whether to include the seeds, by default False.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the checked attributes.

    Raises
    ------
    NotMergeableError
        If the meta data of the runs are not equal.
        If the configuration spaces of the runs are not equal.
        If the budgets of the runs are not equal.
        If the objective of the runs are not equal.
    """
    result: Dict[str, Any] = {}

    if len(runs) == 0:
        return result

    # Check if objectives are mergeable
    if objectives:
        o1 = None
        for run in runs:
            o2 = run.get_objectives()

            if o1 is None:
                o1 = o2
                continue

            if len(o1) != len(o2):
                raise NotMergeableError(
                    "Objectives of runs are not equal.", RunInequality.INEQ_OBJECTIVE
                )

            for o1_, o2_ in zip(o1, o2):
                try:
                    o1_.merge(o2_)
                except NotMergeableError:
                    raise NotMergeableError(
                        "Objectives of runs are not equal.", RunInequality.INEQ_OBJECTIVE
                    )

        assert o1 is not None
        serialized_objectives = [o.to_json() for o in o1]
        result["objectives"] = serialized_objectives
        if meta:
            result["meta"]["objectives"] = serialized_objectives

    # Also check if budgets are the same
    if budgets:
        b1 = runs[0].get_budgets(include_combined=False)
        for run in runs:
            b2 = run.get_budgets(include_combined=False)
            if b1 != b2:
                raise NotMergeableError("Budgets of runs are not equal.", RunInequality.INEQ_BUDGET)

        result["budgets"] = b1
        if meta:
            result["meta"]["budgets"] = b1

    # Make sure the same configspace is used
    # Otherwise it does not make sense to merge
    # the histories
    if configspace:
        cs1 = runs[0].configspace
        for run in runs:
            cs2 = run.configspace
            if cs1 != cs2:
                raise NotMergeableError(
                    "Configspace of runs are not equal.", RunInequality.INEQ_CONFIGSPACE
                )

        result["configspace"] = cs1

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
                    raise NotMergeableError(
                        "Meta data of runs are not equal.", RunInequality.INEQ_META
                    )

        result["meta"] = m1

    # Check if seeds are the same
    if seeds:
        s1 = runs[0].get_seeds(include_combined=False)
        for run in runs:
            s2 = run.get_seeds(include_combined=False)
            if s1 != s2:
                raise NotMergeableError("Seeds of runs are not equal.", RunInequality.INEQ_SEED)

        result["seeds"] = s1
        if meta:
            result["meta"]["seeds"] = s1

    return result
