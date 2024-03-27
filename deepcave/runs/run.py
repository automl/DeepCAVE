#  noqa: D400
"""
# Run

This module provides utilities to create a new run and get its attributes.

## Classes
    - Run: Create a new run.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import json
from pathlib import Path

import ConfigSpace
import jsonlines
import numpy as np
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as cs_json

from deepcave.runs import AbstractRun, Status, Trial
from deepcave.runs.objective import Objective
from deepcave.utils.files import make_dirs
from deepcave.utils.hash import string_to_hash


class Run(AbstractRun, ABC):
    """
    Create a new run and get its attributes.

    If path is given, runs are loaded from the path.

    Properties
    ----------
    configspace : ConfigurationSpace
        The configuration space of the run.
    path : Optional[Union[str, Path]]
        The path of a run to be loaded.
    meta : Dict[str, Any]
        Contains serialized objectives and budgets.
    prefix : str
        The prefix for the id.
    meta_fn : Path
        The path to the meta data.
    configspace_fn : Path
        The path to the configuration space file.
    configs_fn : Path
        The path to the configurations file.
    origins_fn : Path
        The path to the origins file.
    history_fn : Path
        The path to the history file.
    models_dir : Path
        The path to the models directory.
    configs : Dict[int, Configuration]
        Containing the configurations.
    models : Dict[int, Optional[Union[str, "torch.nn.Module"]]]
        Contains the models.
    """

    prefix = "run"
    _initial_order: int

    def __init__(
        self,
        name: str,
        configspace: Optional[ConfigSpace.ConfigurationSpace] = None,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        path: Optional[Path] = None,
    ) -> None:
        super(Run, self).__init__(name)

        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        # Reset and load configspace/path
        self.reset()
        if configspace is not None:
            self.configspace = configspace
        self.path = path
        if self.path is not None:
            self.load()
            return

        if configspace is None and path is None:
            raise RuntimeError(
                "Please provide a configspace or specify a path to load existing trials."
            )

        # Objectives
        if not isinstance(objectives, List):
            objectives = [objectives]

        serialized_objectives = []
        for objective in objectives:
            assert isinstance(objective, Objective)
            serialized_objectives += [objective.to_json()]

        # Meta
        self.meta = {"objectives": serialized_objectives, "budgets": [], "seeds": []}
        self.meta.update(meta)

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> "Run":
        """
        Based on a path, return a new Run object.

        Parameters
        ----------
        path : Path
            The path to get the run from.

        Returns
        -------
        "Run"
            The run loaded from the path.
        """
        pass

    @property
    def id(self) -> str:
        """
        Get a hash as id.

        Returns
        -------
        str
            The hashed id.
        """
        return string_to_hash(f"{self.prefix}:{self.path}")

    @property
    def path(self) -> Optional[Path]:
        """
        Return the path of the run if it exists.

        Returns
        -------
        Optional[Path]
            The path of the run.
        """
        return self._path

    @path.setter
    def path(self, value: Optional[Union[str, Path]]) -> None:
        """
        Set the paths of the run and the JSON files.

        Parameters
        ----------
        value : Optional[Union[str, Path]]
            The path for the directory.
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
        self.models_dir = self._path / "models"

    def exists(self) -> bool:
        """
        Check if the run exists based on the internal path.

        Returns
        -------
        bool
            If run exists.
        """
        if self._path is None:
            return False

        return all(
            f.is_file()
            for f in (
                self.meta_fn,
                self.configspace_fn,
                self.configs_fn,
                self.origins_fn,
                self.history_fn,
            )
        )

    def add(
        self,
        costs: Union[List[float], float],
        config: Union[Dict, Configuration],
        seed: int,
        budget: float = np.inf,
        start_time: float = 0.0,
        end_time: float = 0.0,
        status: Status = Status.SUCCESS,
        origin: Optional[str] = None,
        model: Union[str, "torch.nn.Module"] = None,  # type: ignore # noqa: F821
        additional: Optional[Dict] = None,
    ) -> None:
        """
        Add a trial to the run.

        If combination of config, seed, and budget already exists, it will be overwritten.
        Not successful runs are added with `None` costs.

        Parameters
        ----------
        costs : Union[List[float], float]
            Costs of the run. In case of multi-objective, a list of costs is expected.
        config : Union[Dict, Configuration]
            The corresponding configuration.
        seed : int
            Seed of the run.
        budget : float, optional
            Budget of the run. By default np.inf
        start_time : float, optional
            Start time. By default, 0.0
        end_time : float, optional
            End time. By default, 0.0
        status : Status, optional
            Status of the trial. By default, Status.SUCCESS
        origin : str, optional
            Origin of the trial. By default, None
        model : Union[str, "torch.nn.Module"], optional
            Model of the trial. By default, None
        additional : Optional[Dict], optional
            Additional information of the trial. By default, None.
            Following information is used by DeepCAVE:
            * traceback

        Raises
        ------
        RuntimeError
            If number of costs does not match number of objectives.
        ValueError
            If config id is None.
        """
        if additional is None:
            additional = {}

        if not isinstance(costs, list):
            costs = [costs]

        if len(costs) != len(self.get_objectives()):
            raise RuntimeError("Number of costs does not match number of objectives.")

        updated_objectives = []
        for i in range(len(costs)):
            cost = costs[i]
            objective = self.get_objectives()[i]

            # Update time objective here
            if objective.name == "time" and cost is None:
                costs[i] = end_time - start_time
                cost = costs[i]

            # If cost is none, replace it later with the highest cost
            if cost is not None:
                # Update bounds here
                if not objective.lock_lower and objective.lower is not None:
                    if cost < objective.lower:
                        objective.lower = cost

                if not objective.lock_upper and objective.upper is not None:
                    if cost > objective.upper:
                        objective.upper = cost

            updated_objectives += [objective.to_json()]

        self.meta["objectives"] = updated_objectives

        if isinstance(config, Configuration):
            config = config.get_dictionary()

        if config not in self.configs.values():
            config_id_len = len(self.configs)
            self.configs[config_id_len] = config
            self.origins[config_id_len] = origin

        config_id = self.get_config_id(config)
        if config_id is None:
            raise ValueError("Config id is None.")

        trial = Trial(
            config_id=config_id,
            budget=budget,
            seed=seed,
            costs=costs,
            start_time=np.round(start_time, 2),
            end_time=np.round(end_time, 2),
            status=status,
            additional=additional,
        )

        trial_key = trial.get_key()
        if trial_key not in self.trial_keys:
            self.trial_keys[trial_key] = len(self.history)
            self.history += [trial]
        else:
            # Overwrite
            self.history[self.trial_keys[trial_key]] = trial

        # Update budgets
        if budget not in self.meta["budgets"]:
            self.meta["budgets"].append(budget)
            self.meta["budgets"].sort()

        self._update_highest_budget(config_id, budget, status)

        # Update seeds
        if seed not in self.meta["seeds"]:
            self.meta["seeds"].append(seed)
            self.meta["seeds"].sort()

        # Update models
        # Problem: The model should not be in the cache.
        # Therefore, first the model is kept as it is,
        # but remove it from the dict and save it to the disk later on.
        if model is not None:
            self.models[config_id] = model

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the run and its information.

        Parameters
        ----------
        path : Optional[Union[str, Path]]
            The path in which to save the trials.

        Raises
        ------
        RuntimeError
            If the path is not specified.
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
        with jsonlines.open(self.history_fn, mode="w") as f:
            for trial in self.history:
                f.write(trial.to_json())

        # TODO: Update general cache file and tell him that self.path was used
        # to save the run.
        # Then, DeepCAVE can show direct suggestions in the select path dialog.

        # Models
        if len(self.models) > 0:
            # torch is imported here, because it is not wanted as requirement.
            import torch

            # Iterate over models and save them if they are a module.
            for config_id in list(self.models.keys()):
                filename = self.models_dir / f"{str(config_id)}.pth"
                if not filename.exists():
                    make_dirs(filename)

                    model = self.models[config_id]
                    if isinstance(model, torch.nn.Module):
                        torch.save(model, filename)
                    else:
                        raise RuntimeError("Unknown model type.")

                # Remove from dict
                del self.models[config_id]

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load the run.

        Parameters
        ----------
        path : Optional[Union[str, Path]], optional
            The path where to load the run from.
            Default is None.

        Raises
        ------
        RuntimeError
            If the path is None.
            If the trials were not found.
        """
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
        self.configspace = cs_json.read(self.configspace_fn.read_text())

        # Load configs
        configs = json.loads(self.configs_fn.read_text())
        # Make sure all keys are integers
        self.configs = {int(k): v for k, v in configs.items()}

        # Load origins
        origins = json.loads(self.origins_fn.read_text())
        self.origins = {int(k): v for k, v in origins.items()}

        # Load history
        with jsonlines.open(self.history_fn) as f:
            self.history = []
            for obj in f:
                # Create trial object here
                trial = Trial(*obj)
                self.history.append(trial)

                # Also create trial_keys
                self.trial_keys[trial.get_key()] = len(self.history) - 1

                # Update highest budget
                self._update_highest_budget(trial.config_id, trial.budget, trial.status)
