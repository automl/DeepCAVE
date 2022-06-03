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
    Creates a new run.
    If path is given, runs are loaded from the path.

    Parameters
    ----------
    name : str
        Name of the run.
    configspace : ConfigSpace, optional
        Configuration space of the run. Should be None if `path` is used. By default None.
    objectives : Union[Objective, List[Objective]], optional
        Objectives of the run. Should be None if `path` is used. By default None
    meta : Dict[str, Any], optional
        Meta data of the run. Should be None if `path` is used. By default None.
    path : Optional[Union[str, Path]], optional
        If a path is specified, the run is loaded from there. By default None.

    Raises
    ------
    RuntimeError
        If no configuration space is provided or found.
    """

    prefix = "run"
    _initial_order: int

    def __init__(
        self,
        name: str,
        configspace: ConfigSpace = None,
        objectives: Union[Objective, List[Objective]] = None,
        meta: Dict[str, Any] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        super(Run, self).__init__(name)

        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        # Reset and load configspace/path
        self.reset()
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
        self.meta = {"objectives": serialized_objectives, "budgets": []}
        self.meta.update(meta)

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> "Run":
        """
        Based on a path, return a new Run object.
        """
        pass

    @property
    def id(self) -> str:
        return string_to_hash(f"{self.prefix}:{self.path}")

    @property
    def path(self) -> Optional[Path]:
        return self._path

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
        self.models_dir = self._path / "models"

    def exists(self) -> bool:
        """
        Checks if the run exists based on the internal path.

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
        budget: float = np.inf,
        start_time: float = 0.0,
        end_time: float = 0.0,
        status: Status = Status.SUCCESS,
        origin: str = None,
        model: Union[str, "torch.nn.Module"] = None,  # type: ignore
        additional: Optional[Dict] = None,
    ) -> None:
        """
        Adds a trial to the run.
        If combination of config and budget already exists, it will be overwritten.
        Not successful runs are added with `None` costs.

        Parameters
        ----------
        costs : Union[List[float], float]
            Costs of the run. In case of multi-objective, a list of costs is expected.
        config : Union[Dict, Configuration]
            The corresponding configuration.
        start_time : float, optional
            Start time. By default 0.0
        end_time : float, optional
            End time. By default 0.0
        status : Status, optional
            Status of the trial. By default Status.SUCCESS
        origin : str, optional
            Origin of the trial. By default None
        model : Union[str, &quot;torch.nn.Module&quot;], optional
            Model of the trial. By default None
        additional : Optional[Dict], optional
            Additional information of the trial. By default None.
            Following information is used by DeepCAVE:
            * traceback

        Raises
        ------
        RuntimeError
            If number of costs does not match number of objectives.
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
                if not objective.lock_lower:
                    if cost < objective.lower:  # type: ignore
                        objective.lower = cost

                if not objective.lock_upper:
                    if cost > objective.upper:  # type: ignore
                        objective.upper = cost

            updated_objectives += [objective.to_json()]

        self.meta["objectives"] = updated_objectives

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

        # Update models
        # Problem: We don't want to have the model in the cache.
        # Therefore, we first keep the model as it is,
        # but remove it from the dict and save it to the disk later on.
        if model is not None:
            self.models[config_id] = model

    def save(self, path: Optional[Union[str, Path]] = None):
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
            # We import torch here because we don't want to have it as requirement.
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
