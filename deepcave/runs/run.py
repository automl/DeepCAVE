from abc import ABC, abstractmethod
from typing import Any, Optional, Union

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

    def __init__(
        self,
        name: str,
        configspace=None,
        objectives: Union[Objective, list[Objective]] = None,
        meta: dict[str, Any] = None,
        path: Optional[Union[str, Path]] = None,
    ):
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
        if not isinstance(objectives, list):
            objectives = [objectives]

        for objective in objectives:
            assert isinstance(objective, Objective)

        # Meta
        self.meta = {"objectives": objectives, "budgets": []}
        self.meta.update(meta)

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> "Run":
        """
        Based on a path, return a new Run object.
        """
        pass

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

    def exists(self) -> bool:
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
        costs: Union[list[float], float],
        config: Union[dict, Configuration],  # either dict or Configuration
        budget: float = np.inf,
        start_time: float = 0.0,
        end_time: float = 0.0,
        status: Status = Status.SUCCESS,
        origin=None,
        model=None,
        additional: Optional[dict] = None,
    ):
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
            additional=additional,
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
        with jsonlines.open(self.history_fn, mode="w") as f:
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

        # Load models
        # TODO
