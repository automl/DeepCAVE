#  noqa: D400
"""
# Recorder

This module provides utilities to record information.

## Classes
    - Recorder: Define a Recorder for recording information.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import time
from pathlib import Path

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration
from typing_extensions import Self

from deepcave.runs import Status
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective


class Recorder:
    """
    Define a Recorder for recording information.

    Properties
    ----------
    path : Path
        The path to the recorded information.
    last_trial_id : tuple[Any, Optional[float], Optional[int]]
        The last id containing the configuration, budget, and seed.
    start_time : float
        The current time in seconds since the epoch.
    start_times : Dict[Any, Any]
        A dictionary containing the start times with their id as key.
    models : Dict[Any, Any|
        The models used with their id as key.
    origins : Dict[Any, Any]
        The origins with their id as key.
    additionals : Dict[Any, Any]
        Additional information with the id as key.
    run : DeepCAVERun
        The deepcave run container.
    """

    def __init__(
        self,
        configspace: ConfigSpace.ConfigurationSpace,
        objectives: Optional[List[Objective]] = None,
        meta: Optional[Dict[str, Any]] = None,
        save_path: str = "logs",
        prefix: str = "run",
        overwrite: bool = False,
    ):
        """
        All objectives follow the scheme the lower the better.

        Parameters
        ----------
        save_path : str, otpional
            The path to the recording.
            Default is "logs".
        configspace : ConfigSpace.ConfigurationSpace
            The configuration space.
        objectives Optional[List[Objective]], optional
            The objectives of the run.
            Default is None.
        prefix ; str, optional
            Name of the trial. If not given, trial_x will be used.
            Default is "run".
        overwrite : bool, optional
            Uses the prefix as name and overwrites the file.
            Default is False.
        """
        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        self.path: Path
        self._set_path(save_path, prefix, overwrite)

        # Set variables
        self.last_trial_id: Optional[
            Tuple[Union[Dict[Any, Any], Configuration], Optional[float], int]
        ] = None
        self.start_time = time.time()
        self.start_times: Dict[
            Tuple[Union[Dict[Any, Any], Configuration], Optional[float], Optional[int]], float
        ] = {}
        self.models: Dict[
            Tuple[Union[Dict[Any, Any], Configuration], Optional[float], Optional[int]],
            Optional[Any],
        ] = {}
        self.origins: Dict[
            Tuple[Union[Dict[Any, Any], Configuration], Optional[float], Optional[int]],
            Optional[str],
        ] = {}
        self.additionals: Dict[
            Tuple[Union[Dict[Any, Any], Configuration], Optional[float], Optional[int]],
            Dict[Any, Any],
        ] = {}

        # Define trials container
        self.run = DeepCAVERun(
            self.path.stem, configspace=configspace, objectives=objectives, meta=meta
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback) -> None:  # type: ignore
        pass

    def _set_path(
        self, path: Union[str, Path], prefix: str = "run", overwrite: bool = False
    ) -> None:
        """
        Set the path.

        Parameters
        ----------
        path : Union[str, Path]
            The path to set.
        prefix, optional
            The prefix for the path.
            Default is "run".
        overwrite, optional
            To determine whether to overwrite an existing folder.
            Default is False.
        """
        # Make sure the word is interpreted as folder
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            new_idx = 0
            for file in path.iterdir():
                if not file.name.startswith(f"{prefix}_"):
                    continue
                idx = file.name.split("_")[-1]
                if idx.isnumeric():
                    idx_int = int(idx)
                    if idx_int > new_idx:
                        new_idx = idx_int

            # And increase the id
            new_idx += 1
            self.path = path / f"{prefix}_{new_idx}"
        else:
            self.path = path / f"{prefix}"

    def start(
        self,
        config: Configuration,
        budget: Optional[float] = None,
        seed: int = -1,
        model: Optional[Any] = None,
        origin: Optional[str] = None,
        additional: Optional[dict] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """
        Start recording the information.

        Parameters
        ----------
        config : Configuration
            Holds the configuration settings.
        budget : Optional[float], optional
            The budget.
            Default is None.
        seed : int
            The seed.
            Default is -1.
        model : Optional[Any], optional
            The model used.
            Default is None.
        origin : Optional[str], optional
            The origin.
            Default is None.
        additional : Optional[dict], optional
            Additional information.
            Default is None.
        start_time : Optional[float], optional
            The start time.
            Default is None.
        """
        if additional is None:
            additional = {}

        id: Tuple[Union[Dict[Any, Any], Configuration], Optional[float], int] = (
            config,
            budget,
            seed,
        )

        if start_time is None:
            start_time = time.time() - self.start_time

        # Start timer
        self.start_times[id] = start_time
        self.models[id] = model
        self.origins[id] = origin
        self.additionals[id] = additional

        self.last_trial_id = id

    def end(
        self,
        costs: float = np.inf,
        status: Status = Status.SUCCESS,
        config: Optional[Union[dict, Configuration]] = None,
        budget: Optional[float] = np.inf,
        seed: int = -1,
        additional: Optional[dict] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """
        End the recording and add it to the trial history.

        In case of multi-fidelity, config+budget should be passed.
        In case of non-deterministic runs, seed should be passed.
        If it can't be passed, it can't be matched correctly.

        Parameters
        ----------
        costs : float, optional
            The costs.
            Default is np.inf.
        status : Status, optional
            The status.
            Default is Status.Success.
        config : Union[dict, Configuration], optional
            The configuration.
            Default is None.
        budget : float, optional
            The budget.
            Default is np.inf.
        seed : int
            The seed.
            Default is -1.
        additional : Optional[dict], optional
            Additional information.
            Default is None.
        end_time : Optional[float], optional
            The end time.
            Default is None.

        Raises
        ------
        AssertionError
            If no trial was started yet.
        """
        if additional is None:
            additional = {}

        if config is not None:
            id = (config, budget, seed)
        else:
            assert self.last_trial_id is not None, "No trial started yet."
            id = self.last_trial_id
            config, budget, seed = id[0], id[1], id[2]

        model = self.models[id]
        start_additional = self.additionals[id].copy()
        start_additional.update(additional)
        start_time = self.start_times[id]

        if end_time is None:
            end_time = time.time() - self.start_time

        assert budget is not None
        assert seed is not None

        # Add to trial history
        self.run.add(
            costs=costs,
            config=config,
            budget=budget,
            seed=seed,
            start_time=start_time,
            end_time=end_time,
            status=status,
            model=model,
            additional=start_additional,
        )

        # Clean the dicts
        self.start_times.pop(id)
        self.models.pop(id)
        self.origins.pop(id)
        self.additionals.pop(id)

        # And save the results
        self.run.save(self.path)
