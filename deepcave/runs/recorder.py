#  noqa: D400
"""
# Recorder

This module provides utilities to record the trial information.

## Contents
    - set_path: Identify the latest run and sets the path with increased id.
    - start: Record the trial information.
    - end: End the recording of the trial and add it to trial history.
"""

from typing import Optional, Union

import time
from pathlib import Path

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration

from deepcave.runs import Status
from deepcave.runs.converters.deepcave import DeepCAVERun


class Recorder:
    """
    Define a Recorder for recording trial information.

    Methods
    -------
    set_path
        Identify the latest run and sets the path with increased id.
    start
        Record the trial information.
    end
        End the recording of the trial and add it to trial history.
    """

    def __init__(
        self,
        configspace: ConfigSpace.ConfigurationSpace,
        objectives=None,
        meta=None,
        save_path="logs",
        prefix="run",
        overwrite=False,
    ):
        """
        All objectives follow the scheme the lower the better.

        If file

        Parameters
        ----------
            save_path (str):
            configspace (ConfigSpace):
            objectives (list of Objective):
            prefix: Name of the trial. If not given, trial_x will be used.
            overwrite: Uses the prefix as name and overwrites the file.
        """
        if objectives is None:
            objectives = []
        if meta is None:
            meta = {}

        self.path: Path = None
        self._set_path(save_path, prefix, overwrite)

        # Set variables
        self.last_trial_id = None
        self.start_time = time.time()
        self.start_times = {}
        self.models = {}
        self.origins = {}
        self.additionals = {}

        # Define trials container
        self.run = DeepCAVERun(
            self.path.stem, configspace=configspace, objectives=objectives, meta=meta
        )

    def __enter__(self):  # noqa: D102, D105
        return self

    def __exit__(self, type, value, traceback):  # noqa: D102, D105
        pass

    def _set_path(self, path: Union[str, Path], prefix="run", overwrite=False):
        """
        Identify the latest run and sets the path with increased id.

        Parameters
        ----------
        path : Union[str, Path]
            The path in which to store the run.
        prefix, optional
            The prefix for the path.
            Default is "run".
        overwrite, optional
            To determine wheter to overwrite an existing folder.
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
                    idx = int(idx)
                    if idx > new_idx:
                        new_idx = idx

            # And increase the id
            new_idx += 1
            self.path = path / f"{prefix}_{new_idx}"
        else:
            self.path = path / f"{prefix}"

    def start(
        self,
        config: Union[dict, Configuration],
        budget: Optional[float] = None,
        model=None,
        origin=None,
        additional: Optional[dict] = None,
        start_time: Optional[float] = None,
    ):
        """
        Record the trial information.

        Parameters
        ----------
        config : Union[dict, Configuration]
            Holds the configuration settings for the trial.
        budget : Optional[float], optional
            The budget for the trial.
            Default is None.
        model, optional
            The model used in the trial.
            Default is None.
        origin, optional
            The origin of the trial.
            Default is None.
        additional : Optional[dict], optional
            Additional information of the trial.
            Default is None.
        start_time : Optional[float], optional
            The start time of the trial.
            Default is None.
        """
        if additional is None:
            additional = {}

        id = (config, budget)

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
        config: Union[dict, Configuration] = None,
        budget: float = np.inf,
        additional: Optional[dict] = None,
        end_time: Optional[float] = None,
    ):
        """
        End the recording of the trial and add it to trial history.

        In case of multi-processing, config+budget should be passed.
        If it can't be passed, it can't be matched correctly.
        The results of the trial are saved.

        Parameters
        ----------
        costs : float, optional
            The costs of the trial.
            Default is np.inf.
        status : Status, optional
            The status of the trial.
            Default is Status.Success.
        config : Union[dict, Configuration], optional
            The configuration of the trial.
            Default is None.
        budget : float, optional
            The budget of the trial.
            Default is np.inf.
        additional : Optional[dict], optional
            Additional information of the trial.
            Default is None.
        end_time : Optional[float], optional
            The end time of the trial.
            Default is None.
        """
        if additional is None:
            additional = {}

        if config is not None:
            id = (config, budget)
        else:
            id = self.last_trial_id
            config, budget = id[0], id[1]

        model = self.models[id]
        start_additional = self.additionals[id].copy()
        start_additional.update(additional)
        start_time = self.start_times[id]

        if end_time is None:
            end_time = time.time() - self.start_time

        # Add to trial history
        self.run.add(
            costs=costs,
            config=config,
            budget=budget,
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
