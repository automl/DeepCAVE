# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# OptunaRun

This module provides utilities to create an Optuna run.

## Classes
    - OptunaRun: Define an Optuna run object.
"""

from typing import List, Union

import pickle
from pathlib import Path

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Uniform,
)
from ConfigSpace.hyperparameters import Hyperparameter

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class OptunaRun(Run):
    """
    Define an Optuna run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "Optuna"
    _initial_order = 2

    @staticmethod
    def _get_pickle_file(path: Path) -> Path:
        """
        Get the path to the pickle file from the directory path.

        Parameters
        ----------
        path : Path
            The path to the directory containing the pickle file.

        Returns
        -------
        Path
            The path to the pickle file.
        """
        pickle_files = list(path.glob("*.pkl"))
        if len(pickle_files) != 1:
            raise RuntimeError(f"There should be exactly one pickle file in '{path}'")
        else:
            return pickle_files[0]

    @property
    def hash(self) -> str:
        """
        Hash of the current run.

        If the hash changes, the cache has to be cleared.
        This ensures that the cache always holds the latest results of the run.

        Returns
        -------
        str
            The hash of the run.
        """
        if self.path is None:
            return ""

        pickle_file = OptunaRun._get_pickle_file(self.path)
        return file_to_hash(pickle_file)

    @property
    def latest_change(self) -> Union[float, int]:
        """
        Get the timestamp of the latest change.

        Returns
        -------
        Union[float, int]
            The latest change.
        """
        if self.path is None:
            return 0

        pickle_file = OptunaRun._get_pickle_file(self.path)
        return Path(pickle_file).stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "OptunaRun":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The Optuna run.

        Raises
        ------
        RuntimeError
            Instances are not supported.
        """
        path = Path(path)

        try:
            from optuna.distributions import (
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
            from optuna.search_space import IntersectionSearchSpace
            from optuna.study import StudyDirection
            from optuna.trial import TrialState
        except ImportError:
            raise ImportError(
                "The Optuna package is required to load Optuna runs. "
                "Please install it via `pip install deepcave[optuna]`"
            )

        # Load the optuna study from the file path
        pickle_file_path = OptunaRun._get_pickle_file(path)
        with open(pickle_file_path, "rb") as f:
            optuna_study = pickle.load(f)

        # Read configspace
        optuna_space = IntersectionSearchSpace(include_pruned=True).calculate(study=optuna_study)
        configspace = ConfigurationSpace()

        hyperparameters: List[Hyperparameter] = []

        for hp_name, hp in optuna_space.items():
            if isinstance(hp, FloatDistribution) or isinstance(hp, IntDistribution):
                if hp.step is not None and hp.step != 1:
                    logger.warning(
                        f"Step is not supported. "
                        f'Step={hp.step} will be ignored for hyperparameter "{hp_name}".'
                    )

            if isinstance(hp, FloatDistribution):
                hyperparameters.append(
                    Float(
                        name=hp_name,
                        bounds=(hp.low, hp.high),
                        distribution=Uniform(),
                        default=None,
                        log=hp.log,
                    )
                )
            elif isinstance(hp, IntDistribution):
                hyperparameters.append(
                    Integer(
                        name=hp_name,
                        bounds=(hp.low, hp.high),
                        distribution=Uniform(),
                        default=None,
                        log=hp.log,
                    )
                )
            elif isinstance(hp, CategoricalDistribution):
                hyperparameters.append(
                    Categorical(
                        name=hp_name,
                        default=hp.choices[0],
                        items=hp.choices,
                    )
                )
            else:
                raise ValueError(
                    (
                        "The hyperparameters in the Optuna study must be of type "
                        "`FloatDistribution`, `IntDistribution` or `CategoricalDistribution`, "
                        f"but a hyperparameter of type {type(hp)} was given."
                    )
                )
        configspace.add(hyperparameters)

        n_objectives = max(len(trial.values) for trial in optuna_study.trials)
        obj_list = list()
        for i in range(n_objectives):
            if optuna_study.metric_names is not None:
                metric_name = optuna_study.metric_names[i]
            else:
                metric_name = f"Objective{i}"
            optimize = "lower" if optuna_study.directions[i] == StudyDirection.MINIMIZE else "upper"

            obj_list.append(
                Objective(
                    name=metric_name,
                    lower=None,
                    upper=None,
                    optimize=optimize,
                )
            )

        obj_list.append(Objective("Time"))

        # Let's create a new run object
        run = OptunaRun(name=path.stem, configspace=configspace, objectives=obj_list, meta=None)

        # The path has to be set manually
        run._path = path

        first_starttime = None
        for trial in optuna_study.trials:
            try:
                config = Configuration(configspace, trial.params)
            except ValueError:
                raise ValueError(
                    f"Could not convert the configuration of trial {trial.number} to "
                    f"a ConfigSpace configuration.\nThis might be due to the "
                    f"configuration space containing conditionals or dynamic "
                    f"hyperparameter value ranges, which are currently not supported."
                )

            if first_starttime is None:
                first_starttime = trial.datetime_start.timestamp()

            starttime = trial.datetime_start.timestamp() - first_starttime
            endtime = trial.datetime_complete.timestamp() - first_starttime

            if trial.state == TrialState.COMPLETE:
                status = Status.SUCCESS
            elif trial.state == TrialState.FAIL:
                status = Status.FAILED
            elif trial.state == TrialState.PRUNED:
                status = Status.PRUNED
            elif trial.state == TrialState.RUNNING or trial["status"] == TrialState.WAITING:
                continue
            else:
                status = Status.UNKNOWN

            cost = trial.values

            if status != Status.SUCCESS:
                # Costs which failed, should not be included
                cost = [None] * len(cost) if isinstance(cost, list) else None
                time = None
            else:
                time = float(endtime - starttime)

            run.add(
                costs=cost + [time] if isinstance(cost, list) else [cost, time],  # type: ignore
                config=config,
                budget=0.0,
                seed=-1,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=None,
                additional=None,
            )

        return run
