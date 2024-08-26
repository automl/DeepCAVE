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
# AMLTKRun

This module provides utilities to create an AMLTK (AutoML Toolkit) run.

## Classes
    - AMLTKRun: Define an AMLTK run object.
"""

from typing import Optional, Sequence, Union

import re
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.converters import extract_config, extract_costs, extract_value
from deepcave.utils.hash import file_to_hash


class AMLTKRun(Run):
    """
    Define an AMLTK (AutoML Toolkit) run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "AMLTK"
    _initial_order = 2

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

        # Use hash of history.parquet as id
        return file_to_hash(self.path / "history.parquet")

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

        return Path(self.path / "history.parquet").stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "AMLTKRun":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The AMLTK run.

        Raises
        ------
        RuntimeError
            Instances are not supported.
        """
        path = Path(path)

        # Read configspace
        configspace = ConfigurationSpace.from_json(path / "configspace.json")

        history = pd.read_parquet(path / "history.parquet")

        history["budget"] = history.index.map(
            lambda x: float(value) if (value := extract_value(x, "budget")) is not None else None
        )

        # Extract the objectives from the dataframe
        obj_list = list()
        for metric_string in history.columns:
            if metric_string.startswith("metric:"):
                match = re.match(
                    r"metric:(\w+) \[(-?\d+\.?\d*|[-+]inf), (-?\d+\.?\d*|[-+]inf)\] \((\w+)\)",
                    metric_string,
                )
                assert match is not None
                metric_name = match.group(1)
                lower = float(match.group(2))
                upper = float(match.group(3))
                maximize = match.group(4) == "maximize"

                obj_list.append(
                    Objective(
                        name=metric_name,
                        lower=lower,
                        upper=upper,
                        optimize="upper" if maximize else "lower",
                    )
                )

        obj_list.append(Objective("Time"))

        # Let's create a new run object
        run = AMLTKRun(name=path.stem, configspace=configspace, objectives=obj_list, meta=None)

        # The path has to be set manually
        run._path = path

        first_starttime = None
        seeds = []
        for _, trial in history.iterrows():
            config = extract_config(trial, configspace)

            if trial["trial_seed"] not in seeds:
                seeds.append(trial["trial_seed"])

            # Start and end time of the trial need to be given via a deepcave:time:start and
            # deepcave:time:end column
            starttime_col = "deepcave:time:start"
            endtime_col = "deepcave:time:end"
            if starttime_col not in history.columns:
                raise ValueError(
                    f"Missing DeepCAVE start time column '{starttime_col}' in history.csv."
                )
            if endtime_col not in history.columns:
                raise ValueError(
                    f"Missing DeepCAVE end time column '{endtime_col}' in history.csv."
                )

            if first_starttime is None:
                first_starttime = trial[starttime_col]

            starttime = trial[starttime_col] - first_starttime
            endtime = trial[endtime_col] - first_starttime

            if trial["status"] == "success":
                status = Status.SUCCESS
            elif trial["status"] == "fail":
                status = Status.FAILED
            elif trial["status"] == "crashed":
                status = Status.CRASHED
            else:
                status = Status.UNKNOWN

            cost: Sequence[Optional[float]] = extract_costs(trial)

            if status != Status.SUCCESS:
                # Costs which failed, should not be included
                cost = [None] * len(cost)
                time = None
            else:
                time = float(endtime - starttime)

            # Round budget
            if trial["budget"] is not None:
                budget = np.round(trial["budget"], 2)
            else:
                budget = 0.0

            if trial["traceback"] is not None:
                additional_info = {"traceback": trial["traceback"]}
            else:
                additional_info = None

            run.add(
                costs=cost + [time],  # type: ignore
                config=config,
                budget=budget,
                seed=trial["trial_seed"],
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=None,
                additional=additional_info,
            )

        return run
