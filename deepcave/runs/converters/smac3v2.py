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
# SMAC3v2Run

This module provides utilities to create a SMAC3v2
(Sequential Model-based Algorithm Configuration) run.

Version 2.0.0 is used.

## Classes
    - SMAC3v2Run: Define a SMAC3v2 run object.
"""

from typing import Dict, List, Optional, Union

import json
import os
from pathlib import Path

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class SMAC3v2Run(Run):
    """
    Define a SMAC3v2 (Sequential Model-based Algorithm Configuration) run object.

    Version 2.0.0 is used.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "SMAC3v2"
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

        # Use hash of history.json as id
        return file_to_hash(self.path / "runhistory.json")

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

        return Path(self.path / "runhistory.json").stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "SMAC3v2Run":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The SMAC3v2 run.

        Raises
        ------
        RuntimeError
            Instances are not supported.
        """
        path = Path(path)

        # Read configspace
        configspace = ConfigurationSpace.from_json(path / "configspace.json")

        # Read objectives
        with (path / "scenario.json").open() as json_file:
            all_data = json.load(json_file)
            objectives = all_data["objectives"]

        obj_list = list()
        if not isinstance(objectives, list):
            objectives = [objectives]
        for obj in objectives:
            obj_list.append(Objective(obj))
        # Only lock lower for time
        obj_list.append(Objective("Time"))

        # Read meta
        with (path / "scenario.json").open() as json_file:
            meta = json.load(json_file)
            meta["run_objectives"] = meta.pop("objectives")
            meta["optimizer_seed"] = meta.pop("seed")

        # Let's create a new run object
        run = SMAC3v2Run(name=path.stem, configspace=configspace, objectives=obj_list, meta=meta)

        # The path has to be set manually
        run._path = path

        # Iterate over the runhistory
        with (path / "runhistory.json").open() as json_file:
            all_data = json.load(json_file)
            data = all_data["data"]
            config_origins = all_data["config_origins"]
            configs = all_data["configs"]

        instance_ids: List[int] = []

        first_starttime = None

        if isinstance(data, list):
            import warnings

            warnings.warn(
                "The runhistory.json file is in an outdated format.",
                DeprecationWarning,
                stacklevel=2,  # Adjusts the stack level to point to the caller.
            )
            for (
                config_id,
                instance_id,
                seed,
                budget,
                cost,
                time,
                status,
                starttime,
                endtime,
                additional_info,
            ) in data:
                run_dict = run._process_data_entry(
                    str(config_id),
                    instance_id,
                    seed,
                    budget,
                    cost,
                    time,
                    status,
                    starttime,
                    endtime,
                    additional_info,
                    first_starttime,
                    instance_ids,
                    configs,
                    config_origins,
                )
                if run_dict is not None:
                    run.add(**run_dict)
        elif isinstance(data, dict):
            for config_id, config_data in data.items():
                instance_id = config_data["instance"]
                seed = config_data["seed"]
                budget = config_data["budget"]
                cost = config_data["cost"]
                time = config_data["time"]
                status = config_data["status"]
                starttime = config_data["starttime"]
                endtime = config_data["endtime"]
                additional_info = config_data["additional_info"]
                run_dict = run._process_data_entry(
                    config_id,
                    instance_id,
                    seed,
                    budget,
                    cost,
                    time,
                    status,
                    starttime,
                    endtime,
                    additional_info,
                    first_starttime,
                    instance_ids,
                    configs,
                    config_origins,
                )
                if run_dict is not None:
                    run.add(**run_dict)
        else:
            raise RuntimeError("Data in runhistory.json is not in a valid format.")
        return run

    def _process_data_entry(
        self,
        config_id: str,
        instance_id: int,
        seed: int,
        budget: Optional[float],
        cost: Optional[Union[List[Union[float, None]], float]],
        time: Optional[float],
        status: int,
        starttime: float,
        endtime: float,
        additional_info: Optional[Dict],
        first_starttime: Optional[float],
        instance_ids: List[int],
        configs: Dict,
        config_origins: Dict[str, str],
    ) -> Optional[Dict]:
        if instance_id not in instance_ids:
            instance_ids += [instance_id]

        if len(instance_ids) > 1:
            raise RuntimeError("Instances are not supported.")

        config = configs[config_id]

        if first_starttime is None:
            first_starttime = starttime

        starttime = starttime - first_starttime
        endtime = endtime - first_starttime

        if status == 0:
            # still running
            return None
        elif status == 1:
            status = Status.SUCCESS
        elif status == 3:
            status = Status.TIMEOUT
        elif status == 4:
            status = Status.MEMORYOUT
        else:
            status = Status.CRASHED

        if status != Status.SUCCESS:
            # Costs which failed, should not be included
            cost = [None] * len(cost) if isinstance(cost, list) else None
            time = None
        else:
            time = endtime - starttime

        # Round budget
        if budget:
            budget = np.round(budget, 2)
        else:
            budget = 0.0

        origin = None
        if config_id in config_origins:
            origin = config_origins[config_id]

        return {
            "costs": cost + [time] if isinstance(cost, list) else [cost, time],
            "config": config,
            "budget": budget,
            "seed": seed,
            "start_time": starttime,
            "end_time": endtime,
            "status": status,
            "origin": origin,
            "additional": additional_info,
        }

    @classmethod
    def is_valid_run(cls, path_name: str) -> bool:
        """
        Check whether the path name belongs to a valid smac3v2 run.

        Parameters
        ----------
        path_name: str
            The path to check.

        Returns
        -------
        bool
            True if path is valid run.
            False otherwise.
        """
        if os.path.isfile(path_name + "/runhistory.json") and os.path.isfile(
            path_name + "/configspace.json"
        ):
            return True
        return False
