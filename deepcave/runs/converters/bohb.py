# noqa: D400
"""
# BOHBRun

This module provides utilities for managing and processing data concerning a BOHB
(Bayesian Optimization and Hyperparameter Bandits) run.

## Classes
    - BOHBRun: Create a BOHB Run.
"""

from typing import Any, Dict, Union

from pathlib import Path

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class BOHBRun(Run):
    """
    Create a BOHB (Bayesian Optimization and Hyperparameter Bandits) run.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "BOHB"
    _initial_order = 2

    @property
    def hash(self) -> str:
        """
        Get the hash of the current run.

        If the hash changes, the cache has to be cleared.
        This ensures that the cache always holds the latest results of the run.

        Returns
        -------
        str
            The hash of the run.
        """
        if self.path is None:
            return ""

        # Use hash of results.json as id
        return file_to_hash(self.path / "results.json")

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

        return Path(self.path / "results.json").stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "BOHBRun":
        """
        Create a new BOHB run from a given path and add a new trial to it.

        Parameters
        ----------
        path : Union[Path, str]
            The pathname to base the run on.

        Returns
        -------
        The BOHB run
        """
        path = Path(path)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        configspace = cs_json.read((path / "configspace.json").read_text())

        # Read objectives
        # It has to be defined here, because the type of the objective is not known
        # Only lock lower
        objective = Objective("Cost", lower=0)

        run = BOHBRun(path.stem, configspace=configspace, objectives=objective, meta={})
        run._path = path

        from hpbandster.core.result import logged_results_to_HBS_result

        bohb = logged_results_to_HBS_result(str(path))
        config_mapping = bohb.get_id2config_mapping()

        first_starttime = None
        for bohb_run in bohb.get_all_runs():
            times = bohb_run.time_stamps
            starttime = times["started"]
            endtime = times["finished"]

            if first_starttime is None:
                first_starttime = starttime

            starttime = starttime - first_starttime
            endtime = endtime - first_starttime

            cost = bohb_run.loss
            budget = bohb_run.budget

            if bohb_run.info is None:
                status_string = "CRASHED"
            else:
                status_string = bohb_run.info["state"]

            config = config_mapping[bohb_run.config_id]["config"]

            additional: Dict[str, Any] = {}
            status: Status

            # QUEUED, RUNNING, CRASHED, REVIEW, TERMINATED, COMPLETED, SUCCESS
            if (
                "SUCCESS" in status_string
                or "TERMINATED" in status_string
                or "COMPLETED" in status_string
            ):
                status = Status.SUCCESS
            elif (
                "RUNNING" in status_string or "QUEUED" in status_string or "REVIEW" in status_string
            ):
                status = status_string  # type: ignore
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # Costs which failed, should not be included
                cost = None

            run.add(
                costs=[cost],  # Having only single objective here
                config=config,
                budget=budget,
                seed=-1,
                start_time=starttime,
                end_time=endtime,
                status=status,
                additional=additional,
            )

        return run
