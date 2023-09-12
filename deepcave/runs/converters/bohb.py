"""
# BOHBRun.

This module is for managing and processing data concerning a BOHB Run.

Utilities provide getting a hash, as well as the latest change of a file.

## Classes
    - BOHBRun: Create a BOHB Run.
"""

from pathlib import Path

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class BOHBRun(Run):
    """
    Create a BOHB Run.

    This class extends the Run class.
    Utilities provide getting a hash, as well as the latest change of a file.

    Attributes
    ----------
    prefix
        Set the prefix to "BOHB".

    Properties
    ----------
    path : Path
        The path to the "results.json" file.
    """

    prefix = "BOHB"
    _initial_order = 2

    @property
    def hash(self):
        """Calculate a hash value of the json result file."""
        if self.path is None:
            return ""

        # Use hash of results.json as id
        return file_to_hash(self.path / "results.json")

    @property
    def latest_change(self):
        """Get the timestamp of the latest change made to the results file."""
        if self.path is None:
            return 0

        return Path(self.path / "results.json").stat().st_mtime

    @classmethod
    def from_path(cls, path):
        """
        Create a BOHB Run from a given path and add a new trial to it.

        Parameters
        ----------
        path
            The pathname to create the Run from.

        Returns
        -------
        The BOHB run
        """
        path = Path(path)

        # Read configspace
        from ConfigSpace.read_and_write import json as cs_json

        configspace = cs_json.read((path / "configspace.json").read_text())

        # Read objectives
        # We have to define it ourselves, because we don't know the type of the objective
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
                status = "CRASHED"
            else:
                status = bohb_run.info["state"]

            config = config_mapping[bohb_run.config_id]["config"]

            origin = None
            additional = {}

            # QUEUED, RUNNING, CRASHED, REVIEW, TERMINATED, COMPLETED, SUCCESS
            if "SUCCESS" in status or "TERMINATED" in status or "COMPLETED" in status:
                status = Status.SUCCESS
            elif "RUNNING" in status or "QUEUED" in status or "REVIEW" in status:
                status = Status.RUNNING
            else:
                status = Status.CRASHED

            if status != Status.SUCCESS:
                # We don't want cost included which are failed
                cost = None

            run.add(
                costs=[cost],  # Having only single objective here
                config=config,
                budget=budget,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=origin,
                additional=additional,
            )

        return run
