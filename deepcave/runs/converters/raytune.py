#  noqa: D400
"""
# RayTuneRun

This module provides utilities to create a RayTune run.

## Classes
    - RayTuneRun: Define an RayTune run object.
"""

import os
from pathlib import Path

from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class RayTuneRun(Run):
    """
    Define a RayTune run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "RayTune"

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

        # Use hash of results.json as id
        return file_to_hash(self.path / "results.json")

    @property
    def latest_change(self) -> float:
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
    def from_path(cls, path: Path) -> "RayTuneRun":
        """Return a Run object from a given path."""
        # Warning also in configspace.json
        return RayTuneRun("def")

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
        if os.path.isfile(path_name + "/result.json") and os.path.isfile(
            path_name + "/params.json"
        ):
            if not os.path.isfile(path_name + "/configspace.json"):
                print(
                    "The configspace.json file will be auto extracted. For more"
                    "reliable results please provide your own configspace.json file or "
                    "ajust the one provided."
                )
                # TODO: create issue for the notificationp problem
                return True
            return True
        return False
