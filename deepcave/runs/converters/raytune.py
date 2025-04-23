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
        return ""
        # TODO: What to use as id

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
        return 0.0
        # TODO: Which file for latest change

    @classmethod
    def from_path(cls, path: Path) -> "RayTuneRun":
        """Return a Run object from a given path."""
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
        # TODO: What files to we need to process
        if os.path.isfile(path_name + "/runhistory.json") and os.path.isfile(
            path_name + "/configspace.json"
        ):
            return True
        return False
