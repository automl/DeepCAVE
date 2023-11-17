#  noqa: D400
"""
# DeepCAVE

This module defines the DeepCAVE run object.
It provides utilities to hash and get the DeepCAVE run object, as well as the latest change.

## Classes
    - DeepCAVERun: Define the DeepCAVE run and provide handling utilities.
"""

from typing import Union

from pathlib import Path

from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVERun(Run):
    """
    Define the DeepCAVE run and provide handling utilities.

    Properties
    ----------
    path : Path
        The path to the "history.jsonl" file.
    """

    prefix = "DeepCAVE"
    _initial_order = 1

    @property
    def hash(self) -> str:
        """Calculate a hash value of a jsonl history file to use as id."""
        if self.path is None:
            return ""

        # Use hash of history.json as id
        return file_to_hash(self.path / "history.jsonl")

    @property
    def latest_change(self) -> Union[float, int]:
        """Get the timestamp of the latest change made to the history file."""
        if self.path is None:
            return 0

        return Path(self.path / "history.jsonl").stat().st_mtime

    @classmethod
    def from_path(cls, path: Path) -> "DeepCAVERun":
        """Get a DeepCAVE run from a given path."""
        return DeepCAVERun(path.stem, path=Path(path))
