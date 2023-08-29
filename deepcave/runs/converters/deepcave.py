#  noqa: D400
"""
# DeepCAVE

This module defines the DeepCAVE run object.
It provides utilities to hash and get the DeepCAVE run object.

## Classes
    - DeepCAVERun: Define the DeepCAVE run and provide handling utilties.

## Contents
    - hash: Hash the file.
    - latest_change: Get the latest change of the file.
    - from_path: Get the run from the given path.
"""

from pathlib import Path

from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVERun(Run):
    """
    Define the DeepCAVE run and provide handling utilties.

    Methods
    -------
    hash
        Hash the file.
    latest_change
        Get the latest change of the file.
    from_path
        Get the run from the given path.

    Attributes
    ----------
    prefix
        The prefix of the run.
    initial_order
        The initial order.
    """

    prefix = "DeepCAVE"
    _initial_order = 1

    @property
    def hash(self):  # noqa: D102
        if self.path is None:
            return ""

        # Use hash of history.json as id
        return file_to_hash(self.path / "history.jsonl")

    @property
    def latest_change(self):  # noqa: D102
        if self.path is None:
            return 0

        return Path(self.path / "history.jsonl").stat().st_mtime

    @classmethod
    def from_path(cls, path):  # noqa: D102
        return DeepCAVERun(path.stem, path=Path(path))
