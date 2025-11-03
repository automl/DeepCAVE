"""
# HyperSHAP

This module defines the HyperSHAP run object.

## Classes
    - HyperSHAPRun: Create a HyperSHAP run and provide handling utilities.
"""

from deepcave.runs.run import Run

class HyperSHAP(Run):
    @property
    def hash(self) -> str:
        """
        Return a unique hash for the run (e.g., hashing the trial history).
        """
        pass

    @property
    def latest_change(self) -> float:
        """
        Return the timestamp of the latest change.
        """
        pass

    @classmethod
    def from_path(cls, path: str) -> 'Run':
        """
        Return a Run object from a given path.
        """
        pass

    @classmethod
    def is_valid_run(cls, path: str) -> bool:
        """
        Check if the path belongs to a valid run.
        """
        pass