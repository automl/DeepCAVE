# noqa: D400
"""
# Status

This module provides the information about the status of a run.

A utility to convert a string text to a simpler, lower case text format is provided.

## Classes
    - Status: Represent the status of a run as an Enum.

## Constants
    SUCCESS: int
    TIMEOUT: int
    MEMORYOUT: int
    CRASHED: int
    ABORTED: int
    NOT_EVALUATED: int
"""

from enum import IntEnum


class Status(IntEnum):
    """
    Represent the status of a run as an Enum.

    A utility to convert a string text to a simpler, lower case text format is provided.

    Properties
    ----------
    name : str
        The status name.
    """

    SUCCESS = 1
    TIMEOUT = 2
    MEMORYOUT = 3
    CRASHED = 4
    ABORTED = 5
    NOT_EVALUATED = 6

    def to_text(self) -> str:
        """
        Convert name to simpler, lower case text format.

        Returns
        -------
        str
            The converted name in lower case with spaces added
        """
        return self.name.lower().replace("_", " ")
