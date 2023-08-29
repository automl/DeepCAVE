"""
# Status.

This module provides the information about the status of a run.

## Contents
    - to_text: Convert the name to a simpler text format.
"""

from enum import IntEnum


class Status(IntEnum):
    """
    Represent the status of a run as an Enum.

    Methods
    -------
    to_text
        Convert the name to a simpler text format.
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
