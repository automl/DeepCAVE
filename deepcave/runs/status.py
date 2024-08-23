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
    FAILED: int
    PRUNED: int
    UNKNOWN: int
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
    FAILED = 7
    PRUNED = 8
    UNKNOWN = 9

    def to_text(self) -> str:
        """
        Convert name to simpler, lower case text format.

        Returns
        -------
        str
            The converted name in lower case with spaces added
        """
        return self.name.lower().replace("_", " ")
