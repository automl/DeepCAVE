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

#  noqa: D400
"""
# DeepCAVE

This module defines the DeepCAVE run object.

## Classes
    - DeepCAVERun: Create a DeepCAVE run and provide handling utilities.
"""

from typing import Union

from pathlib import Path

from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash


class DeepCAVERun(Run):
    """
    Create a DeepCAVE run and provide handling utilities.

    Properties
    ----------
    path : Path
        The path the run.
    """

    prefix = "DeepCAVE"
    _initial_order = 1

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

        # Use hash of history.json as id
        return file_to_hash(self.path / "history.jsonl")

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

        return Path(self.path / "history.jsonl").stat().st_mtime

    @classmethod
    def from_path(cls, path: Path) -> "DeepCAVERun":
        """
        Get a DeepCAVE run from a given path.

        Parameters
        ----------
        path : Path
            The path to base the run on.

        Returns
        -------
        The DeepCAVE run.
        """
        return DeepCAVERun(path.stem, path=Path(path))
