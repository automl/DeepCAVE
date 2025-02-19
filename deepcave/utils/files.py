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
# Files

This module provides a utility to create directories from a filename.
"""

from typing import Union

from pathlib import Path


def make_dirs(filename: Union[str, Path], parents: bool = True) -> None:
    """
    Create a directory.

    Parameters
    ----------
    filename : Union[str, Path]
        The name and path of the file.
    parents : bool, optional
        Whether intermediate directories should be created.
        Default is True.
    """
    path = Path(filename)
    if path.suffix != "":  # Is file
        path = path.parent

    path.mkdir(exist_ok=True, parents=parents)
