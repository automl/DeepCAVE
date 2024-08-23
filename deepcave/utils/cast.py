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
# Cast

This module provides a utility to convert any value to an int if possible.
"""
from typing import Any, Optional


def optional_int(value: Any) -> Optional[int]:
    """
    Convert a value to an int if possible.

    Parameters
    ----------
    value : Any
        The value to be turned into an int.

    Returns
    -------
    Optional[int]
        The converted int value.
        If not possible, return None.
    """
    if value is None:
        return None

    return int(value)
