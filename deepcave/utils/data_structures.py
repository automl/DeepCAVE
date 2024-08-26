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
# Data Structures

This module can be used for updating one dictionary with another dictionary inplace.
"""

from typing import Dict


def update_dict(a: Dict[str, Dict], b: Dict[str, Dict]) -> None:
    """
    Update dictionary a with dictionary b inplace.

    Parameters
    ----------
    a : Dict[str, Dict]
        Dictionary to be updated.
    b : Dict[str, Dict]
        Dictionary to be added.
    """
    for k1, v1 in b.items():
        if k1 not in a:
            a[k1] = {}

        for k2, v2 in v1.items():
            a[k1][k2] = v2
