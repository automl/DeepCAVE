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
# Exceptions

This module provides utilities for different errors concerning the runs.

Exceptions will be raised, if a directory is not a valid run,
as well as if runs are not mergeable.

## Classes
    - NotValidRunError: Raised if directory is not a valid run.
    - NotMergeableError: Raised if two or more runs are not mergeable.
"""

from enum import Enum


class NotValidRunError(Exception):
    """Raised if directory is not a valid run."""

    pass


class NotMergeableError(Exception):
    """Raised if two or more runs are not mergeable."""

    pass


class RunInequality(Enum):
    """Check why runs were not compatible."""

    INEQ_META = 1
    INEQ_OBJECTIVE = 2
    INEQ_BUDGET = 3
    INEQ_CONFIGSPACE = 4
    INEQ_SEED = 5
