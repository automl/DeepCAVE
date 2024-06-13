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
