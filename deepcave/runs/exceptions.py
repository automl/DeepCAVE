from enum import Enum


class NotValidRunError(Exception):
    """Raised if directory is not a valid run."""

    pass


class NotMergeableError(Exception):
    """Raised if two or more runs are not mergeable"""

    pass


class RunInequality(Enum):
    """Check why runs were not compatible."""

    INEQ_META = 1
    INEQ_OBJECTIVE = 2
    INEQ_BUDGET = 3
    INEQ_CONFIGSPACE = 4
