"""
# Exceptions.

This module provided utilities for different errors concerning the runs.
The Exceptions will be raised, if a directory is not a valid run,
as well as if runs are not mergable.
"""


class NotValidRunError(Exception):
    """Raised if directory is not a valid run."""

    pass


class NotMergeableError(Exception):
    """Raised if two or more runs are not mergeable."""

    pass
