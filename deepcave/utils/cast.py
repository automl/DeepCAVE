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
