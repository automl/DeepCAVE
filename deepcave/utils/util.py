#  noqa: D400
"""
# Util

This module provides utilities for string generation and shortening.
It also provides a function to get the difference between now and a given timestamp.

## Contents
    - get_random_string: Get a random string with a specific length.
    - short_string: Shorten the given string.
    - get_latest_change: Get the difference between now and a given timestamp.
"""
from typing import Any

import datetime
import random
import string


def get_random_string(length: int) -> str:
    """
    Get a random string with a specific length.

    Parameters
    ----------
    length : int
        The length of the string.

    Returns
    -------
    str
        The random string with the given length.

    Raises
    ------
    ValueError
        If the length is smaller 0.
    """
    if length < 0:
        raise ValueError("Length has to be greater than 0")
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def short_string(value: Any, length: int = 30, *, mode: str = "prefix") -> str:
    """
    Shorten the given string.

    Cut either at prefix or at suffix if necessary.

    Parameters
    ----------
    value : Any
        The value or string to shorten.
    length : int, optional
        The length of the returned string.
        Default is 30.
    mode : str, optional
        Define how to shorten the string.
        Default is "prefix".

    Returns
    -------
    str
        The shortened string.

    Raises
    ------
    ValueError
        If the given mode is unknown.
    """
    value = str(value)
    if len(value) <= length:
        return value

    cut_length = length - 3  # For 3 dots (...)
    if mode == "prefix":
        return f"...{value[-cut_length:]}"
    elif mode == "suffix":
        return f"{value[:cut_length]}..."
    raise ValueError(f"Unknown mode '{mode}'")


def get_latest_change(st_mtime: int) -> str:
    """
    Get the difference between now and a given timestamp.

    Parameters
    ----------
    st_mtime : int
        A timestamp to calculate the difference from.

    Returns
    -------
    str
        A string containig the passed time.
    """
    t = datetime.datetime.fromtimestamp(st_mtime)
    s_diff = (datetime.datetime.now() - t).seconds
    d_diff = (datetime.datetime.now() - t).days

    if s_diff < 60:
        return "Some seconds ago"
    elif s_diff < 3600:
        return f"{int(s_diff / 60)} minutes ago"
    elif s_diff < 86400:
        return f"{int(s_diff / 60 / 60)} hours ago"
    elif d_diff < 7:
        return f"{d_diff} days ago"
    else:
        return t.strftime("%Y/%m/%d")
