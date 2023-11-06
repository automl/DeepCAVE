#  noqa: D400
"""
# Hash

This module provides utilities to convert strings and files to hash.
"""

import hashlib
from pathlib import Path


def string_to_hash(string: str) -> str:
    """
    Convert a string to a hash.

    Parameters
    ----------
    string : str
        The string to be converted to hash.

    Returns
    -------
    str
        The hash object.
    """
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()


def file_to_hash(filename: Path) -> str:
    """
    Convert a file to a hash.

    Parameters
    ----------
    filename : Path
        The path to the file to be converted.

    Returns
    -------
    str
        The hash object.
    """
    hash = hashlib.md5()
    with Path(filename).open("rb") as f:
        while chunk := f.read(4082):
            hash.update(chunk)

    return hash.hexdigest()
