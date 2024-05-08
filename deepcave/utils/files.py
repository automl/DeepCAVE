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
