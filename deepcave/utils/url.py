#  noqa: D400
"""
# URL

This module creates and parses an URL according to its input information.
"""

from typing import Any, Dict, Optional

import json
import urllib


def create_url(pathname: str, inputs: Dict[str, Any]) -> str:
    """
    Create an URL with the given pathname and inputs.

    Parameters
    ----------
    pathname : str
        The name of the path for the URL.
    inputs : Dict[str, Any]
        A dictionary containing the input parameters for the URL.

    Returns
    -------
    str
        The URL.

    """
    string = json.dumps(inputs)
    converted_string = urllib.parse.quote(string)

    return f"{pathname}/?inputs={converted_string}"


def parse_url(pathname: str) -> Optional[Dict[str, Any]]:
    """
    Parse the URL and extract input information if possible.

    Parameters
    ----------
    pathname : str
        The name of the URL.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary with the extracted inputs if available.
        Otherwise return None.
    """
    url = urllib.parse.urlparse(pathname)
    query = urllib.parse.parse_qs(url.query)

    if "inputs" in query:
        string = query["inputs"][0]

        return json.loads(string)

    return None
