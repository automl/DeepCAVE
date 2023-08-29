#  noqa: D400
"""
# Compression

This module provides utilities for serializing and deserializing a dataframe from/to a string.

## Classes
    - Encoder: This class defines a custom JSON Encoder.

## Constants
    - JSON_DENSE_SEPARATORS = (",", ":")
    - JSON_DEFAULT_SEPARATORS = (",", ": ")
    - TYPE = TypeVar("TYPE")

## Contents
    - serilaize: Serialize a dataframe to a string.
    - deserialize: Deserialize a dataframe from a string.
    - default: Return the object either as list or als JSONEncoder.
"""

from typing import Any, Dict, List, TypeVar, Union

import json

import numpy as np
import pandas as pd

JSON_DENSE_SEPARATORS = (",", ":")
JSON_DEFAULT_SEPARATORS = (",", ": ")
TYPE = TypeVar("TYPE")


def serialize(data: Union[Dict, List, pd.DataFrame]) -> str:
    """Serialize a dataframe to a string."""

    class Encoder(json.JSONEncoder):
        """
        Define a custom JSON Encoder.

        Methods
        -------
        default
            Return the object either as list or als JSONEncoder.
        """

        def default(self, obj: Any) -> Any:
            """
            Return the object either as list or als JSONEncoder.

            Parameters
            ----------
            obj : Any
                The object to be converted.

            Returns
            -------
            Any
                The converted object.
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if isinstance(data, pd.DataFrame):
        # TODO(dwoiwode): Why not just data.to_json()? Or at least make json smaller in dumps
        return json.dumps(json.loads(data.to_json()), separators=JSON_DENSE_SEPARATORS)

    return json.dumps(data, cls=Encoder, separators=JSON_DENSE_SEPARATORS)


def deserialize(string: str, dtype: TYPE = pd.DataFrame) -> TYPE:
    """Deserialize a dataframe from a string."""
    if dtype == pd.DataFrame:
        return pd.DataFrame.from_dict(json.loads(string))

    return json.loads(string)
