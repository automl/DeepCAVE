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

#  noqa: D400
"""
# Compression

This module provides utilities for serializing and deserializing a dataframe from/to a string.

## Classes
    - Encoder: This class defines a custom JSON Encoder.

## Constants
    - JSON_DENSE_SEPARATORS: Tuple(str, str)
    - JSON_DEFAULT_SEPARATORS: Tuple(str, str)
    - TYPE: TypeVar
"""

from typing import Any, Dict, List, TypeVar, Union

import json

import numpy as np
import pandas as pd

JSON_DENSE_SEPARATORS = (",", ":")
JSON_DEFAULT_SEPARATORS = (",", ": ")
TYPE = TypeVar("TYPE")


class Encoder(json.JSONEncoder):
    """Define a custom JSON Encoder."""

    def default(self, obj: Any) -> Any:
        """
        Return the object as list if np.ndarray.

        Parameters
        ----------
        obj : Any
            The object to be converted.

        Returns
        -------
        Any
            The converted object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def serialize(data: Union[Dict, List, pd.DataFrame]) -> str:
    """
    Serialize a dataframe to a string.

    Parameters
    ----------
    data : Union[Dict, List, pd.DataFrame]
        The dataframe to be serialized.

    Returns
    -------
    str
        The serialized object as a JSON formatted string.
    """
    if isinstance(data, pd.DataFrame):
        # TODO(dwoiwode): Why not just data.to_json()? Or at least make json smaller in dumps
        return json.dumps(json.loads(data.to_json()), separators=JSON_DENSE_SEPARATORS)

    return json.dumps(data, cls=Encoder, separators=JSON_DENSE_SEPARATORS)


def deserialize(string: str, dtype: TYPE = pd.DataFrame) -> TYPE:
    """
    Deserialize a dataframe from a string.

    Parameters
    ----------
    string : str
        The string to be deserialized.
    dtype : TYPE, optional
        The type of the object.
        Default is pd.DataFrame.

    Returns
    -------
    TYPE
        The deserialized object.
    """
    if dtype == pd.DataFrame:
        return pd.DataFrame.from_dict(json.loads(string))

    return json.loads(string)
