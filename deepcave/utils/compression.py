from typing import Dict, List, TypeVar, Union

import json

import numpy as np
import pandas as pd

JSON_DENSE_SEPARATORS = (",", ":")
JSON_DEFAULT_SEPARATORS = (",", ": ")
TYPE = TypeVar("TYPE")


def serialize(data: Union[Dict, List, pd.DataFrame]) -> str:
    """
    Serialize a dataframe to a string.
    """

    class Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if isinstance(data, pd.DataFrame):
        # TODO(dwoiwode): Why not just data.to_json()? Or at least make json smaller in dumps
        return json.dumps(json.loads(data.to_json()), separators=JSON_DENSE_SEPARATORS)

    return json.dumps(data, cls=Encoder, separators=JSON_DENSE_SEPARATORS)


def deserialize(string: str, dtype: TYPE = pd.DataFrame) -> TYPE:
    """
    Deserialize a dataframe from a string.
    """
    if dtype == pd.DataFrame:
        return pd.DataFrame.from_dict(json.loads(string))

    return json.loads(string)
