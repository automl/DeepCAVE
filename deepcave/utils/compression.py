import json
import numpy as np
import pandas as pd


def serialize(data):
    """
    Serialize a dataframe to a string.
    """

    class Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if isinstance(data, pd.DataFrame):
        return json.dumps(json.loads(data.to_json()))

    return json.dumps(data, cls=Encoder)


def deserialize(string, dtype=pd.DataFrame):
    """
    Deserialize a dataframe from a string.
    """

    if dtype == pd.DataFrame:
        return pd.DataFrame.from_dict(json.loads(string))

    return json.loads(string)
