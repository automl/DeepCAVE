import json
import pandas as pd


def serialize_df(df):
    """
    Serialize a dataframe to a string.
    """

    return json.dumps(json.loads(df.to_json()))


def deserialize_df(string):
    """
    Deserialize a dataframe from a string.
    """

    return pd.DataFrame.from_dict(json.loads(string))
