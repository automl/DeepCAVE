from typing import Any, Iterable, Optional, Union

import base64
import random
import string
from io import BytesIO

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from dash import html
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def get_random_string(length: int) -> str:
    if length < 0:
        raise ValueError("Length has to be greater than 0")
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def matplotlib_to_html_image(fig: plt.Figure) -> html.Img:
    # TODO(dwoiwode): Duplicate code (see ./layout.py)?
    # create a virtual file which matplotlib can use to save the figure
    buffer = BytesIO()
    # save the image to memory to display in the web
    fig.savefig(buffer, format="png", transparent=True)
    buffer.seek(0)
    # display any kind of image taken from
    # https://github.com/plotly/dash/issues/71
    encoded_image = base64.b64encode(buffer.read())
    return html.Img(
        src=f"data:image/png;base64,{encoded_image.decode()}", className="img-fluid"
    )


def encode_data(
    data: pd.DataFrame, cs: Optional[ConfigurationSpace] = None
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[pd.Series, pd.Series]]]:
    # converts only columns with "config." prefix
    if cs:
        return _encode(data, cs)
    else:
        return _infer_encoding(data)


def _infer_encoding(data: pd.DataFrame) -> pd.DataFrame:
    from_cols = []
    for col in data.columns:
        if "config." not in col:
            continue
        if data[col].dtype == "object":
            # if not numerical convert it to numerical, via one-hot encoding
            from_cols.append(col)
    if from_cols:
        dummies = pd.get_dummies(data[from_cols], prefix=from_cols, prefix_sep=".")
        data.drop(columns=from_cols, inplace=True)
        data = data.join(dummies)
    return data


def _encode(data: pd.DataFrame, cs: Optional):
    def _add_entry(hp_name, hp_choices, from_cols, to_cols, choices, ordinal):
        col_name = f"config.{hp_name}"
        from_cols.append(col_name)
        if ordinal:
            columns = [col_name]
        else:
            columns = [f"config.{hp_name}.{col}" for col in list(hp_choices)]
        to_cols.extend(columns)
        choices.append(list(hp_choices))
        return col_name, columns, list(hp_choices)

    def _transform(data, from_cols, to_cols, choices, transformer_class):
        add_kwargs = {}
        if transformer_class is OneHotEncoder:
            add_kwargs = {"sparse": False}
        data[to_cols] = pd.DataFrame(
            transformer_class(categories=choices, **add_kwargs).fit_transform(
                data[from_cols]
            ),
            columns=to_cols,
            index=data.index,
        )
        if transformer_class is OneHotEncoder:
            data.drop(columns=from_cols, inplace=True)
        return data

    org_cols = {col: col for col in data.columns if "config." in col}
    # categorical
    cat_from_cols, cat_to_cols, cat_choices = [], [], []
    ord_from_cols, ord_to_cols, ord_choices = [], [], []
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            from_col, to_col, choices = _add_entry(
                hp.name,
                hp.choices,
                cat_from_cols,
                cat_to_cols,
                cat_choices,
                ordinal=False,
            )
            org_cols[from_col] = to_col
        elif isinstance(hp, IntegerHyperparameter):
            data["config." + hp.name] = data["config." + hp.name].astype(int)
        elif isinstance(hp, FloatHyperparameter):
            data["config." + hp.name] = data["config." + hp.name].astype(float)
        elif isinstance(hp, OrdinalHyperparameter):
            _add_entry(
                hp.name,
                hp.sequence,
                ord_from_cols,
                ord_to_cols,
                ord_choices,
                ordinal=True,
            )

    data = _transform(data, cat_from_cols, cat_to_cols, cat_choices, OneHotEncoder)
    data = _transform(data, ord_from_cols, ord_to_cols, ord_choices, OrdinalEncoder)

    return data, org_cols


def add_prefix_to_dict(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Adds a prefix to every key in a dictionary"""
    return {f"{prefix}{key}": value for key, value in data.items()}


def add_prefix_to_list(data: Iterable[str], prefix: str) -> list[str]:
    """Adds a prefix to every item in an iterable (e.g. a list). Returns a list"""
    return [f"{prefix}{item}" for item in data]
