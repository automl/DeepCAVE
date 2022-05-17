from os import access
from typing import Any, Dict, List, Optional

import pandas as pd
import dash_bootstrap_components as dbc
import base64
import io

from dash import html, dcc
from deepcave.utils.hash import string_to_hash


def help_button(text: str):
    id = "help-button-" + string_to_hash(text)

    return html.Span(
        [
            html.I(id=id, className="ms-1 far fa-question-circle"),
            dbc.Popover(
                dcc.Markdown(text, className="p-3 pb-1"),
                target=id,
                trigger="hover",
                placement="top",
            ),
        ]
    )


def render_table(df):
    pass


def render_mpl_figure(fig):
    # create a virtual file which matplotlib can use to save the figure
    buffer = io.BytesIO()
    # save the image to memory to display in the web
    fig.savefig(buffer, format="png", transparent=True)
    buffer.seek(0)
    # display any kind of image taken from
    # https://github.com/plotly/dash/issues/71
    encoded_image = base64.b64encode(buffer.read())
    return "data:image/png;base64,{}".format(encoded_image.decode())


def get_slider_marks(
    strings: Optional[List[Dict[str, Any]]] = None, steps=10, access_all=False
) -> Dict[int, str]:
    marks = {}
    if strings is None:
        marks[0] = {"label": str("None")}
        return marks

    if len(strings) < steps:
        steps = len(strings)

    for i, string in enumerate(strings):
        if i % (len(strings) / steps) == 0:
            marks[i] = {"label": str(string)}
        else:
            if access_all:
                marks[i] = {"label": ""}

    # Also include the last mark
    marks[len(strings) - 1] = {"label": str(strings[-1])}

    return marks


def get_select_options(
    labels=None, values=None, disabled=None, binary=False
) -> List[Dict[str, Any]]:
    """
    If values are none use labels as values.
    If both are none return empty list.
    """

    if labels is None and values is None:
        if binary:
            return [{"label": "Yes", "value": True}, {"label": "No", "value": False}]

        return []

    if values is None:
        values = labels

    if labels is None:
        labels = values

    if len(labels) != len(values):
        raise ValueError(f"Labels and values have unequal length ({len(labels)} != {len(values)})")

    options = []
    for idx, (l, v) in enumerate(zip(labels, values)):
        if disabled is not None:
            options.append({"label": l, "value": v, "disabled": disabled[idx]})
        else:
            options.append({"label": l, "value": v})

    return options


def get_checklist_options(labels=None, values=None, binary=False):
    return get_select_options(labels=labels, values=values, binary=binary)


def get_radio_options(labels=None, values=None, binary=False):
    return get_select_options(labels=labels, values=values, binary=binary)


def create_table(output: Dict[str, str], mb=True) -> dbc.Table:
    mb = "mb-0" if not mb else ""
    df = pd.DataFrame(output)

    return dbc.Table.from_dataframe(df, striped=True, bordered=True, className=mb)
