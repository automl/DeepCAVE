#  noqa: D400
"""
# Layout

This module provides utilities to customize the layout.

This includes a help button, slider marks, select/checklist/radio options.
"""
from typing import Any, Dict, List, Optional, Union

import uuid

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from deepcave.utils.hash import string_to_hash


def help_button(text: str, placement: str = "top") -> html.Span:
    """
    Generate button with help icon.

    Displays popover when hovered over that contains the provided text.

    Parameters
    ----------
    text : str
        The text that will be displayed in the popover.
    placement : str, optional
        The placement of the button, default is Top.

    Returns
    -------
    html.Span
        An html structure, that wraps the icon and the popover.
    """
    id = "help-button-" + string_to_hash(text)
    id += str(uuid.uuid1())

    return html.Span(
        [
            html.I(id=id, className="ms-1 far fa-question-circle"),
            dbc.Popover(
                dcc.Markdown(text, className="p-3 pb-1"),
                target=id,
                trigger="hover",
                placement=placement,
            ),
        ]
    )


def render_table(df: pd.DataFrame) -> None:  # noqa: D103
    pass


def get_slider_marks(
    strings: Optional[List[Any]] = None,
    steps: int = 10,
    access_all: bool = False,
) -> Dict[int, Dict[str, str]]:
    """
    Generate a dictionary containing slider marks.

    The slider marks are based on the provided list of dictionaries.

    Parameters
    ----------
    strings : Optional[List[Dict[str, Any]]], optional
        List of dictionaries containing information about the marks.
        Default value is None.
    steps : int, optional
        Number of steps or marks on the slider.
        Default is 10.
    access_all : bool, optional
        Indicates whether to create marks for all items.
        Default is False.

    Returns
    -------
    Dict[int, Dict[str, str]]
        Contains information about the positions and labels of the marks.
    """
    marks = {}
    if strings is None:
        marks[0] = {"label": "None"}
        return marks

    if len(strings) < steps:
        steps = len(strings)

    for i, string in enumerate(strings):
        if i % int(len(strings) / steps) == 0:
            marks[i] = {"label": str(string)}
        else:
            if access_all:
                marks[i] = {"label": ""}

    # Also include the last mark
    marks[len(strings) - 1] = {"label": str(strings[-1])}

    return marks


def get_select_options(
    labels: Any = None,
    values: Any = None,
    disabled: Union[List[bool], None] = None,
    binary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get dictionaries of the selected options.

    If values are none use labels as values. If both are none return empty list.

    Parameters
    ----------
    labels : Any, optional
        To specify the labels of the options.
        If labels are None, use values als labels.
        Default is None.
    values : Any, optional
        To specify the values corresponding to the options.
        If values are None, use labels as values.
        Default is None
    disabled : Union[List[Bool], None], optional
        A list containing information whether each option should be disabled or not.
        Default is None.
    binary : bool, optional
        Define whether to return two dictionaries with binary choices.
        Only when both values and labels are None.
        Default is False.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries of the selected options.

    Raises
    ------
    ValueError
        If the labels and values do not have the same length.
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


def get_checklist_options(
    labels: Any = None, values: Any = None, binary: bool = False
) -> List[Dict[str, Any]]:
    """
    Get a list of dictionaries of checklist options.

    Parameters
    ----------
    labels : Any, optional
        To specify the labels of the options.
        If labels are None, use values als labels.
        Default is None.
    values : Any, optional
        To specify the values corresponding to the options.
        If values are None, use labels as values.
        Default is None
    binary : bool, optional
        Define whether to return two dictionaries with binary choices.
        Only when both values and labels are None.
        Default is False.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries of the checklist options.

    Raises
    ------
    ValueError
        If the labels and values do not have the same length.
    """
    return get_select_options(labels=labels, values=values, binary=binary)


def get_radio_options(
    labels: Any = None, values: Any = None, binary: Any = False
) -> List[Dict[str, Any]]:
    """
    Get a list of dictionaries of radio options.

    Parameters
    ----------
    labels : Any, optional
        To specify the labels of the options.
        If labels are None, use values als labels.
        Default is None.
    values : Any, optional
        To specify the values corresponding to the options.
        If values are None, use labels as values.
        Default is None
    binary : bool, optional
        Define whether to return two dictionaries with binary choices.
        Only when both values and labels are None.
        Default is False.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries of the radio options.

    Raises
    ------
    ValueError
        If the labels and values do not have the same length.
    """
    return get_select_options(labels=labels, values=values, binary=binary)


def create_table(
    output: Dict[str, List[str]],
    fixed: bool = False,
    head: bool = True,
    striped: bool = True,
    mb: bool = True,
) -> dbc.Table:
    """
    Create a table from the given data.

    Parameters
    ----------
    output : Dict[str, List[str]]
        Containing the information for the table.
    fixed : bool, optional
        Indicates whether the table has a fixed layout.
        Default is False.
    head : bool, optional
        Indicates whether the head will be included.
        Default is True.
    striped : bool, optional
        Indicates whether the rows will be striped.
        Default is True.
    mb : bool, optional
        Indicates whether the table has a margin.
        Default is True.

    Returns
    -------
    dbc.Table
        The created dash bootstrap component (DBC) table.
    """
    className = ""
    if not head:
        className += "exclude-head "
    if fixed:
        className += "fixed "
    className += "mb-0" if not mb else ""
    df = pd.DataFrame(output)

    return dbc.Table.from_dataframe(df, striped=striped, bordered=True, className=className)
