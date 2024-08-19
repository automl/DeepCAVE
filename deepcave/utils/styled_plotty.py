# noqa: D400
"""
# Styled Plotty

This module provides utilities for styling and customizing different plots with plotly.
For this, it uses plotly as well as dash.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import itertools
import re

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    IntegerHyperparameter,
)
from dash import html
from dash.development.base_component import Component

from deepcave import interactive
from deepcave.constants import CONSTANT_VALUE, NAN_LABEL, NAN_VALUE
from deepcave.runs import AbstractRun
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


@interactive
def save_image(figure: go.Figure, name: str) -> None:
    """
    Save a plotly figure as an image.

    Parameters
    ----------
    figure : go.Figure
        Plotly figure.
    name : str
        Name of the image with extension.
        Will be automatically saved to the cache.
    """
    from deepcave import config

    if not config.SAVE_IMAGES:
        return

    ratio = 16 / 9
    width = 500
    height = int(width / ratio)
    path = config.CACHE_DIR / "figures" / name

    figure.write_image(path, width=width, height=height)
    logger.info(f"Saved figure {name} to {path}.")


def hex_to_rgb(hex_string: str) -> Tuple[int, int, int]:
    """
    Convert a hex_string to a tuple of rgb values.

    Requires format including #, e.g.:
    #000000
    #ff00ff

    Parameters
    ----------
    hex_string : str
        The hex string to be converted.

    Returns
    -------
    Tuple[int, int, int]
        A Tuple of the converted RGB values

    Raises
    ------
    ValueError
        If the hex string is longer than 7.
        If there are invalid characters in the hex string.
    """
    if len(hex_string) != 7:
        raise ValueError(f"Invalid length for #{hex_string}")

    if any(c not in "0123456789ABCDEF" for c in hex_string.lstrip("#").upper()):
        raise ValueError(f"Invalid character in #{hex_string}")

    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)


def get_color(id_: int, alpha: float = 1) -> Union[str, Tuple[float, float, float, float]]:
    """
    Get an RGBA Color, currently (Plotly version 5.3.1) there are 10 possible colors.

    Parameters
    ----------
    id_ : int
        ID for retrieving a specific color.
    alpha : float, optional
        Alpha value for the color, by default 1.

    Returns
    -------
    Union[str, Tuple[float, float, float, float]]
        The color from the color palette.
    """
    if id_ < 10:
        color = px.colors.qualitative.Plotly[id_]
    else:
        color = px.colors.qualitative.Alphabet[id_ - 10]

    r, g, b = hex_to_rgb(color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_discrete_heatmap(
    x: List[Union[float, int]], y: List[int], values: List[Any], labels: List[Any]
) -> go.Heatmap:
    """
    Generate a discrete colorscale from a (nested) list or numpy array of values.

    Parameters
    ----------
    x : List[Union[float, int]]
        List of values that present the x-axis of the heatmap.
    y : List[int]
         List of values that present the y-axis of the heatmap.
    values : List[Any]
        Contains the data values for the heatmap.
    labels : List[Any]
        Contains the labels corresponding to the values.

    Returns
    -------
    go.Heatmap
        A Plotly Heatmap object corresponding to the input.
    """
    flattened_values = list(itertools.chain(*values))
    flattened_labels = list(itertools.chain(*labels))

    unique_values = []
    unique_labels = []

    for value, label in zip(flattened_values, flattened_labels):
        if value not in unique_values:
            unique_values += [value]
            unique_labels += [label]

    sorted_indices = np.argsort(np.array(unique_values))
    unique_sorted_values = []
    unique_sorted_labels = []
    for idx in sorted_indices:
        unique_sorted_values += [unique_values[idx]]
        unique_sorted_labels += [unique_labels[idx]]

    # Now they are given new ids, and new z values should be created
    # For that a mapping from old to new is needed
    mapping = {}
    v = []
    for new, old in enumerate(unique_sorted_values):
        mapping[old] = new / len(unique_sorted_values)
        v += [new]

    z = values
    for i1, v1 in enumerate(values):
        for i2, v2 in enumerate(v1):
            z[i1][i2] = mapping[v2]

    n_intervals_int = v + [len(v)]
    n_intervals = [
        (i - n_intervals_int[0]) / (n_intervals_int[-1] - n_intervals_int[0])
        for i in n_intervals_int
    ]
    colors = [get_color(i) for i in range(len(n_intervals))]

    discrete_colorscale = []
    for k in range(len(v)):
        discrete_colorscale.extend([[n_intervals[k], colors[k]], [n_intervals[k + 1], colors[k]]])

    tickvals = [np.mean(n_intervals[k : k + 2]) for k in range(len(n_intervals) - 1)]
    ticktext = unique_sorted_labels

    x_str = [str(i) for i in x]
    y_str = [str(i) for i in y]

    return go.Heatmap(
        x=x_str,
        y=y_str,
        z=z,
        showscale=True,
        colorscale=discrete_colorscale,
        colorbar={"tickvals": tickvals, "ticktext": ticktext, "tickmode": "array"},
        zmin=0,
        zmax=1,
        # hoverinfo="skip",
    )


def prettify_label(label: Union[str, float, int]) -> str:
    """
    Take a label and prettifies it.

    E.g. floats are shortened.

    Parameters
    ----------
    label : Union[str, float, int]
        Label, which should be prettified.

    Returns
    -------
    str
        Prettified label.
    """
    if type(label) == float:
        if str(label).startswith("0.00") or "e-" in str(label):
            label = np.format_float_scientific(label, precision=2)

            # Replace 1.00e-03 to 1e-03
            if ".00" in label:
                label = label.replace(".00", "")

            # Replace 1e-03 to 1e-3
            if "e-0" in label:
                label = label.replace("e-0", "e-")
        else:
            # Round to 2 decimals
            label = np.round(label, 2)

    return str(label)


def get_hyperparameter_ticks(
    hp: Hyperparameter,
    additional_values: Optional[List] = None,
    ticks: int = 4,
    include_nan: bool = True,
) -> Tuple[List, List]:
    """
    Generate tick data for both tickvals and ticktext.

    The background is that
    you might have encoded data, but you don't want to show all of them.
    With this function, only 6 (default) values are shown. This behavior is
    ignored if `hp` is categorical.

    Parameters
    ----------
    hp : Hyperparameter
        Hyperparameter to generate ticks from.
    additional_values : Optional[List], optional
        Additional values, which are forced in addition. By default, None.
    ticks : int, optional
        Number of ticks, by default 4
    include_nan : bool, optional
        Whether "nan" as tick should be included. By default True.

    Returns
    -------
    Tuple[List, List]
        tickvals and ticktext.
    """
    # This is basically the inverse of `encode_config`.
    tickvals: List[Any]
    if isinstance(hp, CategoricalHyperparameter):
        ticktext = hp.choices
        if len(ticktext) == 1:
            tickvals = [0]
        else:
            tickvals = [hp.to_vector(choice) / (len(hp.choices) - 1) for choice in hp.choices]

    elif isinstance(hp, Constant):
        tickvals = [CONSTANT_VALUE]
        ticktext = [hp.value]
    else:
        min_v = 0
        max_v = 1

        values: List[Union[float, int]] = [min_v]

        # Get values for each tick
        factors = [i / (ticks - 1) for i in range(1, ticks - 1)]

        for factor in factors:
            new_v = (factor * (max_v - min_v)) + min_v
            values += [new_v]

        values += [max_v]

        tickvals = []
        ticktext = []

        inverse_values = []
        for value in values:
            inverse_values += [hp.to_value(value)]

        # Integers are rounded, they are mapped
        if isinstance(hp, IntegerHyperparameter):
            for label in inverse_values:
                value = hp.to_vector(label)

                if value not in tickvals:
                    tickvals += [value]
                    ticktext += [label]

            if additional_values is not None:
                # Now add additional values are added
                for value in additional_values:
                    if not (value is None or np.isnan(value) or value == NAN_VALUE):
                        label = hp.to_value(value)
                        value = hp.to_vector(label)

                        if value not in tickvals:
                            tickvals += [value]
                            ticktext += [label]
        else:
            for value, label in zip(values, inverse_values):
                tickvals += [value]
                ticktext += [label]

            if additional_values is not None:
                # Now additional values are added
                for value in additional_values:
                    if (
                        not (value is None or np.isnan(value) or value == NAN_VALUE)
                        and value not in tickvals
                    ):
                        tickvals += [value]
                        ticktext += [hp.to_value(value)]

    ticktext = [prettify_label(label) for label in ticktext]

    if include_nan:
        tickvals += [NAN_VALUE]
        ticktext += [NAN_LABEL]

    return tickvals, ticktext


def get_hyperparameter_ticks_from_values(
    values: List, labels: List, forced: Optional[List[bool]] = None, ticks: int = 6
) -> Tuple[List, List]:
    """
    Generate tick data for both values and labels.

    The background is that
    you might have encoded data, but you don't want to show all of them.
    With this function, only 6 (default) values are shown. This behavior is
    ignored if `values` is a list of strings.

    Parameters
    ----------
    values : List
        List of values.
    labels : List
        List of labels. Must be the same size as `values`.
    forced : Optional[List[bool]], optional
        List of booleans. If True, displaying the particular tick is enforced.
        Independent of `ticks`.
    ticks : int, optional
        Number of ticks and labels to show. By default 6.

    Returns
    -------
    Tuple[List, List]
        Returns tickvals and ticktext as list.

    Raises
    ------
    RuntimeError
        If values contain both strings and non-strings.
    """
    assert len(values) == len(labels)

    unique_values = []  # df[hp_name].unique()
    unique_labels = []  # df_labels[hp_name].unique()
    for value, label in zip(values, labels):
        if value not in unique_values and label not in unique_labels:
            unique_values.append(value)
            unique_labels.append(label)

    return_all = False
    for v1, v2 in zip(unique_values, unique_values[1:]):
        if isinstance(v1, str) or isinstance(v2, str):
            if type(v1) != type(v2):
                raise RuntimeError("Values have strings and non-strings.")

            return_all = True

    tickvals = []
    ticktext = []

    # If there are less than x values, they are also shown
    if return_all or len(unique_values) <= ticks:
        # Make sure there are no multiple (same) labels for the same value
        for value, label in zip(unique_values, unique_labels):
            tickvals.append(value)
            ticktext.append(label)
    else:
        # Add min+max values
        for idx in [np.argmin(values), np.argmax(values)]:
            tickvals.append(values[idx])
            ticktext.append(labels[idx])

        # After min and max values are added,
        # intermediate values should be added too
        min_v = np.min(values)
        max_v = np.max(values)

        # Get values for each tick
        factors = [i / (ticks - 1) for i in range(1, ticks - 2)]

        for factor in factors:
            new_v = (factor * (max_v - min_v)) + min_v
            idx = np.abs(unique_values - new_v).argmin(axis=-1)

            value = unique_values[idx]
            label = unique_labels[idx]

            # Ignore if they are already in the list
            if value not in tickvals:
                tickvals.append(value)
                ticktext.append(label)

    # Show forced ones
    if forced is not None:
        for value, label, force in zip(values, labels, forced):
            if force and value not in tickvals:
                tickvals.append(value)
                ticktext.append(label)

    return tickvals, ticktext


def get_hovertext_from_config(
    run: AbstractRun, config_id: int, budget: Optional[Union[int, float]] = None
) -> str:
    """
    Generate hover text with metrics for a configuration.

    The method gets information about a given configuration, including a link, its objectives,
    budget, costs and hyperparameters.

    Parameters
    ----------
    run : AbstractRun
        The run instance
    config_id : int
        The id of the configuration
    budget : Optional[Union[int, float]]
            Budget to get the hovertext for. If no budget is given, the highest budget is chosen.
            By default None.

    Returns
    -------
    str
        The hover text string containing the configuration information.
    """
    if config_id < 0:
        return ""

    # Retrieve the link for the config id
    from deepcave.plugins.summary.configurations import Configurations

    link = Configurations.get_link(run, config_id)

    string = "<b>Configuration ID: "
    string += f"<a href='{link}' style='color: #ffffff'>{int(config_id)}</a></b><br><br>"

    # It's also nice to see the metrics
    objectives = run.get_objectives()
    if budget is None or budget == -1:
        highest_budget = run.get_highest_budget(config_id)
        assert highest_budget is not None
        string += f"<b>Objectives</b> (on highest found budget {round(highest_budget, 2)})<br>"
    else:
        string += f"<b>Objectives</b> (on budget {round(budget, 2)})<br>"

    try:
        avg_c, std_c = run.get_avg_costs(config_id, budget=budget)
        avg_costs: List[Optional[float]] = list(avg_c)
        std_costs: List[Optional[float]] = list(std_c)
    except ValueError:
        avg_costs = [None for _ in range(len(objectives))]
        std_costs = [None for _ in range(len(objectives))]

    for objective, cost, std_cost in zip(objectives, avg_costs, std_costs):
        if std_cost == 0.0:
            string += f"{objective.name}: {cost}<br>"
        else:
            string += f"{objective.name}: {cost} ± {std_cost}<br>"

    string += "<br><b>Hyperparameters</b>:<br>"

    config = run.get_config(config_id)
    for k, v in config.items():
        string += f"{k}: {v}<br>"

    return string


def generate_config_code(register: Callable, variables: List[str]) -> List[Component]:
    """
    Generate HTML components to display code.

    Parameters
    ----------
    register : Callable
        A Callable for registering Dash components.
        The register_input function is located in the Plugin class.
    variables : List[str]
        A List of variable names.

    Returns
    -------
    List[Component]
        A List of Dash components.
    """
    code = """
    from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

    # Create configspace
    cs = ConfigurationSpace.from_json({{path}})


    # Create config
    values = {{config_dict}}
    config = Configuration(cs, values=values)
    """

    components = []
    for line in code.splitlines():
        if len(line) == 0:
            components += [html.Br()]
            continue

        count_trailing_spaces = 0
        for char in line:
            if char == " ":
                count_trailing_spaces += 1
            else:
                break

        count_trailing_tabs = (count_trailing_spaces - 4) / 4
        trailing_style = {"margin-left": f"{count_trailing_tabs*2}em"}
        skip = False

        # Check if variable inside
        for variable in variables:
            match = re.search("{{(.+?)}}", line)
            if match:
                link = match.group(1)
                if link == variable:
                    components += [
                        # Add beginning
                        html.Code(line[: match.start()], style=trailing_style),
                        # Add variable
                        html.Code(id=register(variable, "children")),
                        # Add ending
                        html.Code(line[match.end() :]),
                        html.Br(),
                    ]

                    skip = True
                    break

        if skip:
            continue

        components += [
            html.Code(line, style=trailing_style),
            html.Br(),
        ]

    components = components[1 : len(components) - 1]
    return components
