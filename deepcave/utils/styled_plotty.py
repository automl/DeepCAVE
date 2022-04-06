from typing import List, Optional, Tuple, Any, Union
import numpy as np
import plotly.express as px
import itertools
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
)


def hex_to_rgb(hex_string: str) -> Tuple[int, int, int]:
    """
    Converts a hex_string to a tuple of rgb values.
    Requires format including #, e.g.:
    #000000
    #ff00ff
    """
    if len(hex_string) != 7:
        raise ValueError(f"Invalid length for #{hex_string}")

    if any(c not in "0123456789ABCDEF" for c in hex_string.lstrip("#").upper()):
        raise ValueError(f"Invalid character in #{hex_string}")

    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)


def get_color(id_: int, alpha: float = 1) -> str:
    """
    Currently (Plotly version 5.3.1) there are 10 possible colors.
    """
    color = px.colors.qualitative.Plotly[id_]

    r, g, b = hex_to_rgb(color)

    return f"rgba({r}, {g}, {b}, {alpha})"


def get_discrete_heatmap(x, y, values: List[List[Any]], labels: List[List[Any]]):
    """
    Generate a discrete colorscale from a (nested) list or numpy array of values.

    Parameters
    ----------
    values : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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

    # Now we give them new ids and we want to create new z values
    # For that we need a mapping from old to new
    mapping = {}
    v = []
    for new, old in enumerate(unique_sorted_values):
        mapping[old] = new
        v += [new]

    z = values
    for i1, v1 in enumerate(values):
        for i2, v2 in enumerate(v1):
            z[i1][i2] = mapping[v2]

    n_intervals = v + [len(v)]
    n_intervals = [(i - n_intervals[0]) / (n_intervals[-1] - n_intervals[0]) for i in n_intervals]
    colors = [get_color(i) for i in range(len(n_intervals))]

    discrete_colorscale = []
    for k in range(len(v)):
        discrete_colorscale.extend([[n_intervals[k], colors[k]], [n_intervals[k + 1], colors[k]]])

    tickvals = [np.mean(n_intervals[k : k + 2]) for k in range(len(n_intervals) - 1)]
    ticktext = unique_sorted_labels

    return go.Heatmap(
        x=x,
        y=y,
        z=z,
        showscale=True,
        colorscale=discrete_colorscale,
        colorbar={"tickvals": tickvals, "ticktext": ticktext, "tickmode": "array"},
        zmin=0,
        zmax=1,
        hoverinfo="skip",
    )


def prettify_label(label: Union[str, float, int]) -> str:
    """
    Takes a label and prettifies it. E.g. floats are shortened.

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


def get_tick_data(
    hp: Hyperparameter,
    additional_values: Optional[List] = None,
    ticks: int = 6,
    include_nan: bool = True,
) -> Tuple[List, List]:
    """
    Generates tick data for both tickvals and ticktext. The background is that
    you might have encoded data but you don't want to show all of them.
    With this function, only 6 (default) values are shown. This behaviour is
    ignored if `hp` is categorical.

    Parameters
    ----------
    hp : Hyperparameter
        Hyperparameter to generate ticks from.
    additional_values : Optional[List], optional
        Additional values, which are forced in addition. By default None.
    ticks : int, optional
        Number of ticks, by default 6
    include_nan : bool, optional
        Whether "nan" as tick should be included. By default True.

    Returns
    -------
    Tuple[List, List]
        tickvals and ticktext.
    """

    # This is basically the inverse of `encode_config`.
    if isinstance(hp, CategoricalHyperparameter):
        tickvals = [hp._inverse_transform(choice) / (len(hp.choices) - 1) for choice in hp.choices]
        ticktext = hp.choices
    elif isinstance(hp, Constant):
        tickvals = [1.0]
        ticktext = [hp.value]
    else:
        min_v = 0
        max_v = 1

        values = [min_v]

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
            inverse_values += [hp._transform_scalar(value)]

        # Integers are rounded, so we map then back
        if isinstance(hp, IntegerHyperparameter):
            for label in inverse_values:
                value = hp._inverse_transform(label)

                if value not in tickvals:
                    tickvals += [value]
                    ticktext += [label]

            if additional_values is not None:
                # Now we add additional values
                for value in additional_values:
                    if not (value is None or np.isnan(value) or value == -0.2):
                        label = hp._transform_scalar(value)
                        value = hp._inverse_transform(label)

                        if value not in tickvals:
                            tickvals += [value]
                            ticktext += [label]
        else:
            for value, label in zip(values, inverse_values):
                tickvals += [value]
                ticktext += [label]

            if additional_values is not None:
                # Now we add additional values
                for value in additional_values:
                    if (
                        not (value is None or np.isnan(value) or value == -0.2)
                        and value not in tickvals
                    ):
                        tickvals += [value]
                        ticktext += [hp._transform_scalar(value)]

    ticktext = [prettify_label(label) for label in ticktext]

    if include_nan:
        tickvals += [-0.2]
        ticktext += ["NaN"]

    return tickvals, ticktext


def get_tick_data_from_values(
    values: List, labels: List, forced: Optional[List[bool]] = None, ticks: int = 6
) -> Tuple[List, List]:
    """
    Generates tick data for both values and labels. The background is that
    you might have encoded data but you don't want to show all of them.
    With this function, only 6 (default) values are shown. This behaviour is
    ignored if `values` is a list of strings.

    Parameters
    ----------
    values : List
        List of values.
    labels : List
        List of labels. Must be the same size as `values`.
    forced : List[bool], optional
        List of booleans. If True, displaying the particular tick is enforced.
        Independent from `ticks`.
    ticks : int, optional
        Number of ticks and labels to show. By default 6.

    Returns
    -------
    Tuple[List, List]
        Returns tickvals and ticktext as list.
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

    # If we have less than x values, we also show them
    if return_all or len(unique_values) <= ticks:
        # Make sure we don't have multiple (same) labels for the same value
        for value, label in zip(unique_values, unique_labels):
            tickvals.append(value)
            ticktext.append(label)
    else:
        # Add min+max values
        for idx in [np.argmin(values), np.argmax(values)]:
            tickvals.append(values[idx])
            ticktext.append(labels[idx])

        # After we added min and max values, we want to add
        # intermediate values too
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
