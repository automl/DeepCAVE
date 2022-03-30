from typing import List, Tuple, Any
import numpy as np
import plotly.express as px
import itertools
import plotly.graph_objs as go


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
