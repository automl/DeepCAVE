#  noqa: D400
"""
# HyperSHAP

## Classes
    - HyperSHAP:
"""
from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_checklist_options


class Tunability(StaticPlugin):
    """Provide a Plugin for the Tunability analysis of HyperSHAP."""

    id = "hypershap"
    name = "HyperSHAP"
    icon = "fas fa-binoculars"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List:
        """Define the input block of the plugin."""
        return [
            html.Div(
                [
                    dbc.Label("Tunability"),
                    dbc.Select(
                        id=register("tunability", ["value", "options"], type=int),
                        placeholder="Select tunabability ...",
                    ),
                ],
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'."""
        return {
            "tunability": {"options": get_checklist_options()},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process your input data and return raw data to be used in the output layout."""
        return {}

    @staticmethod
    def get_output_layout(register: Callable) -> Any:
        """Define the output block of the plugin."""
        return [
            dcc.Graph(
                register("perf_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            )
        ]

    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:  # type: ignore
        """Read the raw data and prepare it for the layout."""
        return go.Figure()
