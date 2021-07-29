from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from deep_cave import app
from deep_cave.plugins.plugin import Plugin
from deep_cave.util.gui_helper import display_figure
from deep_cave.util.logs import get_logger
from deep_cave.util.styled_plot import plt

logger = get_logger(__name__)


class PerformanceOverTime(Plugin):
    @staticmethod
    def id():
        return "performance_over_time"

    @staticmethod
    def name():
        return "Performance Over Time"

    @staticmethod
    def position():
        return 0

    @staticmethod
    def update_on_changes():
        return True

    def get_input_layout(self):
        return [
            dbc.Label("Fidelity"),
            dcc.Slider(id=self.register_input("fidelity", ["min", "max", "marks", "value"])),
        ]
    
    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph", "figure"))
        ]

    def load_input(self, run):
        fidelities = [str(np.round(float(fidelity), 2)) for fidelity in run.get_fidelities()]
        fidelities = ["Mixed"] + fidelities

        return {
            "fidelity": {
                "min": 0,
                "max": len(fidelities) - 1,
                "marks": {str(i):fidelity for i, fidelity in enumerate(fidelities)},
                "value": 0
            },
        }

    def load_output(self, run, **inputs):
        fidelity_id = inputs["fidelity"]["value"] - 1

        fidelity = None
        if fidelity_id >= 0:
            fidelity = run.get_fidelity(fidelity_id)

        wallclock_times, costs, additional = run.get_trajectory(fidelity)

        trace = go.Scatter(
            x=wallclock_times,
            y=costs,
            name="hv", 
            line_shape='hv',
            hovertext=additional
        )

        layout = go.Layout(
            xaxis=dict(
                title='Wallclock time [s]',
                type='log'
            ),
            yaxis=dict(
                title='Cost',
            ),
        )

        fig = go.Figure(data=[trace], layout=layout)

        return {
            "graph": fig,
        }
