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
from deep_cave.plugins.static_plugin import StaticPlugin
from deep_cave.utils.logs import get_logger

logger = get_logger(__name__)


class CostOverTime(StaticPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "cost_over_time"

    @staticmethod
    def name():
        return "Cost Over Time"

    @staticmethod
    def position():
        return 5

    @staticmethod
    def category():
        return "Performance Analysis"

    def get_input_layout(self):
        return [
            dbc.Label("Fidelity"),
            dcc.Slider(id=self.register_input(
                "fidelity", ["min", "max", "marks", "value"])),
        ]

    def get_filter_layout(self):
        return [
            dbc.Label("filter"),
            dbc.Input(id=self.register_input("filter", "value", filter=True))
        ]

    def load_inputs(self, run):
        fidelities = [str(np.round(float(fidelity), 2))
                      for fidelity in run.get_fidelities()]
        fidelities = ["Mixed"] + fidelities

        return {
            "fidelity": {
                "min": 0,
                "max": len(fidelities) - 1,
                "marks": {str(i): fidelity for i, fidelity in enumerate(fidelities)},
                "value": 0
            },
        }

    @staticmethod
    def blub2():
        print("YAY")
        import time
        time.sleep(5)
        print("... and finished")
        return {"blub": 234}

    @staticmethod
    def process(run, params):
        import time
        time.sleep(20)

        fidelity_id = params["fidelity"]["value"] - 1

        fidelity = None
        if fidelity_id >= 0:
            fidelity = run.get_fidelity(fidelity_id)

        wallclock_times, costs, additional = run.get_trajectory(fidelity)

        return {
            "wallclock_times": wallclock_times,
            "costs": costs,
            # "hovertext": additional
        }

    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph", "figure")),
            dbc.Input(id=self.register_output("blub", "value"))
        ]

    def load_outputs(self, filters, raw_outputs):
        trace = go.Scatter(
            x=raw_outputs["wallclock_times"],
            y=raw_outputs["costs"],
            name="hv",
            line_shape='hv',
            # hovertext=outputs["additional"]
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

        return [fig, filters["filter"]["value"]]

    def get_mpl_output_layout(self):
        return [
            dbc.Input(id=self.register_output("blub", "value", mpl=True))
        ]

    def load_mpl_outputs(self, filters, raw_outputs):
        return [filters["filter"]["value"]]
