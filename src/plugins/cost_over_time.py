from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from src import app
from src.plugins.dynamic_plugin import DynamicPlugin
from src.plugins.static_plugin import StaticPlugin
from src.utils.logs import get_logger

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

    @staticmethod
    def get_input_layout(register):
        return [
            dbc.Label("Fidelity"),
            dcc.Slider(id=register(
                "fidelity", ["min", "max", "marks", "value"])),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            dbc.FormGroup([
                dbc.Label("Logarithmic"),
                dbc.RadioItems(id=register("log", ["options", "value"]))
            ])
        ]

    @staticmethod
    def load_inputs(runs):
        run = list(runs.values())[0]

        fidelities = [
            str(np.round(float(fidelity), 2)) for fidelity in run.get_budgets()]

        return {
            "fidelity": {
                "min": 0,
                "max": len(fidelities) - 1,
                "marks": {str(i): fidelity for i, fidelity in enumerate(fidelities)},
                "value": 0
            },
            "log": {
                "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                "value": 0
            }
        }

    @staticmethod
    def process(run, inputs):
        fidelity_id = inputs["fidelity"]["value"]

        fidelity = None
        if fidelity_id is not None and fidelity_id >= 0:
            fidelity = run.get_budget(fidelity_id)

        costs, times = run.get_trajectory(fidelity)

        import time
        time.sleep(20)

        return {
            "times": times,
            "costs": costs,
            # "hovertext": additional
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        traces = []
        for run_name, run_outputs in outputs.items():

            traces.append(go.Scatter(
                x=run_outputs["times"],
                y=run_outputs["costs"],
                name=run_name,
                line_shape='hv',
                # hovertext=outputs["additional"]
            ))

        type = None
        if inputs["log"]["value"] == 1:
            type = 'log'

        layout = go.Layout(
            xaxis=dict(
                title='Wallclock time [s]',
                type=type
            ),
            yaxis=dict(
                title='Cost',
            ),
        )

        fig = go.Figure(data=traces, layout=layout)

        graphs = []
        for group_name, _ in groups.items():
            graphs.append(fig)
            return graphs

        # return []

        return [fig]

    # def get_mpl_output_layout(self):
    #    return [
    #        dbc.Input(id=self.register_output("blub", "value", mpl=True))
    #    ]

    # def load_mpl_outputs(self, inputs, outputs):
    #    return [inputs["filter"]["value"]]
