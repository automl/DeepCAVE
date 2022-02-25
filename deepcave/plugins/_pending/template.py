import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color

logger = get_logger(__name__)

"""
class Template(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "template"

    @staticmethod
    def name():
        return "Template"

    @staticmethod
    def position():
        return 1

    @staticmethod
    def category():
        return "Examples"

    @staticmethod
    def get_input_layout(register):
        return [
            dbc.Label("Test input"),
            dbc.Input(id=register("test-input", "value"))
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            dbc.Label("Test radio items"),
            dbc.RadioItems(id=register("test-radio", ["options", "value"]))
        ]

    @staticmethod
    def load_inputs(runs):
        return {
            "test-radio": {
                "options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}],
                "value": True
            },
        }

    @staticmethod
    def process(run, inputs):
        return {
            "x": [1, 2, 3],
            "y": [4, 2, 6]
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dbc.Input(id=register("test-output", "value")),
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        traces = []

        for run_name, run_outputs in outputs.items():
            trace = go.Scatter(
                x=run_outputs["x"],
                y=run_outputs["y"],
                name=run_name,
            )

            traces.append(trace)

        return [
            inputs["test-input"]["value"],
            go.Figure(data=traces)
        ]
"""
