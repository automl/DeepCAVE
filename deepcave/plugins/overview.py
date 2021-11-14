from typing import Dict, Type, Any

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from deepcave import app
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.logs import get_logger

from deepcave.evaluators.fanova import fANOVA as _fANOVA


logger = get_logger(__name__)


class Overview(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "overview"

    @staticmethod
    def name():
        return "Overview"

    @staticmethod
    def position():
        return 1

    @staticmethod
    def category():
        return "General"

    @staticmethod
    def activate_run_selection() -> bool:
        return True

    @staticmethod
    def process(run, inputs):
        return {}

    @staticmethod
    def get_output_layout(register):
        return [
            html.H3("Meta-Data"),
            html.Hr(),

            html.H3("Statistics"),
            html.Hr(),

            html.H3("Best Config"),
            html.Hr(),

            html.H3("Configuration Space"),
            html.Div(id=register("cs", "children")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):

        return [
            "hi"
        ]
