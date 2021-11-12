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

"""
class Overview(DynamicPlugin):
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
        return "Meta-Data Analysis"

    @staticmethod
    def description():
        return "especially lists all found configurations and which fidelities were used."

    @staticmethod
    def debug():
        return False

    def get_input_layout(self):
        return []

    def get_filter_layout(self):
        return []

    @staticmethod
    def process(run, inputs):
        return None

    def get_output_layout(self):
        return [
            html.Div(self.register_output("blub", "children"))
        ]

    def load_outputs(self, filters, raw_outputs):

        return ["hi"]
"""
