from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from src import app
from src.plugins.static_plugin import StaticPlugin
from src.plugins.dynamic_plugin import DynamicPlugin
from src.utils.logs import get_logger

from src.evaluators.fanova import fANOVA as _fANOVA


logger = get_logger(__name__)


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
