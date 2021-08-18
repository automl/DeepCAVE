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
from deep_cave.plugins.dynamic_plugin import DynamicPlugin
from deep_cave.utils.logs import get_logger

logger = get_logger(__name__)


class SplitAnalysis(DynamicPlugin):
    @staticmethod
    def id():
        return "split_analysis"

    @staticmethod
    def name():
        return "Split Analysis"

    @staticmethod
    def description():
        return "Uses all the runs to determine the bias using different dataset splits."

    @staticmethod
    def position():
        return 0

    @staticmethod
    def category():
        return "Dataset"

    def get_input_layout(self):
        return []

    def get_filter_layout(self):
        return []

    def load_inputs(self, run):
        return {}

    @staticmethod
    def process(run, **inputs):
        return {}

    def get_output_layout(self):
        return [
            html.Div(id=self.register_output("output", "children"))
        ]

    def load_outputs(self, filters, outputs):
        return "test"
