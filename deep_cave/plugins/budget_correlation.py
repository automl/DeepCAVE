from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import plotly.express as px
from dash.development.base_component import Component
import dash_table

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations_with_replacement

from deep_cave.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger

logger = get_logger(__name__)


class BudgetCorrelation(Plugin):
    @staticmethod
    def id():
        return "budget_correlation"

    @staticmethod
    def name():
        return "Budget Correlation"

    def _get_input_layout(self):
        return [
            dbc.Input(id=self.register_input("objective", "value")),
            dbc.Input(id=self.register_input("fidelity", "value")),
        ]
    
    def _get_output_layout(self):
        return [
            dbc.Input(id=self.register_output("output", "value")),
            dbc.Input(id=self.register_output("output2", "value"))
        ]

    def _process(self, *args):
        objective, fidelity = args

        return [objective, fidelity]