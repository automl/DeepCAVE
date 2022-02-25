from typing import Any, Dict, Type

from itertools import combinations_with_replacement

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html
from scipy.stats import spearmanr

from deepcave.plugins.plugin import Plugin
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plot import plt

logger = get_logger(__name__)

'''
class BudgetCorrelation(Plugin):
    @staticmethod
    def id():
        return "budget_correlation"

    @staticmethod
    def name():
        return "Budget Correlation"

    def _get_input_layout(self):
        return [
            dbc.Input(id=self.register_input("objective")),
            dbc.Select(id=self.register_input("fidelity", attributes=["value", "options"])),
        ]
    
    def _get_output_layout(self):
        return [
            html.Div(id=self.register_output("test", func=self.plot_figure)),
            dbc.Input(id=self.register_output("output")),
            dbc.Input(id=self.register_output("output2"))
        ]

    def plot_figure(self, data):
        fig = plt.figure()
        plt.plot([i for i in range(int(data["yay"]))])
        plt.ylabel(data["blub"])
        plt.xlabel(data["blub"])

        return [
            display_figure(fig)
        ]

    def _load_input(self, run, **inputs):
        return {
            "objective": {
                "value": "hi"
            },
            "fidelity": {
                "value": '2',
                "options": [{"label": i, "value": i} for i in range(5)]
            }
        }

    def _load_output(self, run, **inputs):
        """
        It's important to return only serializable data.
        Layouts should be specified within _get_output_layout.
        """

        objective = inputs["objective"]["value"]
        fidelity = inputs["fidelity"]["value"]

        return {
            "test": {"blub": objective, "yay": fidelity},
            "output": objective,
            "output2": fidelity
        }
'''
