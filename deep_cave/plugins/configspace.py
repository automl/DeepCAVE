from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations_with_replacement

from deep_cave import app
from deep_cave.plugins.plugin import Plugin
from deep_cave.util.gui_helper import display_figure
from deep_cave.util.logs import get_logger
from deep_cave.util.styled_plot import plt

logger = get_logger(__name__)

'''
class ConfigSpace(Plugin):
    @staticmethod
    def id():
        return "configuration_space"

    @staticmethod
    def name():
        return "Configuration Space"

    @staticmethod
    def update_on_changes():
        return True

    def get_input_layout(self):
        return [
            dbc.Label("Fidelity"),
            dcc.Slider(id=self.register_input("fidelity", ["min", "max", "marks", "value"])),

            dbc.Label("Number of Configurations"),
            dcc.Slider(id=self.register_input("num_configs", ["min", "max", "marks", "value"])),
        ]
    
    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("heat_map", "figure"))
        ]

    def load_input(self, run):
        return {
            "fidelity": {
                "min": 0,
                "max": len(run.get_fidelities()) - 1,
                "marks": {str(i):str(np.round(float(fidelity), 2)) for i, fidelity in enumerate(run.get_fidelities())},
                "value": 0
            },
            "num_configs": {
                "min": 0,
                "max": int(run.get_fidelities()[0]),
                "marks": {str(i):str(i) for i in range(int(run.get_fidelities()[0]))},
                "value": 0,
            },
        }
    
    def load_dependency_input(self, run, **inputs):
        count = int(run.get_fidelities()[int(inputs["fidelity"]["value"])])
        value = int(inputs["num_configs"]["value"])

        return {
            "num_configs": {
                "max": count,
                "marks": {str(i):str(i) for i in range(count)},
                "value": value if value <= count else count,
            }
        }

    def load_output(self, run, **inputs):
        heat_map = px.imshow(
            np.random.rand(
                inputs["num_configs"]["value"],
                inputs["num_configs"]["value"],
            )
        )

        return {
            "heat_map": heat_map,
        }
'''