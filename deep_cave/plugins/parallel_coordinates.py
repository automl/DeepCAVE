from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from deep_cave import app
from deep_cave.plugins.plugin import Plugin
from deep_cave.util.gui_helper import display_figure
from deep_cave.util.logs import get_logger
from deep_cave.util.styled_plot import plt

logger = get_logger(__name__)


class ParallelCoordinates(Plugin):
    @staticmethod
    def id():
        return "parallel_coordinates"

    @staticmethod
    def name():
        return "Parallel Coordinates"

    @staticmethod
    def position():
        return 1

    @staticmethod
    def update_on_changes():
        return True

    def get_input_layout(self):
        return [
            dbc.Label("Fidelity"),
            dcc.Slider(id=self.register_input("fidelity", ["min", "max", "marks", "value"])),

            dbc.Label("Hyperparameters"),
            dbc.Checklist(id=self.register_input("hyperparameters", ["options", "value"]))
        ]
    
    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph", "figure"))
        ]

    def load_input(self, run):
        fidelities = [str(np.round(float(fidelity), 2)) for fidelity in run.get_fidelities()]
        fidelities = ["Mixed"] + fidelities

        hp_names = run.cs.get_hyperparameter_names()

        return {
            "fidelity": {
                "min": 0,
                "max": len(fidelities) - 1,
                "marks": {str(i):fidelity for i, fidelity in enumerate(fidelities)},
                "value": 0
            },
            "hyperparameters": {
                "options": [{"label": name, "value": idx} for idx, name in enumerate(hp_names)],
                "value": [idx for idx in range(len(hp_names))]
            }
        }

    def load_output(self, run, **inputs):
        fidelity_id = inputs["fidelity"]["value"] - 1
        selected_hp_ids = sorted(inputs["hyperparameters"]["value"])

        fidelity = None
        if fidelity_id >= 0:
            fidelity = run.get_fidelity(fidelity_id)

        X, y, mapping, _, _ = run.get_encoded_hyperparameters(fidelity, hp_ids=selected_hp_ids)

        data = {
            "cost": {
                "values": y,
                "label": "cost"
            }
        }

        for hp_name, values in X.items():
            data[hp_name] = {}
            data[hp_name]["label"] = hp_name
            data[hp_name]["values"] = values
            data[hp_name]["ticktext"] = list(mapping[hp_name].values())
            data[hp_name]["tickvals"] = list(mapping[hp_name].keys())

        fig = go.Figure(data=
            go.Parcoords(
                line = dict(
                    color = data["cost"]["values"],
                    showscale = True,
                    cmin = 0,
                    cmax = 1
                ),
                dimensions = list([d for d in data.values()])
            )
        )

        return {
            "graph": fig,
        }
