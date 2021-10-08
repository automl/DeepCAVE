from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from deep_cave import app
from deep_cave.plugins.static_plugin import StaticPlugin
from deep_cave.plugins.dynamic_plugin import DynamicPlugin
from deep_cave.utils.logs import get_logger

from deep_cave.evaluators.fanova import fANOVA as _fANOVA


logger = get_logger(__name__)


class MFIS(DynamicPlugin):
    @staticmethod
    def id():
        return "mfis"

    @staticmethod
    def name():
        return "Multi-Fidelity Importance Shift"

    @staticmethod
    def position():
        return 3

    @staticmethod
    def category():
        return "Meta-Data Analysis"

    @staticmethod
    def debug():
        return False

    def get_input_layout(self):
        return [
            dbc.Label("Number of trees"),
            dbc.Input(id=self.register_input(
                "num_trees", "value"), type="number")
        ]

    def get_filter_layout(self):
        return [
        ]

    def load_inputs(self, run):
        return {
            "num_trees": {
                "value": 16
            },
        }

    def load_dependency_inputs(self, run, inputs):
        try:
            int(inputs["num_trees"]["value"])
        except:
            self.update_alert("Only numbers are allowed.", color="danger")
            inputs["num_trees"]["value"] = 16

        return inputs

    @staticmethod
    def process(run, inputs):
        fidelities = run.get_fidelities()
        hp_names = run.cs.get_hyperparameter_names()

        # Collect data
        data = {}
        for fidelity in fidelities:
            X, Y = run.get_encoded_data(fidelities=fidelity, for_tree=True)

            evaluator = _fANOVA(
                X, Y,
                configspace=run.cs,
                num_trees=int(inputs["num_trees"]["value"])
            )
            importance_dict = evaluator.quantify_importance(
                hp_names, depth=1, sorted=False)

            importance_dict = {k[0]: v for k, v in importance_dict.items()}

            data[fidelity] = importance_dict

        return data

    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph", "figure"))
        ]

    def load_outputs(self, filters, raw_outputs):
        # Collect data
        data = {}
        for fidelity_id, (fidelity, importance_dict) in enumerate(raw_outputs.items()):
            if fidelity_id not in filters["fidelities"]["value"]:
                continue

            x = []
            y = []
            error_y = []
            for hp_id, (hp_name, results) in enumerate(importance_dict.items()):
                if hp_id not in filters["hyperparameters"]["value"]:
                    continue

                x += [hp_name]
                y += [results[1]]
                error_y += [results[3]]

            data[fidelity] = (
                np.array(x),
                np.array(y),
                np.array(error_y)
            )

            if filters["sort"]["value"] == fidelity_id:
                selected_fidelity = fidelity

        # Sort by last fidelity now
        idx = np.argsort(data[selected_fidelity][1], axis=None)[::-1]

        bar_data = []
        for fidelity, values in data.items():
            bar_data += [go.Bar(
                name=fidelity,
                x=values[0][idx],
                y=values[1][idx],
                error_y_array=values[2][idx])
            ]

        fig = go.Figure(data=bar_data)
        fig.update_layout(barmode='group')

        return [fig]
