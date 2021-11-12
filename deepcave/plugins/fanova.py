from typing import Dict, Type, Any

from dash import dcc
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
class fANOVA(DynamicPlugin):
    @staticmethod
    def id():
        return "fanova"

    @staticmethod
    def name():
        return "fANOVA"

    @staticmethod
    def position():
        return 100

    @staticmethod
    def category():
        return "Hyperparameter Analysis"

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
            dbc.FormGroup([
                dbc.Label("Hyperparameters"),
                dbc.Checklist(id=self.register_input(
                    "hyperparameters", ["options", "value"], filter=True)),

            ]),
            dbc.FormGroup([
                dbc.Label("Fidelities"),
                dbc.Checklist(id=self.register_input(
                    "fidelities", ["options", "value"], filter=True)),
            ]),
            dbc.FormGroup([
                dbc.Label("Sort by"),
                dbc.RadioItems(id=self.register_input(
                    "sort", ["options", "value"], filter=True))
            ])
        ]

    def load_inputs(self, run):
        hp_names = run.configspace.get_hyperparameter_names()
        fidelities = run.get_budgets()

        return {
            "num_trees": {
                "value": 16
            },
            "hyperparameters": {
                "options": [{"label": name, "value": idx} for idx, name in enumerate(hp_names)],
                "value": [i for i in range(len(hp_names))]
            },
            "fidelities": {
                "options": [{"label": name, "value": idx} for idx, name in enumerate(fidelities)],
                "value": [i for i in range(len(fidelities))]
            },
            "sort": {
                "options": [{"label": name, "value": idx} for idx, name in enumerate(fidelities)],
                "value": len(fidelities) - 1
            }
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
        hp_names = run.configspace.get_hyperparameter_names()
        fidelities = run.get_budgets()

        # Collect data
        data = {}
        for fidelity in fidelities:
            X, Y = run.get_encoded_configs(budget=fidelity, for_tree=True)

            evaluator = _fANOVA(
                X, Y,
                configspace=run.configspace,
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
"""
