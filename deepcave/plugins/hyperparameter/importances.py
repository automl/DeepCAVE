from typing import List
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options


class Importances(StaticPlugin):
    id = "importances"
    name = "Importances"
    icon = "far fa-star"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Method"),
                    dbc.Select(
                        id=register("method", ["options", "value"]), placeholder="Select ..."
                    ),
                ],
                className="mb-3",
            ),
            html.Div([dbc.Label("Trees"), dbc.Input(id=register("n_trees", "value"))]),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameters", ["options", "value"]), inline=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budgets"),
                    dbc.Checklist(id=register("budgets", ["options", "value"]), inline=True),
                ]
            ),
        ]

    def load_inputs(self):
        method_labels = ["Local Parameter Importance (local)", "fANOVA (global)"]
        method_values = ["local", "global"]

        return {
            "method": {
                "options": get_select_options(method_labels, method_values),
                "value": "local",
            },
            "n_trees": {"value": 10},
            "budgets": {"options": get_checklist_options(), "value": []},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        budgets = selected_run.get_budgets(human=True)
        budget_ids = list(range(len(budgets)))
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budgets"]["value"]

        hp_names = selected_run.configspace.get_hyperparameter_names()
        hp_options = get_checklist_options(hp_names)
        hp_value = inputs["hyperparameters"]["value"]

        # Pre-selection of the hyperparameters
        if selected_run is not None:
            if len(hp_value) == 0:
                hp_value = hp_names
            if len(budget_value) == 0:
                budget_value = [budget_ids[-1]]

        new_inputs = {
            "hyperparameters": {
                "options": hp_options,
                "value": hp_value,
            },
            "budgets": {
                "options": budget_options,
                "value": budget_value,
            },
        }

        update_dict(inputs, new_inputs)
        return inputs

    @staticmethod
    def process(run: AbstractRun, inputs):
        hp_names = run.configspace.get_hyperparameter_names()
        budgets = run.get_budgets()
        method = inputs["method"]["value"]
        n_trees = int(inputs["n_trees"]["value"])

        if method == "local":
            # Intiatize the evaluator
            evaluator = LocalEvaluator(run)
        elif method == "global":
            evaluator = GlobalEvaluator(run)
        else:
            raise RuntimeError("Method was not found.")

        # Collect data
        data = {}
        for budget_id, budget in enumerate(budgets):
            evaluator.calculate(budget, n_trees=n_trees, seed=0)

            importances = evaluator.get_importances(hp_names)
            data[budget_id] = importances

        return data

    @staticmethod
    def get_output_layout(register):
        return [dcc.Graph(register("graph", "figure"))]

    def load_outputs(self, inputs, outputs, run):
        # First selected, should always be shown first
        selected_hyperparameters = inputs["hyperparameters"]["value"]
        selected_budget_ids = inputs["budgets"]["value"]

        if len(selected_hyperparameters) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances in outputs.items():
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            x = []
            y = []
            error_y = []
            for hp_name, results in importances.items():
                if hp_name not in inputs["hyperparameters"]["value"]:
                    continue

                x += [hp_name]
                y += [results[0]]
                error_y += [results[1]]

            data[budget_id] = (np.array(x), np.array(y), np.array(error_y))

        # Sort by last fidelity now
        last_selected_budget_id = selected_budget_ids[-1]
        idx = np.argsort(data[last_selected_budget_id][1], axis=None)[::-1]

        bar_data = []
        for budget_id, values in data.items():
            budget = run.get_budget(budget_id)
            bar_data += [
                go.Bar(
                    name=budget,
                    x=values[0][idx],
                    y=values[1][idx],
                    error_y_array=values[2][idx],
                )
            ]

        fig = go.Figure(data=bar_data)
        fig.update_layout(
            barmode="group",
            yaxis_title="Importance",
            legend={"title": "Budget"},
        )

        return [fig]
