from typing import List, Union
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.cast import optional_int
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button


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
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Method"),
                            help_button(
                                "Local Parameter Importance: Quantify importance by changing the "
                                "neighborhood of a configuration. Uses default and incumbent configuration "
                                "as reference. \n\n"
                                "fANOVA: Quantify importance globally."
                            ),
                            dbc.Select(
                                id=register("method", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Trees"),
                            help_button(
                                "The number of trees of the internal random forest to estimate the "
                                "hyperparameter importance. "
                                "The more trees are used the more accurate the results. "
                                "However, also it takes longer to compute."
                            ),
                            dbc.Input(id=register("n_trees", type=optional_int)),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budgets"),
                    dbc.Checklist(id=register("budget_ids", ["value", "options"]), inline=True),
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
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "budget_ids": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        # Prepare objetives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_ids"]["value"]

        hp_names = run.configspace.get_hyperparameter_names()
        hp_options = get_checklist_options(hp_names)
        hp_value = inputs["hyperparameter_names"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]

        # Pre-selection of the hyperparameters
        if run is not None:
            if len(hp_value) == 0:
                hp_value = hp_names
            if len(budget_value) == 0:
                budget_value = [budget_ids[-1]]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_value,
            },
            "budget_ids": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_trees": {"value": inputs["n_trees"]["value"]},
        }

    @staticmethod
    def process(run: AbstractRun, inputs):
        objective = run.get_objective(inputs["objective_id"])
        method = inputs["method"]
        n_trees = inputs["n_trees"]

        if n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        hp_names = run.configspace.get_hyperparameter_names()
        budgets = run.get_budgets(include_combined=True)

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
            evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)

            importances = evaluator.get_importances(hp_names)
            data[budget_id] = importances

        return data

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"))

    @staticmethod
    def load_outputs(run, inputs, outputs):
        # First selected, should always be shown first
        selected_hp_names = inputs["hyperparameter_names"]
        selected_budget_ids = inputs["budget_ids"]

        if len(selected_hp_names) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            x = []
            y = []
            error_y = []
            for hp_name, results in importances.items():
                if hp_name not in selected_hp_names:
                    continue

                x += [hp_name]
                y += [results[0]]
                error_y += [results[1]]

            data[budget_id] = (np.array(x), np.array(y), np.array(error_y))

        # Sort by last fidelity now
        last_selected_budget_id = selected_budget_ids[0]
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

        return fig
