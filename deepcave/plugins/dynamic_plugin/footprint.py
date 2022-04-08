from typing import Any, Dict, List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.layout import get_radio_options, get_select_options
from deepcave.utils.styled_plotty import get_color
from deepcave.evaluators.footprint import Footprint as Evaluator


class FootPrint(DynamicPlugin):
    id = "footprint"
    name = "Configuration Footprint"
    icon = "fas fa-shoe-prints"
    activate_run_selection = True
    use_cache = False

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective", ["options", "value"]),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budget"),
                    dbc.Select(
                        id=register("budget", ["options", "value"]),
                        placeholder="Select budget ...",
                    ),
                ],
                className="",
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"options": get_select_options(), "value": None},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        # Prepare objetives
        objective_names = selected_run.get_objective_names()
        objective_options = get_select_options(objective_names)
        objective_value = inputs["objective"]["value"]

        # Prepare budgets
        budgets = selected_run.get_budgets(human=True)
        budget_options = get_select_options(budgets, range(len(budgets)))

        # Pre-set values
        if objective_value is None:
            objective_value = objective_names[0]
            budget_value = budget_options[-1]["value"]
        else:
            budget_value = inputs["budget"]["value"]

        return {
            "objective": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget": {
                "options": budget_options,
                "value": budget_value,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:
        budget = run.get_budget(inputs["budget"]["value"])
        objective = run.get_objective(inputs["objective"]["value"])

        # Initialize the evaluator
        evaluator = Evaluator(run)
        z = evaluator.calculate(objective, budget)

        return {"z": z}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),
        ]

    def load_outputs(self, inputs, outputs, runs):
        data = go.Heatmap(z=outputs["z"], zsmooth="best")

        layout = go.Layout(
            xaxis=dict(title="MDS X-Axis", tickvals=[]),
            yaxis=dict(title="MDS Y-Axis", tickvals=[]),
        )

        return [go.Figure(data=data, layout=layout)]
