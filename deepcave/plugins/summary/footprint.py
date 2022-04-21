from typing import Any, Dict, List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.static import StaticPlugin
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_radio_options, get_select_options, get_slider_marks
from deepcave.utils.styled_plotty import get_color, get_hovertext_from_config
from deepcave.evaluators.footprint import Footprint as Evaluator


class FootPrint(StaticPlugin):
    id = "footprint"
    name = "Configuration Footprint"
    icon = "fas fa-shoe-prints"
    description = """
        The configuration footprint shows the configuration space in two dimensions.
        Based on the evaluated configurations, a surface is plotted. 
        Additional border and support configurations should answer the question whether the search
        is exhausted or not. For each hyperparameter, ten random support configurations are plotted.
    """
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register):
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective", ["options", "value"]),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            dbc.Select(
                                id=register("budget", ["options", "value"]),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Details"),
                    dcc.Slider(
                        id=register("details", "value"),
                        min=0.1,
                        max=0.9,
                        step=0.4,
                        marks={0.1: "Low", 0.5: "Medium", 0.9: "High"},
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Show Border Configurations"),
                            dbc.Select(
                                id=register("show_borders", ["options", "value"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Support Configurations"),
                            dbc.Select(
                                id=register("show_supports", ["options", "value"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        )

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"options": get_select_options(), "value": None},
            "details": {"value": 0.5},
            "show_borders": {"options": get_select_options(binary=True), "value": "true"},
            "show_supports": {"options": get_select_options(binary=True), "value": "true"},
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

        new_inputs = {
            "objective": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget": {
                "options": budget_options,
                "value": budget_value,
            },
        }

        update_dict(inputs, new_inputs)
        return inputs

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:
        budget = run.get_budget(inputs["budget"]["value"])
        objective = run.get_objective(inputs["objective"]["value"])
        details = 1 - inputs["details"]["value"]

        # Initialize the evaluator
        evaluator = Evaluator(run)
        evaluator.calculate(objective, budget)

        data = evaluator.get_surface(details=details)
        config_points = evaluator.get_points("configs")
        border_points = evaluator.get_points("borders")
        support_points = evaluator.get_points("supports")
        incumbent_points = evaluator.get_points("incumbents")

        return {
            "data": data,
            "config_points": config_points,
            "border_points": border_points,
            "support_points": support_points,
            "incumbent_points": incumbent_points,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),
        ]

    def load_outputs(self, inputs, outputs, run):
        show_borders = inputs["show_borders"]["value"] == "true"
        show_supports = inputs["show_supports"]["value"] == "true"

        traces = []
        x_, y_, z_ = outputs["data"]

        # First add the Heatmap
        data = go.Heatmap(
            x=x_,
            y=y_,
            z=z_,
            zsmooth="best",
            hoverinfo="skip",
            colorbar=dict(
                len=0.5,
                title=inputs["objective"]["value"],
            ),
            colorscale="viridis",
        )
        traces += [data]

        point_names = ["Configuration", "Incumbent"]
        point_values = ["config_points", "incumbent_points"]

        if show_borders:
            point_names += ["Border Configuration"]
            point_values += ["border_points"]
        if show_supports:
            point_names += ["Random (unevaluated) Configuration"]
            point_values += ["support_points"]

        # Now add the points
        for name, points in zip(point_names, point_values):
            x, y, config_ids = outputs[points]
            size = 5
            marker_symbol = "x"
            if points == "incumbent_points":
                size = 10
                marker_symbol = "triangle-up"
            traces += [
                go.Scatter(
                    name=name,
                    x=x,
                    y=y,
                    mode="markers",
                    marker_symbol=marker_symbol,
                    marker={"size": size},
                    hovertext=[
                        get_hovertext_from_config(run, config_id) for config_id in config_ids
                    ],
                    hoverinfo="text",
                )
            ]

        layout = go.Layout(
            xaxis=dict(title="MDS X-Axis", tickvals=[]),
            yaxis=dict(title="MDS Y-Axis", tickvals=[]),
        )

        return [go.Figure(data=traces, layout=layout)]
