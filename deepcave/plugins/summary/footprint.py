from typing import Any, Dict

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.static import StaticPlugin
from deepcave.utils.layout import (
    get_select_options,
    render_mpl_figure,
)
from deepcave.utils.styled_plotty import get_color, get_hovertext_from_config
from deepcave.evaluators.footprint import Footprint as Evaluator
from deepcave.utils.styled_plot import plt
from deepcave.utils.layout import help_button


class FootPrint(StaticPlugin):
    id = "footprint"
    name = "Configuration Footprint"
    icon = "fas fa-shoe-prints"
    help = "docs/plugins/budget_correlation.rst"
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
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
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
                    help_button(
                        "The details parameter controls the resolution of the surface plot."
                    ),
                    dcc.Slider(
                        id=register("details", "value", type=float),
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
                                id=register("show_borders", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Support Configurations"),
                            dbc.Select(
                                id=register("show_supports", ["value", "options"]),
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
            "details": {"value": 0.5},
            "show_borders": {"options": get_select_options(binary=True), "value": "true"},
            "show_supports": {"options": get_select_options(binary=True), "value": "true"},
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
        budget_options = get_select_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
        else:
            budget_value = inputs["budget_id"]["value"]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])
        details = inputs["details"]

        # Initialize the evaluator
        evaluator = Evaluator(run)
        evaluator.calculate(objective, budget)

        performance_data = evaluator.get_surface(details=details, performance=True)
        area_data = evaluator.get_surface(details=details, performance=False)
        config_points = evaluator.get_points("configs")
        border_points = evaluator.get_points("borders")
        support_points = evaluator.get_points("supports")
        incumbent_points = evaluator.get_points("incumbents")

        return {
            "performance_data": performance_data,
            "area_data": area_data,
            "config_points": config_points,
            "border_points": border_points,
            "support_points": support_points,
            "incumbent_points": incumbent_points,
        }

    @staticmethod
    def get_output_layout(register):
        return dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Graph(id=register("performance", "figure"), style={"height": "50vh"}),
                    label="Performance",
                ),
                dbc.Tab(
                    dcc.Graph(id=register("area", "figure"), style={"height": "50vh"}),
                    label="Coverage",
                    style={"height": "50vh"},
                ),
            ]
        )

    @staticmethod
    def get_mpl_output_layout(register):
        return html.Div(
            [
                html.Img(
                    id=register("performance", "src"),
                    className="img-fluid",
                ),
                html.Img(
                    id=register("area", "src"),
                    className="img-fluid",
                ),
            ]
        )

    @staticmethod
    def load_outputs(run, inputs, outputs):
        objective = run.get_objective(inputs["objective_id"])
        show_borders = inputs["show_borders"] == "true"
        show_supports = inputs["show_supports"] == "true"

        traces = []

        # First add the Heatmap
        performance_data = go.Heatmap(
            x=outputs["performance_data"][0],
            y=outputs["performance_data"][1],
            z=outputs["performance_data"][2],
            zsmooth="best",
            hoverinfo="skip",
            colorbar=dict(
                len=0.5,
                title=objective["name"],
            ),
            colorscale="blues",
        )

        area_data = go.Heatmap(
            x=outputs["area_data"][0],
            y=outputs["area_data"][1],
            z=outputs["area_data"][2],
            zsmooth="best",
            hoverinfo="skip",
            colorbar=dict(
                len=0.5,
                title="Valid Configspace Area",
            ),
            colorscale="blues",
        )

        point_names = []
        point_values = []
        point_color_ids = []

        if show_borders:
            point_names += ["Border Configuration"]
            point_values += ["border_points"]
            point_color_ids += [2]
        if show_supports:
            point_names += ["Random (unevaluated) Configuration"]
            point_values += ["support_points"]
            point_color_ids += [3]

        point_names += ["Configuration", "Incumbent"]
        point_values += ["config_points", "incumbent_points"]
        point_color_ids += [4, 1]

        # Now add the points
        for name, points, color_id in zip(point_names, point_values, point_color_ids):
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
                    marker={"size": size, "color": get_color(color_id)},
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

        return [
            go.Figure(data=[performance_data] + traces, layout=layout),
            go.Figure(data=[area_data] + traces, layout=layout),
        ]

    @staticmethod
    def load_mpl_outputs(run, inputs, outputs):
        objective = run.get_objective(inputs["objective_id"])
        show_borders = inputs["show_borders"] == "true"
        show_supports = inputs["show_supports"] == "true"

        images = []
        for data in ["performance", "area"]:
            x_, y_, z_ = outputs[data + "_data"]

            plt.figure(1, 1, 200)
            plt.grid(False)
            plt.contourf(x_, y_, z_, cmap="viridis")
            cb = plt.colorbar()

            if data == "performance":
                cb.ax.set_title(objective["name"])
            else:
                cb.ax.set_title("Valid Area")

            point_names = ["Configuration", "Incumbent"]
            point_values = ["config_points", "incumbent_points"]

            if show_borders:
                point_names += ["Border Configuration"]
                point_values += ["border_points"]
            if show_supports:
                point_names += ["Random Configuration"]
                point_values += ["support_points"]

            # Now add the points
            for id, (name, points) in enumerate(zip(point_names, point_values)):
                x, y, config_ids = outputs[points]
                size = 5
                marker_symbol = "X"
                if points == "incumbent_points":
                    size = 10
                    marker_symbol = "^"

                color = get_color(id, mpl=True)
                plt.scatter(x, y, marker=marker_symbol, s=size, label=name, c=color)

            plt.axis("off")
            plt.xlabel("MDS X-Axis")
            plt.ylabel("MDS Y-Axis")
            plt.legend()

            images += [render_mpl_figure(plt)]

        return images
