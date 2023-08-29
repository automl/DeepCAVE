#  noqa: D400
"""
# CostOverTime

This module provides utilities for visualizing the cost over time.
It includes a plugin class.

## Contents
    - check_runs_compatibility: Check if the runs are elligable for comparing.
    - check_equality: Check if the runs are equal.
    - get_input_layout: Get the layout of the input.
    - get_filter_layout: Get filtered layout for input.
    - load_inputs: Load the inputs.
    - process: Process the run and the inputs.
    - get_output_layout: Get the output layout.
    - load_outputs: Create the figure and safe the image.
"""

from typing import List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import (
    get_color,
    get_hovertext_from_config,
    save_image,
)


class CostOverTime(DynamicPlugin):
    """
    A plugin to provide a visualization for the cost over time.

    Methods
    -------
    check_runs_compatibility
        Check if the runs are elligable for comparing.
    check_equality
        Check if the runs are equal.
    get_input_layout
        Get the layout of the input.
    get_filter_layout
        Get filtered layout for input.
    load_inputs
        Load the inputs.
    process
        Process the run and the inputs.
    get_output_layout
        Get the output layout.
    load_outputs
        Create the figure and safe the image.

    Attributes
    ----------
    id
        The identificator of the plugin.
    name
        The name of the plugin.
    icon
        The icon representing the plugin.
    help
        The path to the documentation of the plugin.
    """

    id = "cost_over_time"
    name = "Cost Over Time"
    icon = "fas fa-chart-line"
    help = "docs/plugins/cost_over_time.rst"

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:  # noqa: D102
        check_equality(runs, objectives=True, budgets=True)

        # Set some attributes here
        run = runs[0]

        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        self.objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        self.budget_options = get_select_options(budgets, budget_ids)

    @staticmethod
    def get_input_layout(register):  # noqa: D102
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
                            help_button(
                                "Combined budget means that the trial on the highest evaluated "
                                "budget is used."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):  # noqa: D102
        return [
            html.Div(
                [
                    dbc.Label("X-Axis"),
                    dbc.Select(
                        id=register("xaxis", ["value", "options"]),
                        placeholder="Select ...",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Show Runs"),
                            dbc.Select(
                                id=register("show_runs", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Groups"),
                            dbc.Select(
                                id=register("show_groups", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    def load_inputs(self):  # noqa: D102
        return {
            "objective_id": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "budget_id": {
                "options": self.budget_options,
                "value": self.budget_options[-1]["value"],
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Logarithmic Time", "value": "times_log"},
                    {"label": "Evaluated trials", "value": "trials"},
                ],
                "value": "times",
            },
            "show_runs": {"options": get_select_options(binary=True), "value": True},
            "show_groups": {"options": get_select_options(binary=True), "value": True},
        }

    @staticmethod
    def process(run, inputs):  # noqa: D102
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        times, costs_mean, costs_std, ids, config_ids = run.get_trajectory(
            objective=objective, budget=budget
        )

        return {
            "times": times,
            "costs_mean": costs_mean,
            "costs_std": costs_std,
            "ids": ids,
            "config_ids": config_ids,
        }

    @staticmethod
    def get_output_layout(register):  # noqa: D102
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    def load_outputs(runs, inputs, outputs):  # noqa: D102
        show_runs = inputs["show_runs"]
        show_groups = inputs["show_groups"]
        objective = None

        if not show_runs and not show_groups:
            return go.Figure()

        traces = []
        for idx, run in enumerate(runs):
            if run.prefix == "group" and not show_groups:
                continue

            if run.prefix != "group" and not show_runs:
                continue

            objective = run.get_objective(inputs["objective_id"])
            config_ids = outputs[run.id]["config_ids"]
            x = outputs[run.id]["times"]
            if inputs["xaxis"] == "trials":
                x = outputs[run.id]["ids"]

            y = np.array(outputs[run.id]["costs_mean"])
            y_err = np.array(outputs[run.id]["costs_std"])
            y_upper = list(y + y_err)
            y_lower = list(y - y_err)
            y = list(y)

            hovertext = ""
            hoverinfo = "skip"
            symbol = None
            mode = "lines"
            if len(config_ids) > 0:
                hovertext = [get_hovertext_from_config(run, config_id) for config_id in config_ids]
                hoverinfo = "text"
                symbol = "circle"
                mode = "lines+markers"

            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    name=run.name,
                    line_shape="hv",
                    line=dict(color=get_color(idx)),
                    hovertext=hovertext,
                    hoverinfo=hoverinfo,
                    marker=dict(symbol=symbol),
                    mode=mode,
                )
            )

            traces.append(
                go.Scatter(
                    x=x,
                    y=y_upper,
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                    marker=dict(symbol=None),
                )
            )

            traces.append(
                go.Scatter(
                    x=x,
                    y=y_lower,
                    fill="tonexty",
                    fillcolor=get_color(idx, 0.2),
                    line=dict(color=get_color(idx, 0)),
                    line_shape="hv",
                    hoverinfo="skip",
                    showlegend=False,
                    marker=dict(symbol=None),
                )
            )

        if objective is None:
            raise PreventUpdate

        type = None
        if inputs["xaxis"] == "times_log":
            type = "log"

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"] == "trials":
            xaxis_label = "Number of evaluated trials"

        layout = go.Layout(
            xaxis=dict(title=xaxis_label, type=type),
            yaxis=dict(title=objective.name),
            margin=config.FIGURE_MARGIN,
        )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "cost_over_time.pdf")

        return figure
