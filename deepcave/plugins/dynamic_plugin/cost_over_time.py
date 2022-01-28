from typing import Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.layout import (
    get_slider_marks,
    get_select_options,
    get_radio_options,
)
from deepcave.utils.styled_plotty import get_color
from deepcave.runs import AbstractRun


class CostOverTime(DynamicPlugin):
    id = "cost_over_time"
    name = "Cost Over Time"

    @staticmethod
    def check_compatibility(run: AbstractRun):
        # Check if selected runs have same budgets+objectives
        try:
            run.get_objective_names()
            run.get_budgets()

            return True
        except:
            return False

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
                    dcc.Slider(id=register("budget", ["min", "max", "marks", "value"])),
                ],
                className="",
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("X-Axis"),
                    dbc.RadioItems(id=register("xaxis", ["options", "value"])),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Logarithmic"),
                    dbc.RadioItems(id=register("log", ["options", "value"])),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Show Groups"),
                    dbc.RadioItems(id=register("groups", ["options", "value"])),
                ],
                className="",
            ),
        ]

    @staticmethod
    def load_inputs(runs):
        # Just select the first run
        # Since we already know the budgets+objectives
        # are the same across the selected runs.
        run = list(runs.values())[0]
        readable_budgets = run.get_budgets(human=True)
        objective_names = run.get_objective_names()

        return {
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_names[0],
            },
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
                "value": 0,
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Number of evaluated configurations", "value": "configs"},
                ],
                "value": "times",
            },
            "log": {"options": get_radio_options(binary=True), "value": True},
            "groups": {"options": get_radio_options(binary=True), "value": False},
        }

    @staticmethod
    def process(run, inputs) -> dict[str, list[Union[float, str]]]:
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        times, costs_mean, costs_std, _ = run.get_trajectory(
            objective_names=[inputs["objective"]["value"]], budget=budget
        )

        return {
            "times": times,
            "costs_mean": costs_mean,
            "costs_std": costs_std,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, runs):
        """
        show_groups = inputs["groups"]["value"]
        if show_groups is not None:
            groups = {}
            for run_name in outputs.keys():
                groups[run_name] = [run_name]
        """

        traces = []
        for idx, (run_name, run) in enumerate(runs.items()):
            x = outputs[run_name]["times"]
            if inputs["xaxis"]["value"] == "configs":
                x = [i for i in range(len(x))]

            if not inputs["groups"]["value"]:
                if run.prefix == "group":
                    continue

            y = np.array(outputs[run_name]["costs_mean"])
            y_err = np.array(outputs[run_name]["costs_std"])
            y_upper = list(y + y_err)
            y_lower = list(y - y_err)
            y = list(y)

            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    name=run_name,
                    line_shape="hv",
                    line=dict(color=get_color(idx)),
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
                )
            )

        type = None
        if inputs["log"]["value"]:
            type = "log"

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"]["value"] == "configs":
            xaxis_label = "Number of evaluated configurations"

        layout = go.Layout(
            xaxis=dict(title=xaxis_label, type=type),
            yaxis=dict(
                title=inputs["objective"]["value"],
            ),
        )

        return [go.Figure(data=traces, layout=layout)]
