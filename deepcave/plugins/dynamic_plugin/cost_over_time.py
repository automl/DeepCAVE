from typing import Union, List, Dict

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.layout import get_radio_options, get_select_options
from deepcave.utils.styled_plotty import get_color


class CostOverTime(DynamicPlugin):
    id = "cost_over_time"
    name = "Cost Over Time"
    icon = "fas fa-chart-line"

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        check_equality(runs, objectives=True, budgets=True)

        # Set some attributes here
        run = runs[0]

        budgets = run.get_budgets(human=True)
        self.budget_options = get_select_options(budgets, range(len(budgets)))
        objective_names = run.get_objective_names()
        self.objective_options = get_select_options(objective_names)

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
                    dbc.Label("Display ..."),
                    dbc.RadioItems(id=register("display", ["options", "value"])),
                ],
                className="",
            ),
        ]

    def load_inputs(self):
        display_options = ["Runs", "Groups"]

        return {
            "objective": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "budget": {
                "options": self.budget_options,
                "value": self.budget_options[0]["value"],
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Time (logarithmic)", "value": "times_log"},
                    {"label": "Number of evaluated configurations", "value": "configs"},
                ],
                "value": "times",
            },
            "display": {
                "options": get_radio_options(display_options),
                "value": display_options[0],
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, List[Union[float, str]]]:
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))
        objective = run.get_objective(inputs["objective"]["value"])

        times, costs_mean, costs_std, ids = run.get_trajectory(objective=objective, budget=budget)

        return {
            "times": times,
            "costs_mean": costs_mean,
            "costs_std": costs_std,
            "ids": ids,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, runs):
        traces = []
        for idx, (run_name, run) in enumerate(runs.items()):
            x = outputs[run.name]["times"]
            if inputs["xaxis"]["value"] == "configs":
                x = outputs[run.name]["ids"]

            if inputs["display"]["value"] == "Runs":
                if run.prefix == "group":
                    continue
            elif inputs["display"]["value"] == "Groups":
                # Prefix could be not only run but also the name of the converter
                if run.prefix != "group":
                    continue
            else:
                raise RuntimeError("Unknown display option")

            y = np.array(outputs[run.name]["costs_mean"])
            y_err = np.array(outputs[run.name]["costs_std"])
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
        if inputs["xaxis"]["value"] == "times_log":
            type = "log"

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"]["value"] == "configs":
            xaxis_label = "Number of evaluated configurations"

        layout = go.Layout(
            xaxis=dict(title=xaxis_label, type=type),
            yaxis=dict(title=inputs["objective"]["value"]),
        )

        return [go.Figure(data=traces, layout=layout)]
