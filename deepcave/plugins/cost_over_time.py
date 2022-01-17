from typing import Union, Optional

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.layout import get_slider_marks, get_select_options, get_radio_options
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color

logger = get_logger(__name__)


class CostOverTime(DynamicPlugin):
    id = "cost_over_time"
    name = "Cost Over Time"
    category = "Performance Analysis"
    position = 10

    @staticmethod
    def check_requirements(runs, groups) -> Union[bool, str]:
        # Check if selected runs have same budgets+objectives
        objectives = None
        budgets = None

        for _, run in runs.items():
            if objectives is None or budgets is None:
                objectives = run.get_objective_names()
                budgets = run.get_budgets()
            else:
                if objectives != run.get_objective_names():
                    return f"Objectives differ across the selected runs."
                if budgets != run.get_budgets():
                    return f"Budgets differ across the selected runs."

        return True

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div([
                dbc.Label("Objective"),
                dbc.Select(
                    id=register("objective", ["options", "value"]),
                    placeholder="Select objective ..."
                ),
            ], className="mb-3"),

            html.Div([
                dbc.Label("Budget"),
                dcc.Slider(id=register(
                    "budget", ["min", "max", "marks", "value"])),
            ], className=""),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div([
                dbc.Label("X-Axis"),
                dbc.RadioItems(id=register("xaxis", ["options", "value"])),
            ], className="mb-3"),

            html.Div([
                dbc.Label("Logarithmic"),
                dbc.RadioItems(id=register("log", ["options", "value"])),
            ], className="mb-3"),

            html.Div([
                dbc.Label("Show Groups"),
                dbc.RadioItems(id=register("groups", ["options", "value"])),
            ], className=""),
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
                "value": objective_names[0]
            },
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
                "value": 0
            },
            "xaxis": {
                "options": [
                    {"label": "Time", "value": "times"},
                    {"label": "Number of evaluated configurations", "value": "configs"}],
                "value": "times"
            },
            "log": {
                "options": get_radio_options(binary=True),
                "value": True
            },
            "groups": {
                "options": get_radio_options(binary=True),
                "value": False
            },
        }

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        costs, times, ids = run.get_trajectory(
            objective_names=[inputs["objective"]["value"]],
            budget=budget)

        return {
            "costs": costs,
            "times": times,
            "configs": ids,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        show_groups = inputs["groups"]["value"]
        if show_groups is not None:
            groups = {}
            for run_name in outputs.keys():
                groups[run_name] = [run_name]

        traces = []
        for group_idx, (group_name, run_names) in enumerate(groups.items()):
            all_x = []  # All required x
            for run_name in run_names:
                name_ = outputs[run_name]
                run_x = name_[inputs["xaxis"]["value"]]

                for x in run_x:
                    if x not in all_x:
                        all_x.append(x)

            all_x.sort()

            # Now look for corresponding ys
            all_y = []  # List of lists

            for x in all_x:
                group_y = []
                for run_name in run_names:
                    run_x = outputs[run_name][inputs["xaxis"]["value"]]
                    run_y = outputs[run_name]["costs"]

                    # Find closest x value
                    idx = min(range(len(run_x)), key=lambda i: abs(run_x[i] - x))
                    group_y.append(run_y[idx])

                all_y.append(group_y)

            all_x = np.array(all_x)
            all_y = np.array(all_y)

            if len(all_x) == 0 or len(all_y) == 0:
                return PreventUpdate

            y_mean = np.mean(all_y, axis=1)
            y_std = np.std(all_y, axis=1)
            y_upper = list(y_mean + y_std)
            y_lower = list(y_mean - y_std)

            traces.append(go.Scatter(
                x=all_x,
                y=y_mean,
                name=group_name,
                line_shape='hv',
                line=dict(color=get_color(group_idx)),
            ))

            traces.append(go.Scatter(
                x=all_x,
                y=y_upper,
                line=dict(color=get_color(group_idx, 0)),
                line_shape='hv',
                hoverinfo="skip",
                showlegend=False,
            ))

            traces.append(go.Scatter(
                x=all_x,
                y=y_lower,
                fill='tonexty',
                fillcolor=get_color(group_idx, 0.2),
                line=dict(color=get_color(group_idx, 0)),
                line_shape='hv',
                hoverinfo="skip",
                showlegend=False,
            ))

        type = None
        if inputs["log"]["value"]:
            type = 'log'

        xaxis_label = "Wallclock time [s]"
        if inputs["xaxis"]["value"] == "configs":
            xaxis_label = "Number of evaluated configurations"

        layout = go.Layout(
            xaxis=dict(
                title=xaxis_label,
                type=type
            ),
            yaxis=dict(
                title=inputs["objective"]["value"],
            ),
        )

        return [go.Figure(data=traces, layout=layout)]
