from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from src import app
from src.plugins.dynamic_plugin import DynamicPlugin
from src.plugins.static_plugin import StaticPlugin
from src.utils.logs import get_logger
from src.utils.styled_plotty import get_color

logger = get_logger(__name__)


class CostOverTime(StaticPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "cost_over_time"

    @staticmethod
    def name():
        return "Cost Over Time"

    @staticmethod
    def position():
        return 5

    @staticmethod
    def category():
        return "Performance Analysis"

    @staticmethod
    def get_input_layout(register):
        return [
            dbc.Label("Budget"),
            dcc.Slider(id=register(
                "budget", ["min", "max", "marks", "value"])),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            dbc.FormGroup([
                dbc.Label("X-Axis"),
                dbc.RadioItems(id=register("xaxis", ["options", "value"]))
            ]),
            dbc.FormGroup([
                dbc.Label("Logarithmic"),
                dbc.RadioItems(id=register("log", ["options", "value"]))
            ]),
            dbc.FormGroup([
                dbc.Label("Show Groups"),
                dbc.RadioItems(id=register("groups", ["options", "value"]))
            ], style={"margin-bottom": "0px"}),
        ]

    @staticmethod
    def load_inputs(runs):
        run = list(runs.values())[0]

        budgets = []
        for budget in run.get_budgets():
            if budget is None:
                continue
            budget = str(np.round(float(budget), 2))

        max_ticks = len(budgets) - 1
        if max_ticks < 0:
            max_ticks = 0

        return {
            "budget": {
                "min": 0,
                "max": max_ticks,
                "marks": {str(i): budget for i, budget in enumerate(budgets)},
                "value": 0
            },
            "xaxis": {
                "options": [{"label": "Time", "value": "times"}, {"label": "Number of evaluated configurations", "value": "configs"}],
                "value": "times"
            },
            "log": {
                "options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}],
                "value": True
            },
            "groups": {
                "options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}],
                "value": False
            },
        }

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        costs, times, ids = run.get_trajectory(budget)

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
        if not show_groups:
            groups = {}
            for run_name in outputs.keys():
                groups[run_name] = [run_name]

        traces = []
        for group_idx, (group_name, run_names) in enumerate(groups.items()):
            all_x = []  # All required x
            for run_name in run_names:
                run_x = outputs[run_name][inputs["xaxis"]["value"]]

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
                    idx = min(range(len(run_x)), key=lambda i: abs(run_x[i]-x))
                    group_y.append(run_y[idx])

                all_y.append(group_y)

            all_x = np.array(all_x)
            all_y = np.array(all_y)
            y_mean = np.mean(all_y, axis=1)
            y_std = np.std(all_y, axis=1)
            y_upper = list(y_mean+y_std)
            y_lower = list(y_mean-y_std)

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
                title='Cost',
            ),
        )

        fig = go.Figure(data=traces, layout=layout)

        graphs = []
        for group_name, _ in groups.items():
            graphs.append(fig)
            return graphs

        # return []

        return [fig]

    # def get_mpl_output_layout(self):
    #    return [
    #        dbc.Input(id=self.register_output("blub", "value", mpl=True))
    #    ]

    # def load_mpl_outputs(self, inputs, outputs):
    #    return [inputs["filter"]["value"]]
