from typing import List
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave.evaluators.ice import ICE as ICEEvaluator
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.layout import (
    get_radio_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.styled_plotty import get_color

"""
class ICE(StaticPlugin):
    id = "ice"
    name = "Individual Conditional Expectation"
    icon = "fas fa-grip-lines"
    activate_run_selection = True

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
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.RadioItems(
                        id=register("hyperparameters", ["options", "value"])
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Show Confidence Bands"),
                    dbc.RadioItems(
                        id=register("confidence_bands", ["options", "value"])
                    ),
                ],
                className="",
            ),
        ]

    def load_inputs(self):
        hp_names = run.configspace.get_hyperparameter_names()
        hp_idx = [
            run.configspace.get_idx_by_hyperparameter_name(hp_name)
            for hp_name in hp_names
        ]
        readable_budgets = run.get_budgets(human=True)
        objective_names = run.get_objective_names()

        budget_marks = get_slider_marks(readable_budgets)
        objective_options = get_select_options(objective_names)
        hp_options = get_radio_options(hp_names, hp_idx)
        ci_options = get_radio_options(binary=True)

        return {
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": budget_marks,
                "value": 0,
            },
            "objective": {
                "options": objective_options,
                "value": objective_options[0]["value"],
            },
            "hyperparameters": {"options": hp_options, "value": hp_options[0]["value"]},
            "confidence_bands": {
                "options": ci_options,
                "value": ci_options[0]["value"],
            },
        }

    @staticmethod
    def process(run, inputs):
        objective_name = inputs["objective"]["value"]
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        X, Y = run.get_encoded_configs(
            objective_names=[objective_name],
            budget=budget,
        )

        evaluator = ICEEvaluator()
        evaluator.fit(run.configspace, X, Y)

        return {"data": serialize(evaluator.get_data())}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph-mean", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        s = inputs["hyperparameters"]["value"]

        if s is None:
            return PreventUpdate

        hp_name = inputs["hyperparameters"]["options"][s]["label"]

        traces = []
        for idx, (run_name, run_outputs) in enumerate(outputs.items()):
            data = deserialize(run_outputs["data"], dtype=np.ndarray)
            evaluator = ICEEvaluator(data)

            all_x, all_y = evaluator.get_ice_data(s, variance_based=False)
            x, y, y_std = evaluator.get_pdp_data(s, variance_based=False)

            y_upper = list(y + y_std)
            y_lower = list(y - y_std)
            y_hat = np.mean(y, axis=0)

            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    showlegend=True,
                    name=f"{run_name} ({np.round(y_hat, 3)})",
                    line_color=get_color(idx, alpha=1),
                )
            )

            if not inputs["confidence_bands"]["value"]:
                for x, y in zip(all_x, all_y):
                    traces.append(
                        go.Scatter(
                            x=x,
                            y=y,
                            showlegend=False,
                            line_color=get_color(idx, alpha=0.05),
                            hoverinfo="skip",
                        )
                    )
            else:
                traces.append(
                    go.Scatter(
                        x=x,
                        y=y_upper,
                        line=dict(color=get_color(idx, 0)),
                        # line_shape='hv',
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
                        # line_shape='hv',
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        layout = go.Layout(
            xaxis=dict(
                title=hp_name,
            ),
            yaxis=dict(
                title=inputs["objective"]["value"],
            ),
        )

        return go.Figure(data=traces, layout=layout)
"""
