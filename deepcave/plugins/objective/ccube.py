import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hovertext_from_config, get_hyperparameter_ticks
from deepcave.runs import Status

logger = get_logger(__name__)


class CCube(DynamicPlugin):
    id = "ccube"
    name = "Configuration Cube"
    icon = "fas fa-cube"
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
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Number of Configurations"),
                    dcc.Slider(
                        id=register("n_configs", ["value", "min", "max", "marks"]), step=None
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
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

        # Prepare others
        hp_names = run.configspace.get_hyperparameter_names()

        # Get selected values
        n_configs_value = inputs["n_configs"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
        else:
            budget_value = inputs["budget_id"]["value"]

        budget = run.get_budget(budget_value)
        configs = run.get_configs(budget=budget)
        if n_configs_value == 0:
            n_configs_value = len(configs) - 1
        else:
            if n_configs_value > len(configs) - 1:
                n_configs_value = len(configs) - 1

        # Restrict to three hyperparameters
        selected_hps = inputs["hyperparameter_names"]["value"]
        n_selected = len(selected_hps)
        if n_selected > 3:
            del selected_hps[0]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": n_configs_value,
            },
            "hyperparameter_names": {
                "options": get_select_options(hp_names),
                "value": selected_hps,
            },
        }

    @staticmethod
    def process(run, inputs):
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        df = run.get_encoded_data(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, include_config_ids=True
        )
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register):
        return (dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),)

    @staticmethod
    def load_outputs(run, inputs, outputs):
        df = deserialize(outputs["df"], dtype=pd.DataFrame)
        hp_names = inputs["hyperparameter_names"]
        n_configs = inputs["n_configs"]
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)

        # Limit to n_configs
        idx = [str(i) for i in range(n_configs, len(df))]
        df = df.drop(idx)

        costs = df[objective.name].values.tolist()
        config_ids = df["config_id"].values.tolist()
        data = []

        # Specify layout kwargs
        layout_kwargs = {}
        if n_configs > 0 and len(hp_names) > 0:
            for i, (hp_name, axis_name) in enumerate(zip(hp_names, ["xaxis", "yaxis", "zaxis"])):
                hp = run.configspace.get_hyperparameter(hp_name)
                values = df[hp_name].values.tolist()

                tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)
                layout_kwargs[axis_name] = {
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                    "title": hp_name,
                }
                data.append(values)

        # Specify scatter kwargs
        scatter_kwargs = {
            "mode": "markers",
            "marker": {
                "size": 5,
                "color": costs,
                "colorbar": {"thickness": 30, "title": objective.name},
            },
            "hovertext": [get_hovertext_from_config(run, config_id) for config_id in config_ids],
            "meta": {"colorbar": costs},
            "hoverinfo": "text",
        }

        if len(data) == 3:
            trace = go.Scatter3d(x=data[0], y=data[1], z=data[2], **scatter_kwargs)
            layout = go.Layout({"scene": {**layout_kwargs}})
        else:
            if len(data) == 1:
                trace = go.Scatter(x=data[0], y=[0 for _ in range(len(data[0]))], **scatter_kwargs)
            elif len(data) == 2:
                trace = go.Scatter(x=data[0], y=data[1], **scatter_kwargs)
            else:
                trace = go.Scatter(x=[], y=[])
            layout = go.Layout(**layout_kwargs)

        fig = go.Figure(data=trace, layout=layout)
        fig.update_layout(
            dict(
                margin=dict(
                    t=30,
                    b=0,
                    l=0,
                    r=0,
                ),
            )
        )
        return fig
