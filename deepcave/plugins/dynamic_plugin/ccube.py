import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_tick_data

logger = get_logger(__name__)


class CCube(DynamicPlugin):
    id = "ccube"
    name = "Configuration Cube"
    icon = "fas fa-cube"
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
                    dbc.Select(
                        id=register("budget", ["options", "value"]),
                        placeholder="Select budget ...",
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
                        id=register("n_configs", ["min", "max", "marks", "value"]), step=None
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(id=register("hyperparameters", ["options", "value"])),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"options": get_select_options(), "value": None},
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        # Prepare objetives
        objective_names = selected_run.get_objective_names()
        objective_options = get_select_options(objective_names)
        objective_value = inputs["objective"]["value"]

        # Prepare budgets
        budgets = selected_run.get_budgets(human=True)
        budget_options = get_select_options(budgets, range(len(budgets)))

        # Prepare others
        hp_names = selected_run.configspace.get_hyperparameter_names()

        # Get selected values
        n_configs_value = inputs["n_configs"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_names[0]
            budget_value = budget_options[-1]["value"]
        else:
            budget_value = inputs["budget"]["value"]

        budget = selected_run.get_budget(int(budget_value))
        configs = selected_run.get_configs(budget=budget)
        if n_configs_value == 0:
            n_configs_value = len(configs) - 1
        else:
            if n_configs_value > len(configs) - 1:
                n_configs_value = len(configs) - 1

        new_inputs = {
            "objective": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": n_configs_value,
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
            },
        }

        # We merge the new inputs with the previous inputs
        update_dict(inputs, new_inputs)

        # Restrict to three hyperparameters
        selected = inputs["hyperparameters"]["value"]
        n_selected = len(selected)
        if n_selected > 3:
            del selected[0]

        inputs["hyperparameters"]["value"] = selected

        return inputs

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))
        objective = run.get_objective(inputs["objective"]["value"])

        df_data, df_labels = run.get_encoded_configs(
            objectives=[objective], budget=budget, specific=False, pandas=True
        )
        return {
            "df_data": serialize(df_data),
            "df_labels": serialize(df_labels),
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),
        ]

    def load_outputs(self, inputs, outputs, run):
        df_data = deserialize(outputs["df_data"], dtype=pd.DataFrame)
        df_labels = deserialize(outputs["df_labels"], dtype=pd.DataFrame)
        hp_names = inputs["hyperparameters"]["value"]
        n_configs = inputs["n_configs"]["value"]

        if n_configs == 0 or len(hp_names) == 0:
            return [px.scatter()]

        x, y, z = None, None, None
        for i, hp_name in enumerate(hp_names):
            if i == 0:
                x = hp_name
            if i == 1:
                y = hp_name
            if i == 2:
                z = hp_name

        # Limit to n_configs
        idx = [str(i) for i in range(n_configs, len(df_data))]
        df_data = df_data.drop(idx)
        df_labels = df_labels.drop(idx)

        column_names = df_data.columns.tolist()
        cost_name = column_names[-1]

        if x is None:
            return [px.scatter()]

        if z is None:
            # Add another column with zeros
            if y is None:
                y = ""

                for df in [df_data, df_labels]:
                    df[y] = df[cost_name]
                    for k in df[y].keys():
                        df[y][k] = 0

            fig = px.scatter(
                df_data,
                x=x,
                y=y,
                color=cost_name,
            )
        else:

            fig = px.scatter_3d(
                df_data,
                x=x,
                y=y,
                z=z,
                color=cost_name,
            )

        scene = {}
        for axis, name in zip([x, y, z], ["xaxis", "yaxis", "zaxis"]):
            if axis is None or axis == "":
                continue

            # Get hyperparameter
            hp = run.configspace.get_hyperparameter(axis)
            tickvals, ticktext = get_tick_data(hp, ticks=4, include_nan=True)

            scene_kwargs = {
                name: {
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                }
            }
            fig.update_scenes(**scene_kwargs)

            if axis is None:
                continue

            # scatter and scatter3d handle the zaxis differently
            scene[name] = scene_kwargs[name]

        if z is None:
            fig.update_layout(**scene)

        fig.update_traces(
            {
                "hovertemplate": None,
                "hoverinfo": "skip",
            }
        )

        return [fig]
