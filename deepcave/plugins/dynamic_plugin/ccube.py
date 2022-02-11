import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html

from deepcave import run_handler
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class CCube(DynamicPlugin):
    id = "ccube"
    name = "Configurations Cube"
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
                    dcc.Slider(id=register("budget", ["min", "max", "marks", "value"])),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Number of Configurations"),
                    dcc.Slider(
                        id=register("n_configs", ["min", "max", "marks", "value"])
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
            "budget": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "objective": {"options": get_select_options(), "value": None},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        budget_id = inputs["budget"]["value"]
        budgets = selected_run.get_budgets()
        hp_names = selected_run.configspace.get_hyperparameter_names()
        readable_budgets = selected_run.get_budgets(human=True)
        configs = selected_run.get_configs(budget=budgets[budget_id])
        objective_names = selected_run.get_objective_names()

        objective_value = inputs["objective"]["value"]
        if objective_value is None:
            objective_value = objective_names[0]

        new_inputs = {
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
            },
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_value,
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
            },
        }
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
        objective_name = inputs["objective"]["value"]
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        df, df_labels = run.get_encoded_configs(
            objective_names=[objective_name], budget=budget, pandas=True
        )

        # Now we also need to know when to use the labels and when to use the encoded data
        show_all_labels = []
        for hp in run.configspace.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter) or isinstance(hp, Constant):
                show_all_labels.append(True)
            else:
                show_all_labels.append(False)

        return {
            "df": serialize(df),
            "df_labels": serialize(df_labels),
            "show_all_labels": show_all_labels,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure"), style={"height": "50vh"}),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        hp_names = inputs["hyperparameters"]["value"]
        n_configs = inputs["n_configs"]["value"]
        # show_all_labels = outputs[run_name]["show_all_labels"]

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

        run = run_handler.from_run_id(inputs["run_name"]["value"])
        output = outputs[run.name]
        df = deserialize(output["df"], dtype=pd.DataFrame)
        df_labels = deserialize(output["df_labels"], dtype=pd.DataFrame)

        # Limit to n_configs
        idx = [str(i) for i in range(n_configs + 1, len(df))]
        df = df.drop(idx)
        df_labels = df_labels.drop(idx)

        column_names = df.columns.tolist()
        cost_name = column_names[-1]

        if x is None:
            return [px.scatter()]

        # hovertemplate = ""
        # for name in column_names:
        #    hovertemplate += name + ": %{df_labels[name]}<br>"

        if z is None:
            # Add another column with zeros
            if y is None:
                y = ""

                # Do it for both dataframes
                for frame in [df, df_labels]:
                    frame[y] = frame[cost_name]
                    for k in frame[y].keys():
                        frame[y][k] = 0

            fig = px.scatter(df, x=x, y=y, color=cost_name, hover_data=df_labels)
        else:
            fig = px.scatter_3d(
                df, x=x, y=y, z=z, color=cost_name, hover_data=df_labels
            )

        scene = {}
        for axis, name in zip([x, y, z], ["xaxis", "yaxis", "zaxis"]):
            if axis is None:
                continue

            values = df[axis].values
            labels = df_labels[axis].values

            unique_values = []  # df[hp_name].unique()
            unique_labels = []  # df_labels[hp_name].unique()
            for value, label in zip(values, labels):
                if value not in unique_values and label not in unique_labels:
                    unique_values.append(value)
                    unique_labels.append(label)

            selected_values = []
            selected_labels = []

            # If we have less than 10 values, we also show them
            # if show_all_labels[axis] or len(unique_values) < 10:
            if len(unique_values) < 10:
                # Make sure we don't have multiple (same) labels for the same value
                for value, label in zip(unique_values, unique_labels):
                    selected_values.append(value)
                    selected_labels.append(label)

            else:
                # Add min+max values
                for idx in [np.argmin(values), np.argmax(values)]:
                    selected_values.append(values[idx])
                    selected_labels.append(labels[idx])

                # After we added min and max values, we want to add
                # intermediate values too
                min_v = np.min(values)
                max_v = np.max(values)
                for factor in [0.2, 0.4, 0.6, 0.8]:
                    new_v = (factor * (max_v - min_v)) + min_v
                    idx = np.abs(unique_values - new_v).argmin(axis=-1)

                    selected_values.append(unique_values[idx])
                    selected_labels.append(unique_labels[idx])

            kwargs = {
                name: {
                    "tickvals": selected_values,
                    "ticktext": selected_labels,
                }
            }
            fig.update_scenes(**kwargs)

            if axis is None:
                continue

            # scatter and scatter3d handle the zaxis differently
            scene[name] = kwargs[name]

        if z is None:
            fig.update_layout(**scene)

        return [fig]
