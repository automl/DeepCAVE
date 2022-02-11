from typing import Optional

from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html

from deepcave import run_handler
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class ParallelCoordinates(DynamicPlugin):
    id = "parallel_coordinates"
    name = "Parallel Coordinates"
    description = """
        This type of visualisation is used for plotting multivariate, numerical data. Parallel
        Coordinates Plots are ideal for comparing many variables together and
        seeing the relationships between them. For example, if you had to compare an array
        of products with the same attributes (comparing computer or cars specs across
        different models).
    """
    icon = "far fa-map"
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
                    dbc.Checklist(id=register("hyperparameters", ["options", "value"])),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run):
        hp_names = selected_run.configspace.get_hyperparameter_names()
        readable_budgets = selected_run.get_budgets(human=True)
        objective_names = selected_run.get_objective_names()

        objective_value = inputs["objective"]["value"]
        if objective_value is None:
            objective_value = objective_names[0]

        new_inputs = {
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_value,
            },
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
            },
        }
        update_dict(inputs, new_inputs)

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
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        hp_names = inputs["hyperparameters"]["value"]
        run = run_handler.from_run_id(inputs["run_name"]["value"])
        outputs = outputs[run.name]

        show_all_labels = outputs["show_all_labels"]

        df = outputs["df"]
        df = deserialize(df, dtype=pd.DataFrame)
        df_labels = outputs["df_labels"]
        df_labels = deserialize(df_labels, dtype=pd.DataFrame)

        # Dummy data to understand the structure
        # data = {
        #     "hp1": {
        #         "values": [0, 1, 2],
        #         "label": "HP1",
        #     },
        #     "hp2": {
        #         "values": [0, 4, 2],
        #         "label": "HP2",
        #     },
        # }

        data = defaultdict(dict)
        for hp_name, show_all in zip(hp_names, show_all_labels):
            values = df[hp_name].values
            labels = df_labels[hp_name].values

            unique_values = []  # df[hp_name].unique()
            unique_labels = []  # df_labels[hp_name].unique()
            for value, label in zip(values, labels):
                if value not in unique_values and label not in unique_labels:
                    unique_values.append(value)
                    unique_labels.append(label)

            data[hp_name]["values"] = values
            data[hp_name]["label"] = hp_name

            selected_values = []
            selected_labels = []

            # If we have less than 10 values, we also show them
            if show_all or len(unique_values) < 10:
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

            data[hp_name]["tickvals"] = selected_values
            data[hp_name]["ticktext"] = selected_labels

        objective = inputs["objective"]["value"]
        data[objective]["values"] = df[objective].values
        data[objective]["label"] = objective

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=data[objective]["values"],
                    showscale=True,
                ),
                dimensions=list([d for d in data.values()]),
            )
        )

        return [fig]
