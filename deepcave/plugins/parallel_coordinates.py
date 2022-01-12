from typing import Optional

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from dash import dcc
from dash import html
from collections import defaultdict
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.compression import serialize, deserialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_slider_marks, get_select_options, get_checklist_options
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class ParallelCoordinates(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id() -> str:
        return "parallel_coordinates"

    @staticmethod
    def name() -> str:
        return "Parallel Coordinates"

    @staticmethod
    def position() -> int:
        return 20

    @staticmethod
    def category() -> Optional[str]:
        return "Hyperparameter Analysis"

    @staticmethod
    def activate_run_selection():
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
                dcc.Slider(
                    id=register("budget", ["min", "max", "marks", "value"])),
            ]),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div([
                dbc.Label("Hyperparameters"),
                dbc.Checklist(
                    id=register("hyperparameters", ["options", "value"])),
            ]),
        ]

    @staticmethod
    def load_inputs(runs):
        return {
            "objective": {
                "options": get_select_options(),
                "value": None
            },
            "budget": {
                "min": 0,
                "max": 0,
                "marks": get_slider_marks(),
                "value": 0
            },
            "hyperparameters": {
                "options": get_checklist_options(),
                "value": []
            },
        }

    @staticmethod
    def load_dependency_inputs(runs, previous_inputs, inputs):
        run = runs[inputs["run_name"]["value"]]
        hp_names = run.configspace.get_hyperparameter_names()
        readable_budgets = run.get_budgets(human=True)
        objective_names = run.get_objective_names()

        objective_value = inputs["objective"]["value"]
        if objective_value is None:
            objective_value = objective_names[0]

        new_inputs = {
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_value
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
            objective_names=[objective_name],
            budget=budget,
            pandas=True
        )

        # Now we also need to know when to use the labels and when to use the encoded data
        show_all_labels = []
        for hp in enumerate(run.configspace.get_hyperparameters()):
            # TODO: FIX THIS
            print(hp)
            if isinstance(hp, CategoricalHyperparameter) or isinstance(hp, Constant):
                show_all_labels.append(True)
            else:
                show_all_labels.append(False)

        return {
            "df": serialize(df),
            "df_labels": serialize(df_labels),
            "show_all_labels": show_all_labels
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(
                register("graph", "figure")
            ),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):

        hp_names = inputs["hyperparameters"]["value"]
        run_name = inputs["run_name"]["value"]
        df = deserialize(outputs[run_name]["df"], dtype=pd.DataFrame)
        df_labels = deserialize(
            outputs[run_name]["df_labels"], dtype=pd.DataFrame)
        show_all_labels = outputs[run_name]["show_all_labels"]

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
            data[hp_name]["values"] = df[hp_name].values
            data[hp_name]["label"] = hp_name

            if show_all:
                data[hp_name]["tickvals"] = df[hp_name].values
                data[hp_name]["ticktext"] = df_labels[hp_name].values
            else:
                values = df[hp_name].values
                labels = df_labels[hp_name].values

                selected_values = []
                selected_labels = []

                min_idx = np.argmin(values)
                max_idx = np.argmax(values)

                selected_values.append(values[min_idx])
                selected_labels.append(labels[min_idx])
                selected_values.append(values[max_idx])
                selected_labels.append(labels[max_idx])

                data[hp_name]["tickvals"] = selected_values
                data[hp_name]["ticktext"] = selected_labels

        objective = inputs["objective"]["value"]
        data[objective]["values"] = df[objective].values
        data[objective]["label"] = objective

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=data[objective]["values"],
                showscale=True,
                # cmin=0,
                # cmax=1
            ),
            dimensions=list([d for d in data.values()])
        ))

        return [fig]

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

        run_name = inputs["run_name"]["value"]
        df = deserialize(outputs[run_name]["df"], dtype=pd.DataFrame)

        # Limit to n_configs
        df = df.drop([str(i) for i in range(n_configs + 1, len(df))])

        if x is None:
            return [px.scatter()]

        column_names = df.columns.tolist()
        cost_name = column_names[-1]

        if z is None:
            if y is None:
                df[""] = df[cost_name]
                for k in df[""].keys():
                    df[""][k] = 0

                y = ""

            fig = px.scatter(df, x=x, y=y, color=cost_name)
        else:
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=cost_name)

        return [fig]
