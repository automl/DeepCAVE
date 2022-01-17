from typing import Optional

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc
from dash import html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs.run import AbstractRun
from deepcave.utils.compression import serialize, deserialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_slider_marks, get_select_options, get_checklist_options
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class CCube(DynamicPlugin):
    id = "ccube"
    name = "Configurations Cube"
    category = "Performance Analysis"
    position = 20

    activate_run_selection = True

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
                dbc.Label("Number of Configurations"),
                dcc.Slider(
                    id=register("n_configs", ["min", "max", "marks", "value"])),
            ], className="mb-3"),

            html.Div([
                dbc.Label("Hyperparameters"),
                dbc.Checklist(
                    id=register("hyperparameters", ["options", "value"])),
            ]),
        ]

    @staticmethod
    def load_inputs(runs):
        return {
            "budget": {
                "min": 0,
                "max": 0,
                "marks": get_slider_marks(),
                "value": 0
            },
            "n_configs": {
                "min": 0,
                "max": 0,
                "marks": get_slider_marks(),
                "value": 0
            },
            "objective": {
                "options": get_select_options(),
                "value": None
            },
            "hyperparameters": {
                "options": get_checklist_options(),
                "value": []
            },
        }

    @staticmethod
    def load_dependency_inputs(runs: dict[str, AbstractRun], previous_inputs, inputs):
        run = runs[inputs["run_name"]["value"]]
        budget_id = inputs["budget"]["value"]
        budgets = run.get_budgets()
        hp_names = run.configspace.get_hyperparameter_names()
        readable_budgets = run.get_budgets(human=True)
        configs = run.get_configs(budget=budgets[budget_id])
        objective_names = run.get_objective_names()

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
                "value": objective_value
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

        df = run.get_encoded_configs(
            objective_names=[objective_name],
            budget=budget,
            pandas=True
        )

        return {
            "df": serialize(df),
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
