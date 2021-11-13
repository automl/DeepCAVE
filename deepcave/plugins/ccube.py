import numpy as np
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import json
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color
from deepcave.utils.layout import get_slider_marks, get_select_options, get_checklist_options

logger = get_logger(__name__)


class CostOverTime(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "ccube"

    @staticmethod
    def name():
        return "Configurations Cube"

    @staticmethod
    def position():
        return 5

    @staticmethod
    def category():
        return "Performance Analysis"

    @staticmethod
    def get_input_layout(register):
        return []

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div([
                dbc.Select(
                    id=register("selected_run_name", ["options", "value"]),
                    placeholder="Select run ..."
                ),
            ], className="mb-3"),

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
            ], className="mb-3"),

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
            "selected_run_name": {
                "options": get_select_options(runs.keys()),
                "value": None
            },
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
    def load_dependency_inputs(runs, previous_inputs, inputs):
        previous_run_name = previous_inputs["selected_run_name"]["value"]
        run_name = inputs["selected_run_name"]["value"]

        # Reset everything
        if previous_run_name is not None and previous_run_name != run_name:
            inputs = __class__.load_inputs(runs)
            inputs["selected_run_name"]["value"] = inputs["selected_run_name"]["value"]

        run = runs[run_name]
        budget_id = inputs["budget"]["value"]
        budgets = run.get_budgets()
        hp_names = run.configspace.get_hyperparameter_names()
        readable_budgets = run.get_budgets(human=True)
        configs = run.get_configs(budget=budgets[budget_id])
        objective_names = run.get_objective_names()

        objective_value = inputs["objective"]["value"]
        if objective_value is None:
            objective_value = objective_names[0]

        inputs.update({
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
                "value": inputs["budget"]["value"]
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": inputs["n_configs"]["value"]
            },
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_value
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
                "value": inputs["hyperparameters"]["value"]
            },
        })

        # Restrict to three hyperparameters
        selected = inputs["hyperparameters"]["value"]
        n_selected = len(selected)
        if n_selected > 3:
            del selected[0]

        inputs["hyperparameters"]["value"] = selected

        return inputs

    @staticmethod
    def process(run, inputs):
        # We leave the process method out here.
        return {}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(runs, inputs, outputs, groups):
        selected_run_name = inputs["selected_run_name"]["value"]
        if selected_run_name is None:
            return [px.scatter()]

        hp_names = inputs["hyperparameters"]["value"]
        n_configs = inputs["n_configs"]["value"]
        objective_name = inputs["objective"]["value"]
        budget_id = inputs["budget"]["value"]
        budget = runs[selected_run_name].get_budget(budget_id)
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

        run = runs[selected_run_name]

        df = run.get_encoded_configs(
            objective_names=[objective_name],
            budget=budget,
            pandas=True)

        # Limit to n_configs
        df.drop(list(range(n_configs + 1, len(df))), inplace=True)

        if x is None:
            return [px.scatter()]

        if z is None:
            if y is None:
                df[""] = df["cost"]
                for k in df[""].keys():
                    df[""][k] = 0

                y = ""

            fig = px.scatter(df, x=x, y=y, color='cost')
        else:
            fig = px.scatter_3d(df, x=x, y=y, z=z, color='cost')

        return [fig]
