import numpy as np
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import json
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color
from deepcave.utils.layout import get_slider_marks

logger = get_logger(__name__)


class CostOverTime(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "configuration_cube"

    @staticmethod
    def name():
        return "Configuration Cube"

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
            dbc.Label("Select Run"),
            dbc.RadioItems(
                id=register("selected_run", ["options", "value"])),

            dbc.Label("Budget"),
            dcc.Slider(
                id=register("budget", ["min", "max", "marks", "value"])),

            dbc.Label("Number of Configurations"),
            dcc.Slider(
                id=register("n_configs", ["min", "max", "marks", "value"])),

            # dbc.FormGroup([
            #    dbc.Label("Objective"),
            #    dbc.RadioItems(
            #        id=register("objective", ["options", "value"]))
            # ]),

            # dbc.FormGroup([
            #    dbc.Label("Hyperparameters"),
            #    dbc.Checklist(
            #        id=register("hyperparameters", ["options", "value"])),
            # ]),
        ]

    @staticmethod
    def load_inputs(runs):
        return {
            "selected_run": {
                "options": [{"label": name, "value": name} for name in runs.keys()],
                "value": list(runs.keys())[0]
            },
            "budget": {
                "min": 0,
                "max": 0,
                "marks": {0: str("None")},
                "value": 0
            },
            "n_configs": {
                "min": 0,
                "max": 0,
                "marks": {0: str("None")},
                "value": 0
            },
            # "objective": {
            #    "options": [],
            #    "value": None
            # },
            # "hyperparameters": {
            #    "options": [],
            #    "value": []
            # },
        }

    @staticmethod
    def load_dependency_inputs(runs, previous_inputs, inputs):
        print(previous_inputs)
        print(inputs)

        run_name = inputs["selected_run"]["value"]
        run = runs[run_name]
        budget_id = inputs["budget"]["value"]
        budgets = run.get_budgets()
        hp_names = run.configspace.get_hyperparameter_names()
        readable_budgets = run.get_budgets(human=True)
        configs = run.get_configs(budget=budgets[budget_id])

        # TODO: After changing run, reset everything
        # call load inputs again

        inputs.update({
            "budget": {
                "min": 0,
                "max": len(readable_budgets) - 1,
                "marks": get_slider_marks(readable_budgets),
                "value": 0
            },
            "n_configs": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": inputs["n_configs"]["value"]
            },

            # "objective": {
            #    "options": [objective for objective in run.get_objectives()],
            #    "value": run.get_objectives()[0]
            # },
            # "hyperparameters": {
            #    "options": [{"label": hp_name, "value": hp_name} for hp_name in hp_names],
            #    "value": []
            # },
        })

        #selected = inputs["hyperparameters"]["value"]
        #n_selected = len(selected)
        # if n_selected > 3:
        #    del selected[0]

        #   inputs["hyperparameters"]["value"] = selected

        return inputs

    @staticmethod
    def process(run, inputs):
        df = run.get_encoded_configs(pandas=True)
        result = df.to_json()

        return {
            "df": json.dumps(json.loads(result))
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        x, y, z = None, None, None
        return [px.scatter()]
        for i, hp_name in enumerate(inputs["hyperparameters"]["value"]):
            if i == 0:
                x = hp_name
            if i == 1:
                y = hp_name
            if i == 2:
                z = hp_name

        for run_name, run_outputs in outputs.items():
            df = json.loads(run_outputs["df"])

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
                fig = px.scatter_3d(df, x=x, y=y, z=z,
                                    color='cost')

            return [fig]
