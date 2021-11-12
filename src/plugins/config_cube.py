import numpy as np
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import json
from src.plugins.dynamic_plugin import DynamicPlugin
from src.utils.logs import get_logger
from src.utils.styled_plotty import get_color

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
    def check_requirements(runs):
        # All runs must have the same configuration space
        pass

    @staticmethod
    def get_filter_layout(register):
        return [
            dbc.FormGroup([
                dbc.Label("Selected Run"),
                dbc.RadioItems(
                    id=register("selected_run", ["options", "value"]))
            ]),

            dbc.FormGroup([
                dbc.Label("Number of Configurations"),
                dcc.Slider(
                    id=register("n_configs", ["min", "max", "marks", "value"])),
            ]),

            dbc.FormGroup([
                dbc.Label("Budget"),
                dcc.Slider(
                    id=register("budget", ["min", "max", "marks", "value"])),
            ]),

            dbc.FormGroup([
                dbc.Label("Objective"),
                dbc.RadioItems(
                    id=register("objective", ["options", "value"]))
            ]),

            dbc.FormGroup([
                dbc.Label("Hyperparameters"),
                dbc.Checklist(
                    id=register("hyperparameters", ["options", "value"])),
            ]),
        ]

    @staticmethod
    def load_inputs(runs):
        return {
            "selected_run": {
                "options": [{name: name} for name in runs.keys()],
                "value": list(runs.keys())[0]
            },
            "n_configs": {
                "min": 0,
                "max": 0,
                "marks": {},
                "value": 0
            },
            "budget": {
                "min": 0,
                "max": 0,
                "marks": {},
                "value": 0
            },
            "objective": {
                "options": [],
                "value": None
            },
            "hyperparameters": {
                "options": [],
                "value": []
            },
        }

    @staticmethod
    def load_dependency_inputs(runs, inputs):
        run_name = inputs["selected_run"]["value"]
        run = runs[run_name]
        budget = inputs["budget"]["value"]
        budgets = run.get_budgets()
        hp_names = run.configspace.get_hyperparameter_names()

        shown_budgets = []
        for b in run.get_budgets():
            if b is None:
                continue
            shown_budgets.append(str(np.round(float(b), 2)))

        max_budget_ticks = len(shown_budgets) - 1
        if max_budget_ticks < 0:
            max_budget_ticks = 0

        # Get configs
        n_configs = len(run.get_configs(budget=budget))

        inputs.update({
            "n_configs": {
                "min": 0,
                "max": n_configs - 1,
                "marks": {str(i): i for i in range(n_configs)},
                "value": 0
            },
            "budget": {
                "min": 0,
                "max": max_budget_ticks,
                "marks": {str(i): budget for i, budget in enumerate(shown_budgets)},
                "value": 0
            },
            "objective": {
                "options": [objective for objective in run.get_objectives()],
                "value": run.get_objectives()[0]
            },
            "hyperparameters": {
                "options": [{"label": hp_name, "value": hp_name} for hp_name in hp_names],
                "value": []
            },
        })

        selected = inputs["hyperparameters"]["value"]
        n_selected = len(selected)
        if n_selected > 3:
            del selected[0]

            inputs["hyperparameters"]["value"] = selected

        return inputs

    @staticmethod
    def process(run, inputs):
        df = run.get_encoded_configs(pandas=True)
        result = df.to_json()
        parsed = json.loads(result)
        p = json.dumps(parsed)

        return {
            "df": p
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        x, y, z = None, None, None
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
