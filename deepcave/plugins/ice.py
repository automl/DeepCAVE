import numpy as np
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import json
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.utils.logs import get_logger
from deepcave.utils.data_structures import update_dict
from deepcave.utils.styled_plotty import get_color
from deepcave.utils.layout import get_slider_marks, get_select_options, get_checklist_options, get_radio_options
from deepcave.utils.compression import serialize, deserialize
from deepcave.evaluators.ice import ICE as ICEEvaluator

logger = get_logger(__name__)


class ICE(StaticPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id():
        return "ice"

    @staticmethod
    def name():
        return "Individual Conditional Expectation"

    @staticmethod
    def position():
        return 30

    @staticmethod
    def category():
        return "Performance Analysis"

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
                dbc.RadioItems(
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
            "objective": {
                "options": get_select_options(),
                "value": None
            },
            "hyperparameters": {
                "options": get_radio_options(),
                "value": None
            },
        }

    @staticmethod
    def load_dependency_inputs(runs, previous_inputs, inputs):
        run = runs[inputs["run_name"]["value"]]
        budget_id = inputs["budget"]["value"]
        budgets = run.get_budgets()
        hp_names = run.configspace.get_hyperparameter_names()
        hp_idx = [run.configspace.get_idx_by_hyperparameter_name(
            hp_name) for hp_name in hp_names]
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
            "objective": {
                "options": get_select_options(objective_names),
                "value": objective_value
            },
            "hyperparameters": {
                "options": get_radio_options(hp_names, hp_idx),
            },
        }
        update_dict(inputs, new_inputs)

        return inputs

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

        return {
            "data": serialize(evaluator.get_data())
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        s = inputs["hyperparameters"]["value"]

        if s is None:
            return PreventUpdate

        run_name = inputs["run_name"]["value"]
        hp_name = inputs["hyperparameters"]["options"][s]["label"]
        run_output = outputs[run_name]

        data = deserialize(run_output["data"], dtype=np.ndarray)
        evaluator = ICEEvaluator(data)
        all_x, all_y = evaluator.get_ice_data(s)

        traces = []
        for x, y in zip(all_x, all_y):
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    showlegend=False,
                    line_color=get_color(0, alpha=0.05),
                    hoverinfo='skip'
                ))

        x, y = evaluator.get_pdp_data(s)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                showlegend=False,
                line_color=get_color(0, alpha=1)
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

        return [go.Figure(data=traces, layout=layout)]
