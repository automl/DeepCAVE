"""
Single Run
^^^^^^^^^^

"""

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from deepcave.plugins.dynamic_plugin.dynamic import DynamicPlugin
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class SingleRun(DynamicPlugin):
    id = "single_run"
    name = "Single Run"
    description = ""
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
        # Default values if the plugin never was used before
        return {
            "objective": {"options": get_select_options(), "value": None},
            "budget": {"options": get_select_options(), "value": None},
            "hyperparameters": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run):
        # Prepare objetives
        objective_names = selected_run.get_objective_names()
        objective_options = get_select_options(objective_names)

        # Prepare budgets
        budgets = selected_run.get_budgets(human=True)
        budget_options = get_select_options(budgets, range(len(budgets)))

        # Prepare others
        hp_names = selected_run.configspace.get_hyperparameter_names()

        # Get selected values
        objective_value = inputs["objective"]["value"]
        budget_value = inputs["budget"]["value"]
        hp_value = inputs["hyperparameters"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_names[0]
            budget_value = budget_options[-1]["value"]
            hp_value = hp_names

        new_inputs = {
            "objective": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget": {
                "options": budget_options,
                "value": budget_value,
            },
            "hyperparameters": {
                "options": get_select_options(hp_names),
                "value": hp_value,
            },
        }

        # Use this method if you only change some inputs in `new_inputs`
        update_dict(inputs, new_inputs)

        return inputs

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))
        objective = run.get_objective(inputs["objective"]["value"])

        test = 1

        # Make sure the output is serializable
        return {
            "test": test,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, run):
        # The selected values from the user
        objective_name = inputs["objective"]["value"]
        budget = inputs["budget"]["value"]
        hyperparameter_names = inputs["hyperparameters"]["value"]

        # Get the data from `process`
        result = outputs["test"]

        # Do some filtering stuff here if needed
        # ...

        return [go.Figure()]
