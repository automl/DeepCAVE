from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html
from deepcave.constants import VALUE_RANGE

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hyperparameter_ticks

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
        update_dict(inputs, new_inputs)

        return inputs

    @staticmethod
    def process(run, inputs):
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)
        objective = run.get_objective(inputs["objective"]["value"])

        df = run.get_encoded_data(objective, budget)
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, run):
        hp_names = inputs["hyperparameters"]["value"]

        df = outputs["df"]
        df = deserialize(df, dtype=pd.DataFrame)

        data = defaultdict(dict)
        for hp_name in hp_names:
            data[hp_name]["values"] = df[hp_name].values
            data[hp_name]["label"] = hp_name
            data[hp_name]["range"] = VALUE_RANGE

            hp = run.configspace.get_hyperparameter(hp_name)
            tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)

            data[hp_name]["tickvals"] = tickvals
            data[hp_name]["ticktext"] = ticktext

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
