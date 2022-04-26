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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(id=register("hyperparameter_names", ["value", "options"])),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        # Prepare objetives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        # Prepare others
        hp_names = run.configspace.get_hyperparameter_names()
        hp_options = get_select_options(hp_names)
        hp_value = inputs["hyperparameter_names"]["value"]

        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
            hp_value = hp_names

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_value,
            },
        }

    @staticmethod
    def process(run, inputs):
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        df = run.get_encoded_data(objective, budget)
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"))

    @staticmethod
    def load_outputs(run, inputs, outputs):
        hp_names = inputs["hyperparameter_names"]

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

        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective["name"]
        data[objective_name]["values"] = df[objective_name].values
        data[objective_name]["label"] = objective_name

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=data[objective_name]["values"],
                    showscale=True,
                ),
                dimensions=list([d for d in data.values()]),
            )
        )

        return fig
