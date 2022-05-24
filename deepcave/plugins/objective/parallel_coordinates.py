from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from dash import dcc, html
from deepcave.constants import VALUE_RANGE
from deepcave.evaluators.fanova import fANOVA

from deepcave.plugins.static import StaticPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hyperparameter_ticks

logger = get_logger(__name__)


class ParallelCoordinates(StaticPlugin):
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Show Important Hyperparameters"),
                            help_button(
                                "Only the most important hyperparameters are shown which are calculated by "
                                "fANOVA using five trees. The more left a hyperparameter stands, the more "
                                "important it is."
                            ),
                            dbc.Select(
                                id=register("show_important_only", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Unsuccessful Configurations"),
                            dbc.Select(
                                id=register("show_unsuccessful", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ],
                className="mt-3",
                id=register("show_hyperparameters", ["hidden"]),
            ),
        ]

    def load_inputs(self):
        return {
            "show_important_only": {"options": get_select_options(binary=True), "value": "true"},
            "show_unsuccessful": {"options": get_select_options(binary=True), "value": "false"},
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

        if inputs["show_important_only"]["value"] == "true":
            hp_options = []
            hp_value = inputs["hyperparameter_names"]["value"]
            hidden = True
        else:
            hp_options = get_select_options(hp_names)

            values = inputs["hyperparameter_names"]["value"]
            if len(values) == 0:
                values = hp_names

            hp_value = values
            hidden = False

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
            "show_hyperparameters": {"hidden": hidden},
        }

    @staticmethod
    def process(run, inputs):
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        # Let's run a quick fANOVA here
        evaluator = fANOVA(run)
        evaluator.calculate(objective, budget, n_trees=5, seed=0)
        importances = evaluator.get_importances()
        importances = {u: v[0] for u, v in importances.items()}
        important_hp_names = sorted(importances, key=lambda key: importances[key], reverse=True)[
            :10
        ]

        df = run.get_encoded_data(objective, budget)
        return {
            "df": serialize(df),
            "important_hp_names": important_hp_names,
        }

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"), style={"height": "50vh"})

    @staticmethod
    def load_outputs(run, inputs, outputs):
        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        show_important_only = inputs["show_important_only"] == "true"
        show_unsuccessful = inputs["show_unsuccessful"] == "true"

        if show_important_only:
            hp_names = outputs["important_hp_names"]
        else:
            hp_names = inputs["hyperparameter_names"]

        df = outputs["df"]
        df = deserialize(df, dtype=pd.DataFrame)
        objective_values = []
        for value in df[objective_name].values:
            b = np.isnan(value)
            if not show_unsuccessful:
                b = not b
            if b:
                objective_values += [value]

        data = defaultdict(dict)
        for hp_name in hp_names:
            values = []
            for hp_v, objective_v in zip(df[hp_name].values, df[objective_name].values):
                b = np.isnan(objective_v)
                if not show_unsuccessful:
                    b = not b
                if b:
                    values += [hp_v]

            data[hp_name]["values"] = values
            data[hp_name]["label"] = hp_name
            data[hp_name]["range"] = VALUE_RANGE

            hp = run.configspace.get_hyperparameter(hp_name)
            tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)

            data[hp_name]["tickvals"] = tickvals
            data[hp_name]["ticktext"] = ticktext

        if show_unsuccessful:
            line = dict()
        else:
            data[objective_name]["values"] = objective_values
            data[objective_name]["label"] = objective_name
            line = dict(
                color=data[objective_name]["values"],
                showscale=True,
                colorscale="aggrnyl",
            )

        fig = go.Figure(
            data=go.Parcoords(
                line=line,
                dimensions=list([d for d in data.values()]),
                labelangle=45,
            ),
            layout=dict(
                margin=dict(
                    t=150,
                    b=50,
                    l=100,
                    r=0,
                )
            ),
        )

        return fig
