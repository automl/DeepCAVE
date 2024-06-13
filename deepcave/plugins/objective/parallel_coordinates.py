#  noqa: D400

"""
# ParallelCoordinates

This module provides utilities for visualizing the parallel coordinates.

## Classes
    - ParallelCoordinates : Can be used for visualizing the parallel coordinates.
"""

from typing import Any, Callable, Dict, List

from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.constants import VALUE_RANGE
from deepcave.evaluators.fanova import fANOVA
from deepcave.plugins.static import StaticPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hyperparameter_ticks, save_image

logger = get_logger(__name__)


class ParallelCoordinates(StaticPlugin):
    """Can be used for visualizing the parallel coordinates."""

    id = "parallel_coordinates"
    name = "Parallel Coordinates"
    icon = "far fa-map"
    activate_run_selection = True
    help = "docs/plugins/parallel_coordinates.rst"

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to regsiter (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the input block.
        """
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
                            help_button(
                                "Budget refers to the multi-fidelity budget. "
                                "Combined budget means that the trial on the highest evaluated"
                                " budget is used.  \n "
                                "Note: Selecting combined budget might be misleading if a time"
                                " objective is used. Often, higher budget take longer to evaluate,"
                                " which might negatively influence the results."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Show Important Hyperparameters"),
                    help_button(
                        "Only the most important hyperparameters are shown which are "
                        "calculated by fANOVA using 10 trees. The more left a "
                        "hyperparameter stands, the more important it is. However, activating "
                        "this option might take longer."
                    ),
                    dbc.Select(
                        id=register("show_important_only", ["value", "options"]),
                        placeholder="Select ...",
                    ),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the filter block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Limit Hyperparameters"),
                            help_button(
                                "Shows either the n most important hyperparameters (if show "
                                "important hyperparameters is true) or the first n selected "
                                "hyperparameters."
                            ),
                            dbc.Input(id=register("n_hps", "value"), type="number"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Unsuccessful Configurations"),
                            help_button("Whether to show all configurations or only failed ones."),
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
                id=register("hide_hps", ["hidden"]),
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        So, if the plugin is called for the first time or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Content to be filled.
        """
        return {
            "show_important_only": {"options": get_select_options(binary=True), "value": "true"},
            "show_unsuccessful": {"options": get_select_options(binary=True), "value": "false"},
            "n_hps": {"value": 0},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "hide_hps": {"hidden": True},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            The dictionary with the changes.
        """
        # Prepare objectives
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
        n_hps = inputs["n_hps"]["value"]
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

        if n_hps == 0:
            n_hps = len(hp_names)

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
            "n_hps": {"value": n_hps},
            "hide_hps": {"hidden": hidden},
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            The serialized dictionary.
        """
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])
        df = run.get_encoded_data(objective, budget)
        df = df.groupby(df.columns.drop(objective.name).to_list(), as_index=False).mean()
        df = serialize(df)
        result: Dict[str, Any] = {"df": df}

        if inputs["show_important_only"]:
            # Let's run a quick fANOVA here
            evaluator = fANOVA(run)
            evaluator.calculate(objective, budget, n_trees=10, seed=0)
            importances_dict = evaluator.get_importances()
            importances = {u: v[0] for u, v in importances_dict.items()}
            important_hp_names = sorted(
                importances, key=lambda key: importances[key], reverse=False
            )
            result["important_hp_names"] = important_hp_names

        return result

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            The layouts for the output block.
        """
        return dcc.Graph(
            register("graph", "figure"),
            style={"height": config.FIGURE_HEIGHT},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            The inputs and filter values fromt the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.Figure
            The output figure.
        """
        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        show_important_only = inputs["show_important_only"]
        show_unsuccessful = inputs["show_unsuccessful"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if show_important_only:
            hp_names = outputs["important_hp_names"]
        else:
            hp_names = inputs["hyperparameter_names"]

        hp_names = hp_names[:n_hps]

        df = outputs["df"]
        df = deserialize(df, dtype=pd.DataFrame)
        objective_values = []
        for value in df[objective_name].values:
            b = np.isnan(value)
            if not show_unsuccessful:
                b = not b
            if b:
                objective_values += [value]

        data: defaultdict = defaultdict(dict)
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

        figure = go.Figure(
            data=go.Parcoords(
                line=line,
                dimensions=list([d for d in data.values()]),
                labelangle=45,
            ),
            layout=dict(
                margin=dict(t=150, b=50, l=100, r=0),
                font=dict(size=config.FIGURE_FONT_SIZE),
            ),
        )
        save_image(figure, "parallel_coordinates.pdf")

        return figure
