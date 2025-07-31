# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Importances

This module provides a plugin for the visualization of the importances.

Provided utilities include getting input and output layout (filtered or non-filtered),
processing the data and loading the outputs. Also provides a matplotlib version.

## Classes
    - Importances: This class provides a plugin for the visualization of the importances.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from io import StringIO

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace import ConfigurationSpace, Constant
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.evaluators.mo_fanova import MOfANOVA
from deepcave.evaluators.mo_lpi import MOLPI
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.cast import optional_int
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color, save_image

logger = get_logger(__name__)


class Importances(StaticPlugin):
    """
    Provide a plugin for the visualization of the importances.

    Evaluators are fANOVA and LPI (local parameter importance).

    Provided utilities include getting input/output layout, data processing
    and loading outputs. Also provides a matplotlib version.
    """

    id = "importances"
    name = "Importances"
    icon = "far fa-star"
    help = "docs/plugins/importances.rst"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layout for the input block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective 1"),
                            dbc.Select(
                                id=register("objective_id1", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Objective 2"),
                            dbc.Select(
                                id=register("objective_id2", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Method"),
                            help_button(
                                "Local Parameter Importance: Quantify importance by changing the "
                                "neighborhood of a configuration. Uses default and incumbent "
                                "configuration as reference. \n\n"
                                "fANOVA: Quantify importance globally."
                            ),
                            dbc.Select(
                                id=register("method", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Trees"),
                            help_button(
                                "The number of trees of the internal random forest to estimate the "
                                "hyperparameter importance. "
                                "The more trees are used the more accurate the results. "
                                "However, also it takes longer to compute."
                            ),
                            dbc.Input(id=register("n_trees", type=optional_int), type="number"),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[html.Div]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[html.Div]
            Layout for the filter block.
        """
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Limit Hyperparameters"),
                    help_button(
                        "Shows only the n most important hyperparameters. If an important "
                        "hyperparameter was de-selected, it is not shown but skipped."
                    ),
                    dbc.Input(id=register("n_hps", "value"), type="number"),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budgets"),
                    help_button(
                        "Budget refers to the multi-fidelity budget. "
                        "The hyperparameters are sorted by the highest budget."
                    ),
                    dbc.Checklist(id=register("budget_ids", ["value", "options"]), inline=True),
                ]
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        If the plugin is called for the first time, or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Content to be filled.
        """
        method_labels = [
            "Local Parameter Importance (local)",
            "fANOVA (global)",
        ]
        method_values = ["local", "global"]

        return {
            "method": {
                "options": get_select_options(method_labels, method_values),
                "value": "local",
            },
            "n_trees": {"value": 10},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "n_hps": {"value": 0},
            "budget_ids": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore # noqa: E501
        """
        Works like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs : Dict[str, Any]
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the changes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_value1 = inputs["objective_id1"]["value"]
        objective_value2 = inputs["objective_id2"]["value"]  # in the multi-objective case

        # Pre-set values
        if objective_value1 is None:
            objective_value1 = objective_ids[0]

        objective_options = get_select_options(objective_names, objective_ids)
        objective_options2 = [
            dict for dict in objective_options if dict["value"] != objective_value1
        ]  # make sure the same objective cannot be chosen twice
        objective_options2 += [
            {"label": "Select objective ...", "value": -1}
        ]  # add the option to deselect the second objective

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_ids"]["value"]

        hp_names = list(run.configspace.keys())
        hp_options = get_checklist_options(hp_names)
        hp_value = inputs["hyperparameter_names"]["value"]
        n_hps = inputs["n_hps"]["value"]

        if n_hps == 0:
            n_hps = len(hp_names)

        # Pre-selection of the hyperparameters
        if run is not None:
            if len(hp_value) == 0:
                hp_value = hp_names
            if len(budget_value) == 0:
                budget_value = [budget_ids[-1]]

        return {
            "objective_id1": {
                "options": objective_options,
                "value": objective_value1,
            },
            "objective_id2": {
                "options": objective_options2,
                "value": objective_value2,
            },
            "method": {
                "value": inputs["method"]["value"],
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_value,
            },
            "budget_ids": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_hps": {"value": n_hps},
            "n_trees": {"value": inputs["n_trees"]["value"]},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return raw data based on the run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs :  Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.

        Raises
        ------
        RuntimeError
            If the number of trees is not specified.
            If the method is not found.
        """
        objective: Any = None
        objective = run.get_objective(inputs["objective_id1"])
        if inputs["objective_id2"] not in (None, -1):
            objective = [objective, run.get_objective(inputs["objective_id2"])]
        method = inputs["method"]
        n_trees = inputs["n_trees"]

        if n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        # Handle constant values in fANOVA: As the fANOVA implementation relies on pyrfr
        # and pyrfr cannot be applied to constant hyperparameters
        # (see https://github.com/automl/fanova/issues/81), as a workaround constant
        # hyperparameters are removed before calculation.
        # Note: This will break if there are conditions or forbiddens including
        # constant hyperparameters.
        hp_dict = run.configspace.get_hyperparameters_dict()
        if method == "global" and any([type(v) == Constant for v in hp_dict.values()]):
            hp_dict_wo_const = {k: v for k, v in hp_dict.items() if type(v) != Constant}
            configspace_wo_const = ConfigurationSpace()
            for k in hp_dict_wo_const.keys():
                configspace_wo_const.add_hyperparameter(hp_dict_wo_const[k])
            configspace_wo_const.add(run.configspace.conditions)
            configspace_wo_const.add(run.configspace.forbidden_clauses)
            run.configspace = configspace_wo_const

            configs_wo_const = []
            for n in range(len(run.configs)):
                configs_wo_const.append(
                    {k: v for k, v in run.configs[n].items() if k in hp_dict_wo_const.keys()}
                )
            run.configs = dict(enumerate(configs_wo_const))

        hp_names = list(run.configspace.keys())
        budgets = run.get_budgets(include_combined=True)

        # Initialize the evaluator
        evaluator: Optional[Union[LocalEvaluator, GlobalEvaluator]] = None
        if method == "local" and isinstance(objective, list):
            evaluator = MOLPI(run)
        elif method == "local":
            evaluator = LocalEvaluator(run)
        elif method == "global" and isinstance(objective, list):
            evaluator = MOfANOVA(run)
        elif method == "global":
            evaluator = GlobalEvaluator(run)
        else:
            raise RuntimeError("Method was not found.")

        # Collect data
        data = {}
        importances: Union[
            str,
            dict[str, tuple[float, float]],
            dict[Union[str, tuple[str, ...]], tuple[float, float, float, float]],
        ]
        for budget_id, budget in enumerate(budgets):
            assert isinstance(budget, (int, float))
            evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)
            if isinstance(objective, list):
                assert isinstance(evaluator, (MOLPI, MOfANOVA))
                importances = evaluator.get_importances_(hp_names)
                if any(pd.read_json(StringIO(importances))["importance"].isna()):
                    logger.warning(f"Nan encountered in importance values for budget {budget}.")
            else:
                importances = evaluator.get_importances(hp_names)
                assert isinstance(importances, Dict)
                if any(np.isnan(val) for value in importances.values() for val in value):
                    logger.warning(f"Nan encountered in importance values for budget {budget}.")
            data[budget_id] = importances

        return data  # type: ignore

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            Layout for the output block.
        """
        return dcc.Graph(
            register("graph", "figure"),
            style={"height": config.FIGURE_HEIGHT},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in raw data and prepare for layout.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.figure
            The figure of the importances.
        """
        if inputs["objective_id2"] not in (None, -1):
            # MO case: other plot
            return Importances.load_ouputs_mo_fanova(run, inputs, outputs)

        # First selected, should always be shown first
        selected_hp_names = inputs["hyperparameter_names"]
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if len(selected_hp_names) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            x = []
            y = []
            error_y = []
            for hp_name, results in importances.items():
                if hp_name not in selected_hp_names:
                    continue
                x += [hp_name]
                y += [results[0]]
                error_y += [results[1]]

            data[budget_id] = (np.array(x), np.array(y), np.array(error_y))

        # Sort by last fidelity
        selected_budget_id = max(selected_budget_ids)
        idx = np.argsort(data[selected_budget_id][1], axis=None)[::-1]
        idx = idx[:n_hps]

        bar_data = []
        for budget_id, values in data.items():
            budget = run.get_budget(budget_id, human=True)

            x = values[0][idx]

            bar_data += [
                go.Bar(
                    name=budget,
                    x=x,
                    y=values[1][idx],
                    error_y_array=values[2][idx],
                    marker_color=get_color(budget_id),
                )
            ]

        figure = go.Figure(data=bar_data)
        figure.update_layout(
            barmode="group",
            yaxis_title="Importance",
            legend={"title": "Budget"},
            margin=config.FIGURE_MARGIN,
            xaxis=dict(tickangle=-45),
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        save_image(figure, "importances.pdf")

        return figure

    @staticmethod
    def load_ouputs_mo_fanova(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Multi-objective case for read in raw data and prepare for layout.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.figure
            The figure of the importances.
        """
        # First selected, should always be shown first
        objective1 = run.get_objective(inputs["objective_id1"]).name
        selected_hp_names = inputs["hyperparameter_names"]
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if len(selected_hp_names) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances_json in outputs.items():
            df_importances = pd.read_json(StringIO(importances_json))
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            df_importances = df_importances[
                df_importances["hp_name"].isin(selected_hp_names)
            ]  # only keep selected hps
            data[budget_id] = df_importances

        # Keep only n_hps most important hyperparameters according to max importance
        selected_budget_id = max(selected_budget_ids)
        idx = (
            data[selected_budget_id]
            .groupby("hp_name")["importance"]
            .max()
            .sort_values(ascending=False)
            .index
        )
        idx = idx[:n_hps]

        color_palette = px.colors.qualitative.Plotly  # Choose a color palette
        colors = {
            hp: color_palette[i % len(color_palette)]
            for i, hp in enumerate(list(run.configspace.keys()))
        }

        # Create the figure
        figure = go.Figure()
        df = data[selected_budget_id][
            data[selected_budget_id]["hp_name"].isin(idx)
        ]  # only keep top hps

        # Group by 'hp_name' and plot each group
        for group_id, group_data in df.groupby("hp_name"):
            # Sort data by the weight column
            group_data = group_data.sort_values(by="weight")

            figure.add_trace(
                go.Scatter(
                    x=group_data["weight"],
                    y=group_data["importance"],
                    mode="lines",
                    name=group_id,
                    line=dict(color=colors[group_id]),
                )
            )

            # Add the shaded area representing the variance
            x = group_data["weight"]
            y1 = (group_data["importance"] - group_data["variance"]).to_list()
            y2 = group_data["importance"] + group_data["variance"]

            figure.add_trace(
                go.Scatter(
                    x=x.tolist() + x[::-1].tolist(),
                    y=y1 + y2[::-1].tolist(),
                    fill="toself",
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.2,
                    fillcolor=colors[group_id],
                    line=dict(color=colors[group_id]),
                    mode="lines",
                )
            )

        # Update the layout for labels, title, and axis limits
        figure.update_layout(
            xaxis_title="Weight for " + objective1,
            yaxis_title="Importance",
            xaxis=dict(range=[0, 1], tickangle=-45),
            yaxis=dict(range=[0, df["importance"].max()]),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
        )

        save_image(figure, "importances.pdf")

        return figure
