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

# noqa: D400
"""
# HypeAblation

This module provides a class for the visualization of HyperSHAP ablation evaluations.

Provided utilities include getting input and output layout,
processing the data and loading the outputs.

## Classes
    - HypeAblation: Provide a plugin for HyperSHAP ablation evaluation.
"""

from typing import Any, Callable, Dict, List

import ast

import dash_bootstrap_components as dbc
import matplotlib
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from deepcave import config
from deepcave.evaluators.hypershap import HyperSHAP_Eval as Evaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button

matplotlib.use("Agg")


class HypeAblation(StaticPlugin):
    """Provide a plugin for HyperSHAP tunability evaluation."""

    id = "hype_ablation"
    name = "HyperSHAP Ablation"
    icon = "fas fa-puzzle-piece"
    help = "plugins/hype_ablation.html"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List:
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
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            Layout for the filter block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Budgets"),
                                    help_button(
                                        "Budget refers to the multi-fidelity budget. "
                                        "The hyperparameters are sorted by the highest budget."
                                    ),
                                    dbc.RadioItems(
                                        id=register("budget_id", ["value", "options"]), inline=True
                                    ),
                                ]
                            ),
                        ],
                        md=6,
                    ),
                ],
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
        return {
            "budget_id": {"options": get_checklist_options(), "value": None},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Work like 'load_inputs' but called after inputs have changed.

        It is restricted to three Hyperparameters.

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

        if objective_value is None:
            objective_value = objective_ids[0]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        if run is not None:
            if budget_value is None:
                budget_value = budget_ids[-1]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
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
        """
        eval = Evaluator(run)
        eval.hype_ablation(
            objective_id=inputs["objective_id"],
            budget_id=inputs["budget_id"],
        )

        return {"inputs": eval.get_ablation()}

    @staticmethod
    def get_output_layout(register: Callable) -> Any:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dcc.Graph]
            Layout for the output block.
        """
        return [
            dcc.Graph(
                register("perf_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            )
        ]

    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:  # type: ignore
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
        go.Figure
            The figure for tuneability visualization.
        """
        iv_values = outputs["inputs"]

        restored_interactions = {}
        for key_str, value in iv_values["interactions"].items():
            if key_str == "()":
                restored_interactions[()] = value
            else:
                restored_interactions[ast.literal_eval(key_str)] = value
        print(restored_interactions)
        cs = runs.configspace
        hp_names = list(cs.keys())

        sorted_data = sorted(restored_interactions.items(), key=lambda x: x[1], reverse=True)
        values = [item[1] for item in sorted_data]
        intersections = [
            tuple(hp_names[i] for i in item[0]) for item in sorted_data  # type: ignore
        ]
        num_params = len(hp_names)
        num_cols = len(values)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3]
        )

        pos_color = "#FF1053"
        neg_color = "#1E88E5"
        gray_dot = "#E0E0E0"
        bg_stripe = "#F5F5F5"

        bar_colors = [pos_color if v >= 0 else neg_color for v in values]

        fig.add_trace(
            go.Bar(
                x=list(range(num_cols)),
                y=values,
                marker_color=bar_colors,
                text=[f"{v:.3f}" if abs(v) > 0.0001 else "" for v in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        for i in range(num_params):
            if i % 2 == 0:
                fig.add_hrect(
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor=bg_stripe,
                    layer="below",
                    line_width=0,
                    row=2,
                    col=1,
                )

        for i, param in enumerate(hp_names):
            x_coords = list(range(num_cols))
            y_coords = [i] * num_cols

            is_active = [param in tup for tup in intersections]
            colors = ["black" if active else gray_dot for active in is_active]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(size=14, color=colors),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

        for j, tup in enumerate(intersections):
            if len(tup) > 1:
                row_indices = [hp_names.index(p) for p in tup]

                fig.add_trace(
                    go.Scatter(
                        x=[j],
                        y=[min(row_indices), max(row_indices)],
                        mode="lines",
                        line=dict(color="black", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )

        fig.update_layout(
            plot_bgcolor="white",
            autosize=True,
            margin=dict(t=30, b=0, l=0, r=0),
        )
        if values:
            max_val = max(values)
            min_val = min(values)
            padding = (max_val - min_val) * 0.15 if max_val != min_val else 0.1
            y_max = max_val + padding
            y_min = min_val - padding if min_val < 0 else 0
        else:
            y_min, y_max = 0, 1

        fig.update_yaxes(
            title_text=runs.get_objective(inputs["objective_id"]).name,
            range=[y_min, y_max],
            row=1,
            col=1,
            gridcolor="#EEEEEE",
            zerolinecolor="black",
        )

        fig.update_yaxes(
            tickvals=list(range(num_params)),
            ticktext=[p for p in hp_names],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            title_text="Parameter",
            row=2,
            col=1,
        )

        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)

        return fig
