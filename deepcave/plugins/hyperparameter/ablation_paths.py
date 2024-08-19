#  noqa: D400
"""
# Ablation Paths

This module provides a plugin for the visualization of the ablation paths.

Provided utilities include getting input and output layout (filtered or non-filtered),
processing the data and loading the outputs.

## Classes
    - Ablation_Paths: This class provides a plugin for the visualization of the ablation paths.
"""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.evaluators.ablation import Ablation
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.cast import optional_int
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, save_image


class AblationPaths(StaticPlugin):
    """
    Provide a plugin for the visualization of the ablation paths.

    Provided utilities include getting input/output layout, data processing
    and loading outputs.
    """

    id = "ablation_paths"
    name = "Ablation Paths"
    icon = "fas fa-sort-amount-up"
    help = "docs/plugins/ablation_paths.rst"
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
                            dbc.Label("Trees"),
                            help_button(
                                "The number of trees of the internal random forest to estimate the "
                                "ablation path importance. "
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
                                    dbc.Label("Limit Hyperparameters"),
                                    help_button(
                                        "Shows only the n most important (i.e. first selected) "
                                        "hyperparameters."
                                    ),
                                    dbc.Input(id=register("n_hps", "value"), type="number"),
                                ],
                                className="mb-3",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Show confidence"),
                                    help_button("Displays the confidence bands."),
                                    dbc.Select(
                                        id=register("show_confidence", ["value", "options"])
                                    ),
                                ]
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
            "n_trees": {"value": 100},
            "n_hps": {"value": 0},
            "budget_id": {"options": get_checklist_options(), "value": None},
            "show_confidence": {"options": get_select_options(binary=True), "value": "false"},
        }

    def load_dependency_inputs(self, run, _: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore # noqa: E501
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
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        hp_names = run.configspace.get_hyperparameter_names()
        n_hps = inputs["n_hps"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]

        if n_hps == 0:
            n_hps = len(hp_names)

        # Pre-selection of the hyperparameters
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
        """
        objective = run.get_objective(inputs["objective_id"])
        n_trees = inputs["n_trees"]

        if n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        budgets = run.get_budgets(include_combined=True)

        evaluator = Ablation(run)

        # Collect data
        data = {}
        for budget_id, budget in enumerate(budgets):
            assert isinstance(budget, (int, float))
            assert isinstance(budget, (int, float))
            evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)

            performances = evaluator.get_ablation_performances()
            improvements = evaluator.get_ablation_improvements()
            data[budget_id] = [performances, improvements]
        return data  # type: ignore

    @staticmethod
    def get_output_layout(register: Callable) -> List[dcc.Graph]:
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
            ),
            dcc.Graph(
                register("impr_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
        ]

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[go.Figure]:  # type: ignore
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
        return [figure1, figure2]
            The figures of the ablation paths.
        """
        # First selected, should always be shown first
        selected_budget_id = inputs["budget_id"]
        objective = run.get_objective(inputs["objective_id"])
        n_hps = inputs["n_hps"]
        show_confidence = inputs["show_confidence"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        # Collect data
        data1, data2 = {}, {}
        for budget_id, results in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id != selected_budget_id:
                continue

            x = []
            y1, y2 = [], []
            error_y1, error_y2 = [], []
            for hp_name, result in results[0].items():
                x += [hp_name]
                y1 += [result[0]]
                error_y1 += [result[1]]
            for _, result in results[1].items():
                y2 += [result[0]]
                error_y2 += [result[1]]

            data1[budget_id] = (np.array(x), np.array(y1), np.array(error_y1))
            data2[budget_id] = (np.array(x), np.array(y2), np.array(error_y2))

        bar_data1, bar_data2 = [], []

        for budget_id, values in data1.items():
            budget = run.get_budget(budget_id, human=True)

            x = list(values[0][:n_hps])

            bar_data1 += [
                go.Scatter(
                    name=budget,
                    x=x,
                    y=values[1][:n_hps],
                    error_y=dict(array=values[2][:n_hps]) if show_confidence else None,
                    line=dict(color=get_color(0)),
                )
            ]

        for budget_id, values in data2.items():
            budget = run.get_budget(budget_id, human=True)

            x = list(values[0][:n_hps])

            bar_data2 += [
                go.Bar(
                    name=budget,
                    x=x,
                    y=values[1][:n_hps],
                    error_y_array=values[2][:n_hps] if show_confidence else None,
                    marker_color=get_color(0),
                )
            ]

        figure1 = go.Figure(data=bar_data1)
        figure1.update_layout(
            barmode="group",
            title={
                "text": "Ablation Path when Iteratively Setting the Hyperparameters to Their "
                "Incumbent Value",
                "font": {"size": config.FIGURE_FONT_SIZE + 2},
            },
            yaxis_title=objective.name,
            legend={"title": "Budget"},
            margin=dict(t=50, b=0, l=0, r=0),
            xaxis=dict(tickangle=-45),
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        save_image(figure1, "ablation_path_performance.pdf")

        figure2 = go.Figure(data=bar_data2)
        figure2.update_layout(
            barmode="group",
            title={
                "text": "Change in Objective with Respect to the Previous Ablation Step",
                "font": {"size": config.FIGURE_FONT_SIZE + 2},
            },
            yaxis_title=f"Change in {objective.name}",
            legend={"title": "Budget"},
            margin=dict(t=50, b=0, l=0, r=0),
            xaxis=dict(tickangle=-45),
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        save_image(figure2, "ablation_path_improvement.pdf")

        return [figure1, figure2]
