#  noqa: D400
"""
# PartialDependencies

This module provides utilities for generating Partial Dependency Plots (PDP).

Provided utilities include getting input and output layout (filtered or non-filtered),
processing the data and loading the outputs.

## Classes
    - PartialDependencies: Generate a Partial Dependency Plot (PDP).

## Constants
    GRID_POINTS_PER_AXIS : int
    SAMPLES_PER_HP : int
    MAX_SAMPLES : int
    MAX_SHOWN_SAMPLES : int
"""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from pyPDP.algorithms.pdp import PDP

from deepcave import config
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import Status
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, get_hyperparameter_ticks, save_image

GRID_POINTS_PER_AXIS = 20
SAMPLES_PER_HP = 10
MAX_SAMPLES = 10000
MAX_SHOWN_SAMPLES = 100


class PartialDependencies(StaticPlugin):
    """
    Generate Partial Dependency Plots (PDP).

    Provided utilities include getting input and output layout (filtered or non-filtered),
    processing the data and loading the outputs.
    """

    id = "pdp"
    name = "Partial Dependencies"
    icon = "fas fa-grip-lines"
    help = "docs/plugins/partial_dependencies.rst"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            The layout for the input block.
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
                                "Combined budget means that the trial on the highest"
                                " evaluated budget is used.  \n "
                                "Note: Selecting combined budget might be misleading if"
                                " a time objective is used. Often, higher budget take "
                                " longer to evaluate, which might negatively influence "
                                " the results."
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter #1"),
                            dbc.Select(
                                id=register("hyperparameter_name_1", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter #2"),
                            dbc.Select(
                                id=register("hyperparameter_name_2", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
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
            The layout for the filter block.
        """
        return [
            dbc.Row(
                [
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
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Show ICE curves"),
                                    help_button(
                                        "Displays the ICE curves from which the PDP curve is "
                                        "derivied."
                                    ),
                                    dbc.Select(id=register("show_ice", ["value", "options"])),
                                ]
                            )
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
            "show_confidence": {"options": get_select_options(binary=True), "value": "true"},
            "show_ice": {"options": get_select_options(binary=True), "value": "true"},
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs) -> Dict[str, Any]:  # type: ignore # noqa: E501
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned. The returned dictionary
        will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Current content of the inputs.
        previous_inputs
            Previous content of the inputs.
            Not used in this specific function.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the changes.
        """
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)

        hp_names = run.configspace.get_hyperparameter_names()

        # Get selected values
        objective_value = inputs["objective_id"]["value"]
        budget_value = inputs["budget_id"]["value"]
        hp1_value = inputs["hyperparameter_name_1"]["value"]

        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
            hp1_value = hp_names[0]

        return {
            "objective_id": {"options": objective_options, "value": objective_value},
            "budget_id": {"options": budget_options, "value": budget_value},
            "hyperparameter_name_1": {
                "options": get_checklist_options(hp_names),
                "value": hp1_value,
            },
            "hyperparameter_name_2": {
                "options": get_checklist_options([None] + hp_names),
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and the input data.

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
        run
            The run to process.
        inputs
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.

        Raises
        ------
        RuntimeError
            If the objective is None.
        """
        # Surrogate
        hp_names = run.configspace.get_hyperparameter_names()
        objective = run.get_objective(inputs["objective_id"])
        budget = run.get_budget(inputs["budget_id"])
        hp1 = inputs["hyperparameter_name_1"]
        hp2 = inputs["hyperparameter_name_2"]

        if objective is None:
            raise RuntimeError("Objective not found.")

        # Encode data
        df = run.get_encoded_data(
            objective,
            budget,
            specific=True,
            statuses=Status.SUCCESS,
        )

        X = df[hp_names].to_numpy()
        Y = df[objective.name].to_numpy()

        # Let's initialize the surrogate
        surrogate_model = RandomForestSurrogate(run.configspace, seed=0)
        surrogate_model.fit(X, Y)

        # Prepare the hyperparameters
        selected_hyperparameters = [hp1]
        if hp2 is not None and hp2 != "":
            selected_hyperparameters += [hp2]

        num_samples = SAMPLES_PER_HP * len(X)
        # The samples are limited to max 10k
        if num_samples > MAX_SAMPLES:
            num_samples = MAX_SAMPLES

        # And finally call PDP
        pdp = PDP.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            seed=0,
            num_grid_points_per_axis=GRID_POINTS_PER_AXIS,
            num_samples=num_samples,
        )

        x = pdp.x_pdp.tolist()
        y = pdp.y_pdp.tolist()

        # The ICE curves have to be cut because it's too much data
        x_ice = pdp._ice.x_ice.tolist()
        y_ice = pdp._ice.y_ice.tolist()

        if len(x_ice) > MAX_SHOWN_SAMPLES:
            x_ice = x_ice[:MAX_SHOWN_SAMPLES]
            y_ice = y_ice[:MAX_SHOWN_SAMPLES]

        return {
            "x": x,
            "y": y,
            "variances": pdp.y_variances.tolist(),
            "x_ice": x_ice,
            "y_ice": y_ice,
        }

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
    def get_pdp_figure(  # type: ignore
        run, inputs, outputs, show_confidence, show_ice, title=None, fontsize=None
    ) -> go.Figure:
        """
        Create a figure of the Partial Dependency Plot (PDP).

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.
        show_confidence
            Whether to show confidence in the plot.
        show_ice
            Whether to show ice curves in the plot.
        title
            Title of the plot.
        fontsize
            Fontsize of the plot.

        Returns
        -------
        go.Figure
            The figure of the Partial Dependency Plot (PDP).
        """
        # Parse inputs
        hp1_name = inputs["hyperparameter_name_1"]
        hp1_idx = run.configspace.get_idx_by_hyperparameter_name(hp1_name)
        hp1 = run.configspace.get_hyperparameter(hp1_name)

        hp2_name = inputs["hyperparameter_name_2"]
        hp2_idx = None
        hp2 = None
        if hp2_name is not None and hp2_name != "":
            hp2_idx = run.configspace.get_idx_by_hyperparameter_name(hp2_name)
            hp2 = run.configspace.get_hyperparameter(hp2_name)

        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        # Parse outputs
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        sigmas = np.sqrt(np.asarray(outputs["variances"]))

        x_ice = np.asarray(outputs["x_ice"])
        y_ice = np.asarray(outputs["y_ice"])

        traces = []
        if hp2_idx is None:  # 1D
            # Add ICE curves
            if show_ice:
                for x_, y_ in zip(x_ice, y_ice):
                    traces += [
                        go.Scatter(
                            x=x_[:, hp1_idx],
                            y=y_,
                            line=dict(color=get_color(1, 0.1)),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    ]

            if show_confidence:
                traces += [
                    go.Scatter(
                        x=x[:, hp1_idx],
                        y=y + sigmas,
                        line=dict(color=get_color(0, 0.1)),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                ]

                traces += [
                    go.Scatter(
                        x=x[:, hp1_idx],
                        y=y - sigmas,
                        fill="tonexty",
                        fillcolor=get_color(0, 0.2),
                        line=dict(color=get_color(0, 0.1)),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                ]

            traces += [
                go.Scatter(
                    x=x[:, hp1_idx],
                    y=y,
                    line=dict(color=get_color(0, 1)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            ]

            tickvals, ticktext = get_hyperparameter_ticks(hp1)
            # Allow to pass a fontsize (necessary when leveraging PDP in Symbolic Explanation)
            if fontsize is None:
                fontsize = config.FIGURE_FONT_SIZE
            layout = go.Layout(
                {
                    "xaxis": {
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "title": hp1_name,
                    },
                    "yaxis": {
                        "title": objective_name,
                    },
                    "title": title,
                    "font": dict(size=fontsize),
                }
            )
        else:
            z = y
            if show_confidence:
                z = sigmas
            traces += [
                go.Contour(
                    z=z,
                    x=x[:, hp1_idx],
                    y=x[:, hp2_idx],
                    colorbar=dict(
                        title=objective_name if not show_confidence else "Confidence (1-Sigma)",
                    ),
                    hoverinfo="skip",
                )
            ]

            x_tickvals, x_ticktext = get_hyperparameter_ticks(hp1)
            y_tickvals, y_ticktext = get_hyperparameter_ticks(hp2)

            layout = go.Layout(
                dict(
                    xaxis=dict(tickvals=x_tickvals, ticktext=x_ticktext, title=hp1_name),
                    yaxis=dict(tickvals=y_tickvals, ticktext=y_ticktext, title=hp2_name),
                    margin=config.FIGURE_MARGIN,
                    title=title,
                    font=dict(size=fontsize),
                )
            )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "pdp.pdf")

        return figure

    @staticmethod
    def load_outputs(run, inputs, outputs):  # type: ignore
        """
        Read the raw data and prepare it for the layout.

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
            The figure of the Partial Dependency Plot (PDP).
        """
        show_confidence = inputs["show_confidence"]
        show_ice = inputs["show_ice"]

        figure = PartialDependencies.get_pdp_figure(run, inputs, outputs, show_confidence, show_ice)

        return figure
