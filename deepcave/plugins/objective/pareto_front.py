#  noqa: D400
"""
# ParetoFront

This module provides utilities for creating a visualization of the Pareto Front.

It includes the corresponding Pareto Front plugin.

## Classes
    - ParetoFront: Generate an interactive Pareto Front visualization.
"""

from typing import Any, Callable, Dict, List, Literal, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config, notification
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status, check_equality
from deepcave.runs.exceptions import NotMergeableError, RunInequality
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plot import plt
from deepcave.utils.styled_plotty import (
    get_color,
    get_hovertext_from_config,
    save_image,
)


class ParetoFront(DynamicPlugin):
    """
    Generate an interactive Pareto Front visualization.

    Properties
    ----------
    objective_options : List[Dict[str, Any]]
        A list of the objective options.
    budget_options : List[Dict[str, Any]]
        A list of the budget options.
    """

    id = "pareto_front"
    name = "Pareto Front"
    icon = "fas fa-wind"
    help = "docs/plugins/pareto_front.rst"

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """
        Check if the runs are compatible.

        This function is needed if all selected runs need something in common
        (e.g. budget or objective).
        Since this function is called before the layout is created,
        it can be also used to set common values for the plugin.

        If the runs are not mergeable, they still should be displayed
        but with a corresponding warning message

        Parameters
        ----------
        runs : List[AbstractRun]
            A list containing the selected runs.

        Raises
        ------
        NotMergeableError
            If the meta data of the runs are not equal.
            If the configuration spaces of the runs are not equal.
            If the budgets of the runs are not equal.
            If the objective of the runs are not equal.
        """
        try:
            check_equality(runs, objectives=True, budgets=True)
        except NotMergeableError as e:
            run_inequality = e.args[1]
            if run_inequality == RunInequality.INEQ_BUDGET:
                notification.update("The budgets of the runs are not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_CONFIGSPACE:
                notification.update(
                    "The configuration spaces of the runs are not equal.", color="warning"
                )
            elif run_inequality == RunInequality.INEQ_META:
                notification.update("The meta data of the runs is not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_OBJECTIVE:
                raise NotMergeableError("The objectives of the selected runs cannot be merged.")

        # Set some attributes here
        # It is necessary to get the run with the smallest budget and objective options
        # as first comparative value, else there is gonna be an index problem
        objective_options = []
        budget_options = []
        for run in runs:
            objective_names = run.get_objective_names()
            objective_ids = run.get_objective_ids()
            objective_options.append(get_select_options(objective_names, objective_ids))

            budgets = run.get_budgets(human=True)
            budget_ids = run.get_budget_ids()
            budget_options.append(get_select_options(budgets, budget_ids))
        self.objective_options = min(objective_options, key=len)
        self.budget_options = min(budget_options, key=len)

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
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
                            dbc.Label("Objective #1"),
                            dbc.Select(
                                id=register("objective_id_1", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Objective #2"),
                            dbc.Select(
                                id=register("objective_id_2", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budget"),
                    help_button(
                        "Budget refers to the multi-fidelity budget. "
                        "Combined budget means that the trial on the highest"
                        " evaluated budget is used.  \n "
                        "Note: Selecting combined budget might be misleading if a time objective "
                        "is used. Often, higher budget take longer to evaluate, which might "
                        "negatively influence the results."
                    ),
                    dbc.Select(
                        id=register("budget_id", ["value", "options"], type=int),
                        placeholder="Select budget ...",
                    ),
                ],
                className="",
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
                            dbc.Label("Show all configurations"),
                            help_button(
                                "Additionally to the pareto front, also the other configurations "
                                "are displayed. This makes it easier to see the performance "
                                "differences."
                            ),
                            dbc.Select(
                                id=register("show_all", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show error bars"),
                            help_button(
                                "Show error bars In the case of non-deterministic runs with "
                                "multiple seeds evaluated per configuration."
                            ),
                            dbc.Select(
                                id=register("show_error", ["value", "options"]),
                                placeholder="Select ...",
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
                            dbc.Label("Show Runs"),
                            dbc.Select(
                                id=register("show_runs", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Groups"),
                            dbc.Select(
                                id=register("show_groups", ["value", "options"]),
                                placeholder="Select ...",
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
        So, if the plugin is called for the first time or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The content to be filled.
        """
        return {
            "objective_id_1": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "objective_id_2": {
                "options": self.objective_options,
                "value": self.objective_options[-1]["value"],
            },
            "budget_id": {
                "options": self.budget_options,
                "value": self.budget_options[-1]["value"],
            },
            "show_all": {"options": get_select_options(binary=True), "value": "false"},
            "show_error": {"options": get_select_options(binary=True), "value": "false"},
            "show_runs": {"options": get_select_options(binary=True), "value": "true"},
            "show_groups": {"options": get_select_options(binary=True), "value": "true"},
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
        # Get budget
        budget = run.get_budget(inputs["budget_id"])

        # Get objectives
        objective_id_1 = inputs["objective_id_1"]
        objective_1 = run.get_objective(objective_id_1)
        objective_id_2 = inputs["objective_id_2"]
        objective_2 = run.get_objective(objective_id_2)

        points_avg: Union[List, np.ndarray] = []
        points_std: Union[List, np.ndarray] = []
        config_ids: Union[List, np.ndarray] = list(
            run.get_configs(budget, statuses=[Status.SUCCESS]).keys()
        )

        for config_id in config_ids:
            avg_costs, std_costs = run.get_avg_costs(config_id, budget, statuses=[Status.SUCCESS])
            points_avg += [[avg_costs[objective_id_1], avg_costs[objective_id_2]]]
            points_std += [[std_costs[objective_id_1], std_costs[objective_id_2]]]

        points_avg = np.array(points_avg)
        points_std = np.array(points_std)
        config_ids = np.array(config_ids)

        # Sort the points s.t. x axis is monotonically increasing
        sorted_idx = np.argsort(points_avg[:, 0])
        points_avg = points_avg[sorted_idx]
        points_std = points_std[sorted_idx]
        config_ids = config_ids[sorted_idx]

        is_front: np.ndarray = np.ones(points_avg.shape[0], dtype=bool)
        for point_idx, costs in enumerate(points_avg):
            if is_front[point_idx]:
                # Keep any point with a lower/upper cost
                # This loop is a little bit complicated than
                # is_front[is_front] = np.any(points[is_front] < c, axis=1)
                # because objectives can be optimized in different directions.
                # Therefore it has to be checked for each objective separately.
                select = None
                for idx, (objective, cost) in enumerate(zip([objective_1, objective_2], costs)):
                    if objective.optimize == "upper":
                        select2 = np.any(points_avg[is_front][:, idx, np.newaxis] > [cost], axis=1)
                    else:
                        select2 = np.any(points_avg[is_front][:, idx, np.newaxis] < [cost], axis=1)

                    if select is None:
                        select = select2
                    else:
                        select = np.logical_or(select, select2)

                is_front[is_front] = select

                # And keep self
                is_front[point_idx] = True

        return {
            "points_avg": points_avg.tolist(),
            "points_std": points_std.tolist(),
            "pareto_points": is_front.tolist(),
            "config_ids": config_ids.tolist(),
        }

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
            The layout for the output block.
        """
        return dcc.Graph(
            register("graph", "figure"),
            style={"height": config.FIGURE_HEIGHT},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(runs, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        runs :
            The selected runs.
        inputs :
            The input and filter values from the user.
        outputs :
            Raw outputs from the runs.

        Returns
        -------
        go.Figure
            The output figure.
        """
        show_all = inputs["show_all"]
        show_error = inputs["show_error"]

        traces = []
        for idx, run in enumerate(runs):
            show_runs = inputs["show_runs"]
            show_groups = inputs["show_groups"]

            if run.prefix == "group" and not show_groups:
                continue

            if run.prefix != "group" and not show_runs:
                continue

            points_avg = np.array(outputs[run.id]["points_avg"])
            points_std = np.array(outputs[run.id]["points_std"])
            config_ids = outputs[run.id]["config_ids"]
            budget = run.get_budget(inputs["budget_id"])
            pareto_config_ids = []

            x, y, x_std, y_std = [], [], [], []
            x_pareto, y_pareto, x_pareto_std, y_pareto_std = [], [], [], []

            pareto_points = outputs[run.id]["pareto_points"]
            for point_idx, pareto in enumerate(pareto_points):
                if pareto:
                    x_pareto += [points_avg[point_idx][0]]
                    y_pareto += [points_avg[point_idx][1]]
                    x_pareto_std += [points_std[point_idx][0]]
                    y_pareto_std += [points_std[point_idx][1]]
                    pareto_config_ids += [config_ids[point_idx]]
                else:
                    x += [points_avg[point_idx][0]]
                    y += [points_avg[point_idx][1]]
                    x_std += [points_std[point_idx][0]]
                    y_std += [points_std[point_idx][1]]

            color = get_color(idx, alpha=0.5)
            color_pareto = get_color(idx)

            if show_all:
                error_x = (
                    dict(array=x_std, color="rgba(0, 0, 0, 0.3)")
                    if show_error and not all(value == 0.0 for value in x_std)
                    else None
                )
                error_y = (
                    dict(array=y_std, color="rgba(0, 0, 0, 0.3)")
                    if show_error and not all(value == 0.0 for value in y_std)
                    else None
                )

                traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        error_x=error_x,
                        error_y=error_y,
                        name=run.name,
                        mode="markers",
                        showlegend=False,
                        line=dict(color=color),
                        hoverinfo="skip",
                    )
                )

            # Check if hv or vh is needed
            objective_1 = run.get_objective(inputs["objective_id_1"])
            objective_2 = run.get_objective(inputs["objective_id_2"])
            optimize1 = objective_1.optimize
            optimize2 = objective_2.optimize

            if optimize1 == optimize2:
                if objective_1.optimize == "lower":
                    line_shape = "hv"
                else:
                    line_shape = "vh"
            else:
                line_shape = "hv"

            hovertext = [
                get_hovertext_from_config(run, config_id, budget) for config_id in pareto_config_ids
            ]

            error_pareto_x = (
                dict(array=x_pareto_std, color="rgba(0, 0, 0, 0.3)")
                if show_error and not all(value == 0.0 for value in x_pareto_std)
                else None
            )
            error_pareto_y = (
                dict(array=y_pareto_std, color="rgba(0, 0, 0, 0.3)")
                if show_error and not all(value == 0.0 for value in y_pareto_std)
                else None
            )

            traces.append(
                go.Scatter(
                    x=x_pareto,
                    y=y_pareto,
                    error_x=error_pareto_x,
                    error_y=error_pareto_y,
                    name=run.name,
                    line_shape=line_shape,
                    showlegend=True,
                    line=dict(color=color_pareto),
                    hovertext=hovertext,
                    hoverinfo="text",
                )
            )

        if len(traces) > 0:
            layout = go.Layout(
                xaxis=dict(title=objective_1.name),
                yaxis=dict(title=objective_2.name),
                margin=config.FIGURE_MARGIN,
                font=dict(size=config.FIGURE_FONT_SIZE),
            )
        else:
            layout = None

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "pareto_front.pdf")

        return figure

    @staticmethod
    def get_mpl_output_layout(register: Callable) -> html.Img:
        """
        Get the layout for the matplotlib output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        html.Img
            The layout for the matplotlib output block.
        """
        return html.Img(
            id=register("graph", "src"),
            className="img-fluid",
        )

    @staticmethod
    def load_mpl_outputs(runs, inputs, outputs):  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        runs :
            The selected runs.
        inputs :
            Input and filter values from the user.
        outputs :
            Raw outputs from the runs.

        Returns
        -------
        The rendered matplotlib figure.
        """
        show_all = inputs["show_all"] == "true"

        plt.figure()
        for idx, run in enumerate(runs):
            show_runs = inputs["show_runs"] == "true"
            show_groups = inputs["show_groups"] == "true"

            if run.prefix == "group" and not show_groups:
                continue

            if run.prefix != "group" and not show_runs:
                continue

            points = np.array(outputs[run.id]["points"])

            x, y = [], []
            x_pareto, y_pareto = [], []

            pareto_points = outputs[run.id]["pareto_points"]
            for point_idx, pareto in enumerate(pareto_points):
                if pareto:
                    x_pareto += [points[point_idx][0]]
                    y_pareto += [points[point_idx][1]]
                else:
                    x += [points[point_idx][0]]
                    y += [points[point_idx][1]]

            color = plt.get_color(idx)  # type: ignore
            color_pareto = plt.get_color(idx)  # type: ignore

            if show_all:
                plt.scatter(x, y, color=color, marker="o", s=3)

            # Check if hv or vh is needed
            objective_1 = run.get_objective(inputs["objective_id_1"])
            objective_2 = run.get_objective(inputs["objective_id_2"])
            optimize1 = objective_1.optimize
            optimize2 = objective_2.optimize

            line_shape: Union[Literal["post"], Literal["pre"], Literal["mid"]]
            if optimize1 == optimize2:
                if objective_1.optimize == "lower":
                    line_shape = "post"
                else:
                    line_shape = "pre"
            else:
                line_shape = "post"

            plt.step(
                x_pareto,
                y_pareto,
                color=color_pareto,
                marker="o",
                label=run.name,
                linewidth=1,
                markersize=3,
                where=line_shape,
            )
            plt.xlabel(objective_1.name)
            plt.ylabel(objective_2.name)

        plt.legend()

        return plt.render()  # type: ignore
