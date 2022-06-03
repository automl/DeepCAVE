from typing import List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import Status, check_equality
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plot import plt
from deepcave.utils.styled_plotty import (
    get_color,
    get_hovertext_from_config,
    save_image,
)


class ParetoFront(DynamicPlugin):
    id = "pareto_front"
    name = "Pareto Front"
    icon = "fas fa-wind"
    help = "docs/plugins/pareto_front.rst"

    def check_runs_compatibility(self, runs):
        check_equality(runs, objectives=True, budgets=True)

        # Set some attributes here
        run = runs[0]

        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        self.objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        self.budget_options = get_select_options(budgets, budget_ids)

    @staticmethod
    def get_input_layout(register):
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
                        "Combined budget means that the trial on the highest evaluated budget is "
                        "used.\n\n"
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
    def get_filter_layout(register):
        return [
            html.Div(
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
                className="mb-3",
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

    def load_inputs(self):
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
            "show_runs": {"options": get_select_options(binary=True), "value": "true"},
            "show_groups": {"options": get_select_options(binary=True), "value": "true"},
        }

    @staticmethod
    def process(run, inputs):
        # Get budget
        budget = run.get_budget(inputs["budget_id"])

        # Get objectives
        objective_id_1 = inputs["objective_id_1"]
        objective_1 = run.get_objective(objective_id_1)
        objective_id_2 = inputs["objective_id_2"]
        objective_2 = run.get_objective(objective_id_2)

        points: Union[List, np.ndarray] = []
        config_ids: Union[List, np.ndarray] = []
        for config_id, costs in run.get_all_costs(budget, statuses=[Status.SUCCESS]).items():
            points += [[costs[objective_id_1], costs[objective_id_2]]]
            config_ids += [config_id]

        points = np.array(points)
        config_ids = np.array(config_ids)

        # Sort the points s.t. x axis is monotonically increasing
        sorted_idx = np.argsort(points[:, 0])
        points = points[sorted_idx]
        config_ids = config_ids[sorted_idx]

        is_front: Union[List, np.ndarray] = np.ones(points.shape[0], dtype=bool)
        for point_idx, costs in enumerate(points):

            if is_front[point_idx]:
                # Keep any point with a lower/upper cost
                # This loop is a little bit complicated than
                # is_front[is_front] = np.any(points[is_front] < c, axis=1)
                # because objectives can be optimized in different directions.
                # We therefore have to check for each objective separately.
                select = None
                for idx, (objective, cost) in enumerate(zip([objective_1, objective_2], costs)):
                    if objective.optimize == "upper":
                        select2 = np.any(points[is_front][:, idx, np.newaxis] > [cost], axis=1)
                    else:
                        select2 = np.any(points[is_front][:, idx, np.newaxis] < [cost], axis=1)

                    if select is None:
                        select = select2
                    else:
                        select = np.logical_or(select, select2)

                is_front[is_front] = select

                # And keep self
                is_front[point_idx] = True

        return {
            "points": points.tolist(),
            "pareto_points": is_front.tolist(),
            "config_ids": config_ids.tolist(),
        }

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    def load_outputs(runs, inputs, outputs):
        show_all = inputs["show_all"]

        traces = []
        for idx, run in enumerate(runs):
            show_runs = inputs["show_runs"]
            show_groups = inputs["show_groups"]

            if run.prefix == "group" and not show_groups:
                continue

            if run.prefix != "group" and not show_runs:
                continue

            points = np.array(outputs[run.id]["points"])
            config_ids = outputs[run.id]["config_ids"]
            pareto_config_ids = []

            x, y = [], []
            x_pareto, y_pareto = [], []

            pareto_points = outputs[run.id]["pareto_points"]
            for point_idx, pareto in enumerate(pareto_points):
                if pareto:
                    x_pareto += [points[point_idx][0]]
                    y_pareto += [points[point_idx][1]]
                    pareto_config_ids += [config_ids[point_idx]]
                else:
                    x += [points[point_idx][0]]
                    y += [points[point_idx][1]]

            color = get_color(idx, alpha=0.1)
            color_pareto = get_color(idx)

            if show_all:
                traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=run.name,
                        mode="markers",
                        showlegend=False,
                        line=dict(color=color),
                        hoverinfo="skip",
                    )
                )

            # Check if we need hv or vh
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
                get_hovertext_from_config(run, config_id) for config_id in pareto_config_ids
            ]

            traces.append(
                go.Scatter(
                    x=x_pareto,
                    y=y_pareto,
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
            )
        else:
            layout = None

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "pareto_front.pdf")

        return figure

    @staticmethod
    def get_mpl_output_layout(register):
        return html.Img(
            id=register("graph", "src"),
            className="img-fluid",
        )

    @staticmethod
    def load_mpl_outputs(runs, inputs, outputs):
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

            color = plt.get_color(idx)  # , alpha=0.1)
            color_pareto = plt.get_color(idx)

            if show_all:
                plt.scatter(x, y, color=color, marker="o", alpha=0.1, s=3)

            # Check if we need hv or vh
            objective_1 = run.get_objective(inputs["objective_id_1"])
            objective_2 = run.get_objective(inputs["objective_id_2"])
            optimize1 = objective_1.optimize
            optimize2 = objective_2.optimize

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

        return plt.render()
