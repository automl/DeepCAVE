from typing import Dict, List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status, check_equality
from deepcave.utils.layout import get_select_options, get_slider_marks
from deepcave.utils.styled_plotty import get_color, get_hovertext_from_config


class ParetoFront(DynamicPlugin):
    id = "pareto_front"
    name = "Pareto Front"
    icon = "fas fa-wind"
    description = """
        Pareto efficiency or Pareto optimality is a situation where no individual or preference
        criterion can be better off without making at least one individual or preference
        criterion worse off or without any loss thereof.
    """

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
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
                    dbc.RadioItems(id=register("show_all", ["value", "options"])),
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
                "value": self.budget_options[0]["value"],
            },
            "show_all": {
                "options": get_select_options(binary=True),
                "value": False,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, List[Union[float, str]]]:
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
                    if objective["optimize"] == "upper":
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
        return dcc.Graph(register("graph", "figure"))

    @staticmethod
    def load_outputs(runs, inputs, outputs):

        traces = []
        for idx, run in enumerate(runs):
            points = np.array(outputs[run.id]["points"])
            config_ids = outputs[run.id]["config_ids"]

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

            color = get_color(idx, alpha=0.1)
            color_pareto = get_color(idx)

            if inputs["show_all"]:
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
            optimize1 = objective_1["optimize"]
            optimize2 = objective_2["optimize"]

            if optimize1 == optimize2:
                line_shape = "vh"
            else:
                line_shape = "hv"

            hovertext = [get_hovertext_from_config(run, config_id) for config_id in config_ids]

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

        layout = go.Layout(
            xaxis=dict(title=objective_1["name"]),
            yaxis=dict(title=objective_2["name"]),
        )

        return [go.Figure(data=traces, layout=layout)]
