from typing import Dict, Union, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.utils.layout import (
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.styled_plotty import get_color
from deepcave.runs import Status


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

        budgets = run.get_budgets(human=True)
        self.budget_options = get_select_options(budgets, range(len(budgets)))

        objective_names = run.get_objective_names()
        self.objective_options = get_select_options(objective_names)

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective #1"),
                    dbc.Select(
                        id=register("objective1", ["options", "value"]),
                        placeholder="Select objective ...",
                        className="mb-3",
                    ),
                    dbc.Label("Objective #2"),
                    dbc.Select(
                        id=register("objective2", ["options", "value"]),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budget"),
                    dbc.Select(
                        id=register("budget", ["options", "value"]),
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
                    dbc.RadioItems(id=register("all_configs", ["options", "value"])),
                ],
            ),
        ]

    def load_inputs(self):
        return {
            "objective1": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "objective2": {
                "options": self.objective_options,
                "value": self.objective_options[-1]["value"],
            },
            "budget": {
                "options": self.budget_options,
                "value": self.budget_options[0]["value"],
            },
            "all_configs": {
                "options": get_select_options(binary=True),
                "value": False,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, List[Union[float, str]]]:
        # Get budget
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(int(budget_id))

        # Get objectives
        objective1 = run.get_objective(inputs["objective1"]["value"])
        objective2 = run.get_objective(inputs["objective2"]["value"])
        objective1_id = run.get_objective_id(objective1)
        objective2_id = run.get_objective_id(objective2)

        points: Union[List, np.ndarray] = []
        config_ids: Union[List, np.ndarray] = []
        for config_id, costs in run.get_costs(budget, statuses=[Status.SUCCESS]).items():
            points += [[costs[objective1_id], costs[objective2_id]]]
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
                for idx, (objective, cost) in enumerate(zip([objective1, objective2], costs)):
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
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, runs):

        traces = []
        for idx, (run_name, run) in enumerate(runs.items()):
            points = np.array(outputs[run.name]["points"])

            x, y = [], []
            x_pareto, y_pareto = [], []

            pareto_points = outputs[run.name]["pareto_points"]
            for point_idx, pareto in enumerate(pareto_points):
                if pareto:
                    x_pareto += [points[point_idx][0]]
                    y_pareto += [points[point_idx][1]]
                else:
                    x += [points[point_idx][0]]
                    y += [points[point_idx][1]]

            # And get configs for the hovers
            hovertext = []
            for config_id in outputs[run.name]["config_ids"]:
                config = run.get_config(config_id)

                text = f"<br>Config ID: {config_id}<br>"
                for k, v in config.items():
                    text += f"{k}: {v}<br>"
                hovertext += [text]

            color = get_color(idx, alpha=0.1)
            color_pareto = get_color(idx)

            if inputs["all_configs"]["value"]:
                traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=run_name,
                        mode="markers",
                        showlegend=False,
                        line=dict(color=color),
                    )
                )

            # Check if we need hv or vh
            optimize1 = run.get_objective(inputs["objective1"]["value"])["optimize"]
            optimize2 = run.get_objective(inputs["objective2"]["value"])["optimize"]

            if optimize1 == optimize2:
                line_shape = "vh"
            else:
                line_shape = "hv"

            traces.append(
                go.Scatter(
                    x=x_pareto,
                    y=y_pareto,
                    name=run_name,
                    line_shape=line_shape,
                    showlegend=True,
                    line=dict(color=color_pareto),
                    hovertext=hovertext,
                )
            )

        layout = go.Layout(
            xaxis=dict(title=inputs["objective1"]["value"]),
            yaxis=dict(title=inputs["objective2"]["value"]),
        )

        return [go.Figure(data=traces, layout=layout)]
