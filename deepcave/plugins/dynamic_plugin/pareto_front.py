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
        self.readable_budgets = run.get_budgets(human=True)
        self.objectives = run.get_objectives()
        self.objective_names = run.get_objective_names()
        self.objective_ids = list(range(len(self.objective_names)))

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
                    dcc.Slider(id=register("budget", ["min", "max", "marks", "value"])),
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
                "options": get_select_options(self.objective_names, self.objective_ids),
                "value": self.objective_ids[0],
            },
            "objective2": {
                "options": get_select_options(self.objective_names, self.objective_ids),
                "value": self.objective_ids[0],
            },
            "budget": {
                "min": 0,
                "max": len(self.readable_budgets) - 1,
                "marks": get_slider_marks(self.readable_budgets),
                "value": len(self.readable_budgets) - 1,
            },
            "all_configs": {
                "options": get_select_options(binary=True),
                "value": False,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, List[Union[float, str]]]:
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)
        objectives = run.get_objectives()

        o1_idx = int(inputs["objective1"]["value"])
        o2_idx = int(inputs["objective2"]["value"])
        o1 = objectives[o1_idx]
        o2 = objectives[o2_idx]

        points: Union[List, np.ndarray] = []
        config_ids: Union[List, np.ndarray] = []
        for config_id, costs in run.get_costs(
            budget, statuses=[Status.SUCCESS]
        ).items():
            points += [[costs[o1_idx], costs[o2_idx]]]
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
                for idx, (objective, cost) in enumerate(zip([o1, o2], costs)):
                    if objective["optimize"] == "upper":
                        select2 = np.any(
                            points[is_front][:, idx, np.newaxis] > [cost], axis=1
                        )
                    else:
                        select2 = np.any(
                            points[is_front][:, idx, np.newaxis] < [cost], axis=1
                        )

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
            optimize1 = self.objectives[int(inputs["objective1"]["value"])]["optimize"]
            optimize2 = self.objectives[int(inputs["objective2"]["value"])]["optimize"]

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
            xaxis=dict(title=self.objective_names[int(inputs["objective1"]["value"])]),
            yaxis=dict(title=self.objective_names[int(inputs["objective2"]["value"])]),
        )

        return [go.Figure(data=traces, layout=layout)]
