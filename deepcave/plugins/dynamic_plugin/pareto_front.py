from typing import Union, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun, NotMergeableError, check_equality
from deepcave.utils.layout import (
    get_radio_options,
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
                "value": 0,
            },
        }

    @staticmethod
    def process(run, inputs) -> dict[str, list[Union[float, str]]]:
        budget_id = inputs["budget"]["value"]
        budget = run.get_budget(budget_id)

        o1_idx = int(inputs["objective1"]["value"])
        o2_idx = int(inputs["objective2"]["value"])

        points = []
        for config_id, costs in run.get_costs(
            budget, statuses=[Status.SUCCESS]
        ).items():
            points += [[costs[o1_idx], costs[o2_idx]]]

        points = np.array(points)

        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)

                # And keep self
                is_efficient[i] = True

        return {"points": points.tolist(), "pareto_points": is_efficient.tolist()}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, runs: dict[str, AbstractRun]) -> list[Component]:

        traces = []
        for idx, (run_name, run) in enumerate(runs.items()):
            points = outputs[run.name]["points"]
            pareto_points = outputs[run.name]["pareto_points"]

            for ((x, y), pareto) in zip(points, pareto_points):
                color = get_color(idx, alpha=0.1)
                if pareto:
                    color = get_color(idx)

                traces.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        name=run_name,
                        # line_shape="hv",
                        line=dict(color=color),
                    )
                )

        layout = go.Layout(
            xaxis=dict(title=inputs["objective1"]["value"]),
            yaxis=dict(title=inputs["objective2"]["value"]),
        )

        return [go.Figure(data=traces, layout=layout)]
