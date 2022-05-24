from typing import Dict

from collections import defaultdict

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from scipy import stats

from deepcave import notification
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status
from deepcave.utils.layout import get_select_options
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color

logger = get_logger(__name__)


class BudgetCorrelation(DynamicPlugin):
    id = "budget_correlation"
    name = "Budget Correlation"
    icon = "fas fa-coins"
    description = """
        The budget correlation tells how much the costs of two budgets are correlated
        to each other. The budget shown in the legend is compared to all other (lower) budgets.
    """
    activate_run_selection = True
    help = "docs/plugins/budget_correlation.md"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        if len(run.get_budgets()) == 1:
            notification.update(f"{run.name} can not be selected because it has only one budget.")
            return False

        return True

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                ],
            ),
        ]

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)

        value = inputs["objective_id"]["value"]
        if value is None:
            value = objective_ids[0]

        return {
            "objective_id": {
                "options": objective_options,
                "value": value,
            },
        }

    @staticmethod
    def process(run, inputs):
        objective_id = inputs["objective_id"]

        correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        for budget1 in run.get_budgets(include_combined=False):
            for budget2 in run.get_budgets(include_combined=False):
                # Exclude if budget2 is higher than budget1
                if budget2 > budget1:
                    continue

                costs1 = run.get_all_costs(budget1, statuses=[Status.SUCCESS])
                costs2 = run.get_all_costs(budget2, statuses=[Status.SUCCESS])

                # Combine config ids
                # So we make sure we have the same number of configs for each budget
                config_ids = set(costs1.keys()) & set(costs2.keys())

                c1, c2 = [], []
                for config_id in config_ids:
                    c1 += [costs1[config_id][objective_id]]
                    c2 += [costs2[config_id][objective_id]]

                correlations[budget1][budget2] = round(stats.spearmanr(c1, c2).correlation, 2)

        return {"correlations": correlations}

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(id=register("graph", "figure"))

    @staticmethod
    def load_outputs(run, inputs, outputs):
        traces = []

        correlations = outputs["correlations"]
        for idx, (budget1, budgets) in enumerate(correlations.items()):
            x, y = [], []
            for budget2, correlation in budgets.items():
                x += [float(budget2)]
                y += [correlation]

            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    name=budget1,
                    line=dict(color=get_color(idx)),
                )
            )

        layout = go.Layout(
            xaxis=dict(title="Budget"),
            yaxis=dict(title="Correlation"),
            margin=dict(
                t=30,
                b=0,
                l=0,
                r=0,
            ),
        )

        return [go.Figure(data=traces, layout=layout)]
