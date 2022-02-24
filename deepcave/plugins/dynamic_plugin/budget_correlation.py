from typing import Dict

from collections import defaultdict

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from scipy import stats
from dash.exceptions import PreventUpdate

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import Status
from deepcave.utils.data_structures import update_dict
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

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective", ["options", "value"]),
                        placeholder="Select objective ...",
                    ),
                ],
            ),
        ]

    def load_inputs(self):
        return {
            "objective": {"options": get_select_options(), "value": None},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run):
        objective_names = selected_run.get_objective_names()
        objective_ids = list(range(len(objective_names)))

        value = inputs["objective"]["value"]
        if value is None:
            value = objective_ids[0]

        new_inputs = {
            "objective": {
                "options": get_select_options(objective_names, objective_ids),
                "value": value,
            },
        }
        update_dict(inputs, new_inputs)

        return inputs

    @staticmethod
    def process(run, inputs):
        objective_id = int(inputs["objective"]["value"])

        correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        for budget1 in run.get_budgets():
            for budget2 in run.get_budgets():
                # Exclude if budget2 is higher than budget1
                if budget2 > budget1:
                    continue

                costs1 = run.get_costs(budget1, statuses=[Status.SUCCESS])
                costs2 = run.get_costs(budget2, statuses=[Status.SUCCESS])

                # Combine config ids
                # So we make sure we have the same number of configs for each budget
                config_ids = set(costs1.keys()) & set(costs2.keys())

                c1, c2 = [], []
                for config_id in config_ids:
                    c1 += [costs1[config_id][objective_id]]
                    c2 += [costs2[config_id][objective_id]]

                correlations[budget1][budget2] = round(
                    stats.spearmanr(c1, c2).correlation, 2
                )

        return {"correlations": correlations}

    @staticmethod
    def get_output_layout(register):
        return [
            dcc.Graph(register("graph", "figure")),
        ]

    def load_outputs(self, inputs, outputs, run):
        if len(run.get_budgets()) == 1:
            self.update_alert("Only on budget found but need at least two.")
            raise PreventUpdate()

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
        )

        return [go.Figure(data=traces, layout=layout)]
