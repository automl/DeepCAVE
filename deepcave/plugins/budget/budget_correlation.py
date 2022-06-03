from typing import Dict

from collections import defaultdict

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from scipy import stats

from deepcave import config, notification
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import Status
from deepcave.utils.layout import create_table, get_select_options
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color, save_image

logger = get_logger(__name__)


class BudgetCorrelation(DynamicPlugin):
    id = "budget_correlation"
    name = "Budget Correlation"
    icon = "fas fa-coins"
    help = "docs/plugins/budget_correlation.rst"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run) -> bool:
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

    def load_dependency_inputs(self, run, _, inputs):
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
        budget_ids = run.get_budget_ids(include_combined=False)

        # Add symmetric correlations; table ready
        correlations_symmetric: Dict[str, Dict[str, float]] = defaultdict(dict)

        correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        for budget1_id in budget_ids:
            budget1 = run.get_budget(budget1_id)
            budget1_readable = run.get_budget(budget1_id, human=True)

            for budget2_id in budget_ids:
                budget2 = run.get_budget(budget2_id)
                budget2_readable = run.get_budget(budget2_id, human=True)

                costs1 = run.get_all_costs(budget1, statuses=[Status.SUCCESS])
                costs2 = run.get_all_costs(budget2, statuses=[Status.SUCCESS])

                # Combine config ids
                # So we make sure we have the same number of configs for each budget
                config_ids = set(costs1.keys()) & set(costs2.keys())

                c1, c2 = [], []
                for config_id in config_ids:
                    c1 += [costs1[config_id][objective_id]]
                    c2 += [costs2[config_id][objective_id]]

                correlation = round(stats.spearmanr(c1, c2).correlation, 2)
                correlations_symmetric["Budget"][budget2_readable] = budget2_readable
                correlations_symmetric[budget1_readable][budget2_readable] = correlation

                # Exclude if budget2 is higher than budget1
                if budget2 > budget1:
                    continue

                correlations[budget1_readable][budget2_readable] = correlation

        return {
            "correlations": correlations,
            "correlations_symmetric": correlations_symmetric,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            html.Div(id=register("text", "children"), className="mb-3"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        dcc.Graph(
                            id=register("graph", "figure"), style={"height": config.FIGURE_HEIGHT}
                        ),
                        label="Graph",
                    ),
                    dbc.Tab(html.Div(id=register("table", "children")), label="Table"),
                ]
            ),
        ]

    @staticmethod
    def load_outputs(run, _, outputs):
        traces = []
        categories = defaultdict(list)
        correlations = outputs["correlations"]
        correlations_symmetric = outputs["correlations_symmetric"]
        for idx, (budget1, budgets) in enumerate(correlations.items()):
            x, y = [], []
            for budget2, correlation in budgets.items():
                x += [float(budget2)]
                y += [correlation]

                category = None
                if correlation >= 0.7:
                    category = "very strong"
                elif correlation >= 0.4:
                    category = "strong"
                elif correlation >= 0.3:
                    category = "moderate"
                elif correlation >= 0.2:
                    category = "weak"
                else:
                    category = "not given"

                key = (budget1, budget2)
                key2 = (budget2, budget1)

                if budget1 == budget2:
                    continue

                if key not in categories[category] and key2 not in categories[category]:
                    if float(budget1) < float(budget2):
                        categories[category] += [key]
                    else:
                        categories[category] += [key2]

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
            margin=config.FIGURE_MARGIN,
            legend=dict(title="Budgets"),
        )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "budget_correlation.png")

        # Add vertical lines
        readable_budgets = run.get_budgets(human=True, include_combined=False)
        for idx, budget in enumerate(readable_budgets):
            figure.add_vline(
                x=budget,
                line=dict(
                    color=get_color(idx),
                    width=1,
                    dash="dot",
                ),
            )

        text = "The budget correlation of"
        n_categories = len(categories)
        for i, (relation, pairs) in enumerate(categories.items()):
            if n_categories > 1 and i == n_categories - 1:
                text += ", and "
            elif n_categories > 1 and i > 0:
                text += ", "
            else:
                text += " "

            formated_pairs = []
            for pair in pairs:
                formated_pairs += [f"({pair[0]}, {pair[1]})"]

            pairs_text = ", ".join(formated_pairs)
            text += f"{pairs_text} is {relation}"
            if i == n_categories - 1:
                text += "."

        return [
            text,
            figure,
            create_table(correlations_symmetric, mb=False),
        ]
