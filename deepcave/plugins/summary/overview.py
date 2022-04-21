import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
import plotly.graph_objs as go

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import Status
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from deepcave.utils.layout import create_table
from deepcave.utils.styled_plotty import get_discrete_heatmap
from deepcave.utils.util import get_latest_change


class Overview(DynamicPlugin):
    id = "overview"
    name = "Overview"
    icon = "fas fa-search"

    activate_run_selection = True

    @staticmethod
    def process(run, inputs):
        # Get best cost across all objectives, highest budget
        config, _ = run.get_incumbent()
        config_id = run.get_config_id(config)

        best_performance = {}
        for idx, cost in enumerate(run.get_cost(config_id)):
            best_performance[run.get_objective_names()[idx]] = cost

        # Card information
        card_information = {
            "optimizer": run.prefix,
            "latest_change": run.latest_change,
            "best_performance": best_performance,
            "num_configs": run.get_num_configs(),
        }

        # Meta
        meta = {"Attribute": [], "Value": []}
        for k, v in run.get_meta().items():
            if k == "objectives":
                continue

            if isinstance(v, list):
                v = ", ".join(str(_v) for _v in v)

            meta["Attribute"].append(k)
            meta["Value"].append(str(v))

        # Objectives
        objectives = {"Name": [], "Bounds": []}
        for objective in run.get_objectives():
            objectives["Name"].append(objective["name"])
            objectives["Bounds"].append(f"[{objective['lower']}, {objective['upper']}]")

        # Budgets
        budgets = run.get_budgets()

        status_statistics = {}
        # Statistics
        for budget in budgets:
            if budget not in status_statistics:
                status_statistics[budget] = {}

                for s in Status:
                    status_statistics[budget][s.name] = 0

        for trial in run.get_trials():
            status_statistics[trial.budget][trial.status.name] += 1

        # Now remove status that are not used
        for budget in list(status_statistics.keys()):
            for status in list(status_statistics[budget].keys()):
                if status_statistics[budget][status] == 0:
                    del status_statistics[budget][status]

        # It is interesting to see on which budget a configuration was evaluated
        config_statistics = {"X": budgets}
        configs = run.get_configs(budget)
        y = []
        z_values = []
        z_labels = []
        for config_id, config in configs.items():
            column_values = []
            column_labels = []
            for budget in budgets:
                trial_key = run.get_trial_key(config_id, budget)
                trial = run.get_trial(trial_key)

                status = Status.NOT_EVALUATED
                if trial is not None:
                    status = trial.status

                column_values += [status.value]
                column_labels += [status.name]

            y += [config_id]
            z_values += [column_values]
            z_labels += [column_labels]

        config_statistics["Y"] = y
        config_statistics["Z_values"] = z_values
        config_statistics["Z_labels"] = z_labels

        configspace = {
            "Hyperparameter": [],
            "Possible Values": [],
            "Default": [],
            "Log": [],
        }

        for hp_name, hp in run.configspace.get_hyperparameters_dict().items():

            log = False
            value = None
            if (
                isinstance(hp, UniformIntegerHyperparameter)
                or isinstance(hp, NormalIntegerHyperparameter)
                or isinstance(hp, UniformFloatHyperparameter)
                or isinstance(hp, NormalFloatHyperparameter)
            ):
                value = str([hp.lower, hp.upper])
                log = hp.log
            elif isinstance(hp, CategoricalHyperparameter):
                value = ", ".join([str(i) for i in hp.choices])
            elif isinstance(hp, OrdinalHyperparameter):
                value = ", ".join([str(i) for i in hp.sequence])
            elif isinstance(hp, Constant):
                value = str(hp.value)

            default = str(hp.default_value)
            log = str(log)

            configspace["Hyperparameter"].append(hp_name)
            configspace["Possible Values"].append(value)
            configspace["Default"].append(default)
            configspace["Log"].append(log)

        return {
            "card_information": card_information,
            "meta": meta,
            "objectives": objectives,
            "status_statistics": status_statistics,
            "config_statistics": config_statistics,
            "configspace": configspace,
        }

    @staticmethod
    def get_output_layout(register):

        return [
            dbc.Card(
                [
                    dbc.CardHeader(html.H3("Quick Information", className="mb-0")),
                    dbc.CardBody(
                        [
                            # html.H5("Run Card", className="card-title"),
                            html.Div(
                                [
                                    html.Span("Optimizer: "),
                                    html.Span(id=register("optimizer", "children")),
                                ],
                                className="card-text",
                            ),
                            html.Div(
                                [
                                    html.Span("Latest change: "),
                                    html.Span(id=register("latest_change", "children")),
                                ],
                                className="card-text",
                            ),
                            html.Div(
                                [
                                    html.Span("Best performance: "),
                                    html.Span(id=register("best_performance", "children")),
                                ],
                                className="card-text",
                            ),
                            html.Div(
                                [
                                    html.Span("Total configurations: "),
                                    html.Span(id=register("num_configs", "children")),
                                ],
                                className="card-text",
                            ),
                        ]
                    ),
                ],
                color="secondary",
                inverse=True,
                className="mb-3",
            ),
            html.H3("Meta"),
            html.Div(id=register("meta", "children")),
            html.H3("Objectives"),
            html.Div(id=register("objectives", "children")),
            html.Hr(),
            html.H3("Statuses"),
            dbc.Tabs(
                [
                    dbc.Tab(dcc.Graph(id=register("status_statistics", "figure")), label="Barplot"),
                    dbc.Tab(dcc.Graph(id=register("config_statistics", "figure")), label="Heatmap"),
                ]
            ),
            html.Hr(),
            html.H3("Configuration Space"),
            html.Div(id=register("configspace", "children")),
        ]

    def load_outputs(self, inputs, outputs, run):
        stats_data = []
        for budget, stats in outputs["status_statistics"].items():
            trace = go.Bar(x=list(stats.keys()), y=list(stats.values()), name=budget)
            stats_data.append(trace)

        stats_layout = go.Layout(
            barmode="group",
            xaxis=dict(title="Status"),
            yaxis=dict(title="Number of configurations"),
        )
        stats_figure = go.Figure(data=stats_data, layout=stats_layout)

        config_layout = go.Layout(
            showlegend=True,
            xaxis=dict(title="Budget"),
            yaxis=dict(title="Configuration ID"),
        )
        config_figure = go.Figure(
            data=get_discrete_heatmap(
                outputs["config_statistics"]["X"],
                outputs["config_statistics"]["Y"],
                outputs["config_statistics"]["Z_values"],
                outputs["config_statistics"]["Z_labels"],
            ),
            layout=config_layout,
        )

        best_performances = []
        for name, value in outputs["card_information"]["best_performance"].items():
            best_performances += [f"{round(value, 2)} ({name})"]

        return [
            outputs["card_information"]["optimizer"],
            get_latest_change(outputs["card_information"]["latest_change"]),
            ", ".join(best_performances),
            outputs["card_information"]["num_configs"],
            create_table(outputs["meta"]),
            create_table(outputs["objectives"]),
            stats_figure,
            config_figure,
            create_table(outputs["configspace"], mb=False),
        ]
