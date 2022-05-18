import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
import plotly.graph_objs as go

from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.summary.configurations import Configurations
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
from deepcave.utils.layout import create_table, help_button
from deepcave.utils.styled_plotty import get_discrete_heatmap
from deepcave.utils.util import get_latest_change


class Overview(DynamicPlugin):
    id = "overview"
    name = "Overview"
    icon = "fas fa-search"

    use_cache = False
    activate_run_selection = True

    @staticmethod
    def get_output_layout(register):
        return [
            html.Div(
                id=register("card", "children"),
                className="mb-3",
            ),
            html.Hr(),
            html.H3("Meta"),
            html.Div(id=register("meta", "children")),
            html.Hr(),
            html.H3("Objectives"),
            html.Div(id=register("objectives", "children")),
            html.Hr(),
            html.H3("Statuses"),
            dbc.Tabs(
                [
                    dbc.Tab(dcc.Graph(id=register("status_statistics", "figure")), label="Barplot"),
                    dbc.Tab(dcc.Graph(id=register("config_statistics", "figure")), label="Heatmap"),
                    dbc.Tab(html.Div(id=register("status_details", "children")), label="Details"),
                ]
            ),
            html.Hr(),
            html.H3("Configuration Space"),
            html.Div(id=register("configspace", "children")),
        ]

    @staticmethod
    def load_outputs(run, *_):
        # Get best cost across all objectives, highest budget
        config, _ = run.get_incumbent()
        config_id = run.get_config_id(config)
        objective_names = run.get_objective_names()

        best_performance = {}
        # Budget might not be evaluated
        try:
            costs = run.get_costs(config_id)
        except Exception:
            costs = [None for _ in range(len(objective_names))]

        for idx, cost in enumerate(costs):
            best_performance[objective_names[idx]] = cost

        best_performances = []
        for name, value in best_performance.items():
            best_performances += [f"{round(value, 2)} ({name})"]

        # Design card for quick information here
        card = dbc.Card(
            [
                dbc.CardHeader(html.H3("Quick Information", className="mb-0")),
                dbc.CardBody(
                    [
                        html.Div(
                            f"Optimizer: {run.prefix}",
                            className="card-text",
                        ),
                        html.Div(
                            f"Latest change: {get_latest_change(run.latest_change)}",
                            className="card-text",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"Best average performance: {', '.join(best_performances)} "
                                ),
                                html.A(
                                    "(Details)",
                                    href=Configurations.get_link(run, config_id),
                                    style={"color": "white"},
                                ),
                            ],
                            className="card-text",
                        ),
                        html.Div(
                            [
                                html.Span(f"Total configurations: {run.get_num_configs()}"),
                            ],
                            className="card-text",
                        ),
                    ]
                ),
            ],
            color="secondary",
            inverse=True,
        )

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
        status_details = {"Configuration ID": [], "Budget": [], "Status": [], "Error": []}
        # Statistics
        for budget in budgets:
            if budget not in status_statistics:
                status_statistics[budget] = {}

                for s in Status:
                    status_statistics[budget][s.name] = 0

        for trial in run.get_trials():
            status_statistics[trial.budget][trial.status.name] += 1

            # Add to table data
            if trial.status != Status.SUCCESS:
                status_details["Configuration ID"] += [trial.config_id]
                status_details["Budget"] += [trial.budget]
                status_details["Status"] += [trial.status.to_text()]

                if "traceback" in trial.additional:
                    traceback = trial.additional["traceback"]
                    status_details["Error"] += [help_button(traceback)]
                else:
                    status_details["Error"] += [""]

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

        # Prepare configspace table
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

        stats_data = []
        for budget, stats in status_statistics.items():
            trace = go.Bar(x=list(stats.keys()), y=list(stats.values()), name=budget)
            stats_data.append(trace)

        stats_layout = go.Layout(
            legend={"title": "Budget"},
            barmode="group",
            xaxis=dict(title="Status"),
            yaxis=dict(title="Number of configurations"),
        )
        stats_figure = go.Figure(data=stats_data, layout=stats_layout)

        config_layout = go.Layout(
            xaxis=dict(title="Budget"),
            yaxis=dict(title="Configuration ID"),
        )
        config_figure = go.Figure(
            data=get_discrete_heatmap(
                config_statistics["X"],
                config_statistics["Y"],
                config_statistics["Z_values"],
                config_statistics["Z_labels"],
            ),
            layout=config_layout,
        )

        return [
            card,
            create_table(meta, fixed=True),
            create_table(objectives, fixed=True),
            stats_figure,
            config_figure,
            create_table(status_details),
            create_table(configspace, mb=False),
        ]
