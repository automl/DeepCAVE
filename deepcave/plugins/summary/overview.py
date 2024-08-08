# noqa: D400
"""
# Overview

This module provides utilities for visualizing an overview of the selected runs.

It holds the most important information, e.g. meta data, objectives and statistics.

The module includes a dynamic plugin for the overview.

## Classes
    - Overview: Visualize an overall overview of the selected run.
"""

from typing import Any, Callable, Dict, List

import itertools

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import dcc, html

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.plugins.summary.configurations import Configurations
from deepcave.runs.group import Group
from deepcave.runs.status import Status
from deepcave.utils.layout import create_table, help_button
from deepcave.utils.styled_plotty import get_discrete_heatmap, save_image
from deepcave.utils.util import custom_round, get_latest_change


class Overview(DynamicPlugin):
    """Visualize an overall overview of the selected run."""

    id = "overview"
    name = "Overview"
    icon = "fas fa-search"
    help = "docs/plugins/overview.rst"
    use_cache = False
    activate_run_selection = True

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register the outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the output block.
        """
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
            html.Div(id=register("status_text", "children"), className="mb-3"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        dcc.Graph(
                            id=register("status_statistics", "figure"),
                            style={"height": config.FIGURE_HEIGHT},
                            config={
                                "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                            },
                        ),
                        label="Barplot",
                    ),
                    dbc.Tab(
                        dcc.Graph(
                            id=register("config_statistics", "figure"),
                            style={"height": config.FIGURE_HEIGHT},
                            config={
                                "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                            },
                        ),
                        label="Heatmap",
                    ),
                    dbc.Tab(html.Div(id=register("status_details", "children")), label="Details"),
                ]
            ),
            html.Hr(),
            html.H3("Configuration Space"),
            html.Div(id=register("configspace", "children")),
        ]

    @staticmethod
    def load_outputs(run, *_: Any) -> List[Any]:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.

        Returns
        -------
        List[Any]
            A list of the created tables of the overview.
        """
        optimizer = run.prefix
        if isinstance(run, Group):
            optimizer = run.get_runs()[0].prefix

        performance_outputs = []
        for idx, obj in enumerate(run.get_objectives()):
            # Get best cost for the objective, highest budget
            incumbent, _ = run.get_incumbent(objectives=obj, statuses=[Status.SUCCESS])
            config_id = run.get_config_id(incumbent)
            avg_costs, std_costs = run.get_avg_costs(config_id)

            if len(run.get_seeds(include_combined=False)) > 1:
                best_performance = (
                    f"{custom_round(avg_costs[idx])} " f"± {custom_round(std_costs[idx])}"
                )
            else:
                best_performance = f"{custom_round(avg_costs[idx])}"

            performance_outputs.append(
                html.Div(
                    [
                        html.Span(f"Best {obj.name}: {best_performance} "),
                        html.A(
                            "(See Configuration)",
                            href=Configurations.get_link(run, config_id),
                            style={"color": "white"},
                        ),
                    ],
                    className="card-text",
                ),
            )

        if isinstance(run, Group):
            runtime_str = "Maximum runtime"
        else:
            runtime_str = "Total runtime"

        # Design card for quick information here
        card = dbc.Card(
            [
                dbc.CardHeader(html.H3("Quick Information", className="mb-0")),
                dbc.CardBody(
                    [
                        html.Div(
                            f"Optimizer: {optimizer}",
                            className="card-text",
                        ),
                        html.Div(
                            f"Latest change: {get_latest_change(run.latest_change)}",
                            className="card-text",
                        ),
                        *performance_outputs,
                        html.Div(
                            [
                                html.Span(
                                    f"{runtime_str} [s]: "
                                    f"{max(trial.end_time for trial in run.history)}"
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
                        html.Div(
                            [
                                html.Span(f"Total trials: {len(run.history)}"),
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
        meta: Dict[str, List[str]] = {"Attribute": [], "Value": []}
        for k, v in run.get_meta().items():
            if k == "objectives":
                continue

            if isinstance(v, list):
                v = ", ".join(str(_v) for _v in v)

            meta["Attribute"].append(k)
            meta["Value"].append(str(v))

        # Objectives
        objectives: Dict[str, List[str]] = {"Name": [], "Bounds": []}
        for objective in run.get_objectives():
            objectives["Name"].append(objective.name)
            objectives["Bounds"].append(f"[{objective.lower}, {objective.upper}]")

        # Budgets
        budgets = run.get_budgets(include_combined=False)

        # Seeds
        seeds = run.get_seeds(include_combined=False)

        # Budget-seed combinations
        budget_seed_combinations = list(itertools.product(budgets, seeds))

        # Setup statistics dict for bar plot
        status_statistics: Dict[float, Dict[Status, int]] = {}

        for budget in budgets:
            budget = round(budget, 2)
            if budget not in status_statistics:
                status_statistics[budget] = {}

                for s in Status:
                    status_statistics[budget][s] = 0

        # Setup details dict for to collect information on failed trials
        status_details: Dict[str, List[Any]] = {
            "Configuration ID": [],
            "Budget": [],
            "Seed": [],
            "Status": [],
            "Error": [],
        }

        status_count = {}
        budget_count = {}
        len_trials = 0
        for trial in run.get_trials():
            budget = round(trial.budget, 2)
            seed = trial.seed

            len_trials += 1

            # Status count over budget for bar plot
            status_statistics[budget][trial.status] += 1

            # Total status count for text information
            if trial.status not in status_count:
                status_count[trial.status] = 1
            else:
                status_count[trial.status] += 1

            # Total budget count for text information
            if budget not in budget_count:
                budget_count[budget] = 1
            else:
                budget_count[budget] += 1

            # Add to table data
            if trial.status != Status.SUCCESS:
                link = Configurations.get_link(run, trial.config_id)

                status_details["Configuration ID"] += [
                    html.A(trial.config_id, href=link, target="_blank")
                ]
                status_details["Budget"] += [budget]
                status_details["Seed"] += [seed]
                status_details["Status"] += [trial.status.to_text()]

                if "traceback" in trial.additional:
                    traceback = trial.additional["traceback"]
                    status_details["Error"] += [help_button(traceback)]
                else:
                    status_details["Error"] += ["No traceback available."]

        # Successful / unsuccessful trials rate for text information
        successful_trials_rate = status_count[Status.SUCCESS] / len_trials * 100
        successful_trials_rate = round(successful_trials_rate, 2)

        trials_rates = []
        for status, count in status_count.items():
            if status == Status.SUCCESS:
                continue

            rate = round(count / len_trials * 100, 2)
            trials_rates += [status.to_text() + f" ({rate}%)"]

        # Add an "or" to the last rate
        if successful_trials_rate != 100 and len(trials_rates) > 0:
            unsuccessful_trials_text = "The other trials are "
            if len(trials_rates) == 1:
                unsuccessful_trials_text += trials_rates[0]
            elif len(trials_rates) == 2:
                unsuccessful_trials_text += "either "
                unsuccessful_trials_text += trials_rates[0] + " or " + trials_rates[1]
            else:
                unsuccessful_trials_text += "either "
                trials_rates[-1] = " or " + trials_rates[-1]
                unsuccessful_trials_text += ", ".join(trials_rates)
            unsuccessful_trials_text += "."
        else:
            unsuccessful_trials_text = ""

        # Budget rate for text information
        budget_rate = [
            str(round(count / len_trials * 100, 2)) + "%" for count in budget_count.values()
        ]
        budget_rate_text = "/".join(budget_rate)
        budget_keys_text_list = [str(key) for key in budget_count.keys()]
        budget_keys_text = "/".join(budget_keys_text_list)

        # Text information
        status_text = f"""
        Taking all evaluated trials into account, {successful_trials_rate}% have been successful.
        {unsuccessful_trials_text}
        Moreover, {budget_rate_text} of the trials were evaluated on budget
        {budget_keys_text}, respectively.
        """

        # Status statistics for bar plot: remove status that are not used
        for budget in list(status_statistics.keys()):
            for status in list(status_statistics[budget].keys()):
                if status_statistics[budget][status] == 0:
                    del status_statistics[budget][status]

        # Config statistics for heatmap showing on which budget / seed a configuration was evaluated
        config_statistics = {}
        configs = run.get_configs()
        config_ids = list(configs.keys())

        z_values = np.zeros((len(config_ids), len(budget_seed_combinations))).tolist()
        z_labels = np.zeros((len(config_ids), len(budget_seed_combinations))).tolist()

        for i, config_id in enumerate(configs.keys()):
            for j, (b, s) in enumerate(budget_seed_combinations):
                trial_key = run.get_trial_key(config_id, b, s)
                trial = run.get_trial(trial_key)

                status = Status.NOT_EVALUATED
                if trial is not None:
                    status = trial.status
                z_values[i][j] = status.value
                z_labels[i][j] = status.to_text()

        config_statistics["X"] = budget_seed_combinations
        config_statistics["Y"] = config_ids
        config_statistics["Z_values"] = z_values
        config_statistics["Z_labels"] = z_labels

        # Prepare configspace table
        configspace: Dict[str, List] = {
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
            log_str = str(log)

            configspace["Hyperparameter"].append(hp_name)
            configspace["Possible Values"].append(value)
            configspace["Default"].append(default)
            configspace["Log"].append(log_str)

        stats_data = []
        for budget, stats in status_statistics.items():
            x = [s.to_text() for s in stats.keys()]
            trace = go.Bar(x=x, y=list(stats.values()), name=budget)
            stats_data.append(trace)

        stats_layout = go.Layout(
            legend={"title": "Budget (Seed)"},
            barmode="group",
            xaxis=dict(title="Status"),
            yaxis=dict(title="Number of configurations"),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        stats_figure = go.Figure(data=stats_data, layout=stats_layout)
        save_image(stats_figure, "status_bar.pdf")

        config_layout = go.Layout(
            legend={"title": "Status"},
            xaxis=dict(title="Budget (Seed)"),
            yaxis=dict(title="Configuration ID"),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
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
        save_image(config_figure, "status_heatmap.pdf")

        return [
            card,
            create_table(meta, fixed=True),
            create_table(objectives, fixed=True),
            status_text,
            stats_figure,
            config_figure,
            create_table(status_details),
            create_table(configspace, mb=False),
        ]
