#  noqa: D400
"""
# Importances

This module provides a plugin for the visualization of the importances.

Provided utilities include getting input and output layout (filtered or non-filtered),
processing the data and loading the outputs. Also provides a matplotlib version.

## Classes
    - Importances: This class provides a plugin for the visualization of the importances.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.cast import optional_int
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.styled_plot import plt
from deepcave.utils.styled_plotty import get_color, save_image


class Importances(StaticPlugin):
    """
    Provide a plugin for the visualization of the importances.

    Provided utilities include getting input/output layout, data processing
    and loading outputs. Also provides a matplotlib version.
    """

    id = "importances"
    name = "Importances"
    icon = "far fa-star"
    help = "docs/plugins/importances.rst"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get the html container for the layout of the input.

        Parameters
        ----------
        register : Callable
            Used to get the id of the objective.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            An html container for the layout of the input.
        """
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Method"),
                            help_button(
                                "Local Parameter Importance: Quantify importance by changing the "
                                "neighborhood of a configuration. Uses default and incumbent "
                                "configuration as reference. \n\n"
                                "fANOVA: Quantify importance globally."
                            ),
                            dbc.Select(
                                id=register("method", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Trees"),
                            help_button(
                                "The number of trees of the internal random forest to estimate the "
                                "hyperparameter importance. "
                                "The more trees are used the more accurate the results. "
                                "However, also it takes longer to compute."
                            ),
                            dbc.Input(id=register("n_trees", type=optional_int), type="number"),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[html.Div]:
        """
        Get the layout for a filtered html container.

        Parameters
        ----------
        register : Callable
            Used for the id of the Checklist.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[html.Div]
            A filtered html container.
        """
        return [
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Limit Hyperparameters"),
                    help_button(
                        "Shows only the n most important hyperparameters. If an important "
                        "hyperparameter was de-selected, it is not shown but skipped."
                    ),
                    dbc.Input(id=register("n_hps", "value"), type="number"),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budgets"),
                    help_button("The hyperparameters are sorted by the highest budget."),
                    dbc.Checklist(id=register("budget_ids", ["value", "options"]), inline=True),
                ]
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the method labels, values and different attributes.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The attributes of the inputs.
        """
        method_labels = ["Local Parameter Importance (local)", "fANOVA (global)"]
        method_values = ["local", "global"]

        return {
            "method": {
                "options": get_select_options(method_labels, method_values),
                "value": "local",
            },
            "n_trees": {"value": 10},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "n_hps": {"value": 0},
            "budget_ids": {"options": get_checklist_options(), "value": []},
        }

    # Types dont match superclass
    def load_dependency_inputs(self, run, _: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load the objective, budgets and hyperparameters (HPs) and its attributes.

        Parameters
        ----------
        run
            The run to get the objective from.
        inputs : Dict[str, Any]
            Contains information about the objective, budgets
            and number of hyperparamaters (HPs).

        Returns
        -------
        Dict[str, Any]
            The objective, budgets, hyperparameters (HPs) and their attributes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_ids"]["value"]

        hp_names = run.configspace.get_hyperparameter_names()
        hp_options = get_checklist_options(hp_names)
        hp_value = inputs["hyperparameter_names"]["value"]
        n_hps = inputs["n_hps"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]
            n_hps = len(hp_names)

        if n_hps == 0:
            n_hps = len(hp_names)

        # Pre-selection of the hyperparameters
        if run is not None:
            if len(hp_value) == 0:
                hp_value = hp_names
            if len(budget_value) == 0:
                budget_value = [budget_ids[-1]]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "method": {
                "value": inputs["method"]["value"],
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_value,
            },
            "budget_ids": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_hps": {"value": n_hps},
            "n_trees": {"value": inputs["n_trees"]["value"]},
        }

    @staticmethod
    # Return doesnt match superclass type
    def process(run: AbstractRun, inputs: Dict[str, Any]):
        """
        Initialize the evaluator, calculate and get the processed data.

        Parameters
        ----------
        run : AbstractRun
            The run to get the objective from.
        inputs :  Dict[str, Any]
            Contains information about the method and the number of trees.

        Returns
        -------
        The processed data.

        Raises
        ------
        RuntimeError
            If the number of trees is not specified.
        RuntimeError
            If the method is not found.
        """
        objective = run.get_objective(inputs["objective_id"])
        method = inputs["method"]
        n_trees = inputs["n_trees"]

        if n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        hp_names = run.configspace.get_hyperparameter_names()
        budgets = run.get_budgets(include_combined=True)

        evaluator: Optional[Union[LocalEvaluator, GlobalEvaluator]] = None
        if method == "local":
            # Initialize the evaluator
            evaluator = LocalEvaluator(run)
        elif method == "global":
            evaluator = GlobalEvaluator(run)
        else:
            raise RuntimeError("Method was not found.")

        # Collect data
        data = {}
        for budget_id, budget in enumerate(budgets):
            assert isinstance(budget, (int, float))
            evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)

            importances = evaluator.get_importances(hp_names)
            data[budget_id] = importances

        return data

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get a graph with the layout of the output.

        Parameters
        ----------
        register : Callable
            A function to get the id for the graph.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            The graph with the layout of the output.
        """
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    # Types dont match superclass
    def load_outputs(run, inputs, outputs):
        """
        Load the importances and the corresponding layout of the figure.

        Also safes the image of the figure.

        Parameters
        ----------
        run
            The run to get the budget from.
        inputs
            Containing the hyperparameter (HPs) names, the budget ids
            and the number of hyperparamaters (HPs).
        outputs
            Containing the budget id and importances.

        Returns
        -------
        The figure of the importances.
        """
        # First selected, should always be shown first
        selected_hp_names = inputs["hyperparameter_names"]
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if len(selected_hp_names) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            if budget_id not in selected_budget_ids:
                continue

            x = []
            y = []
            error_y = []
            for hp_name, results in importances.items():
                if hp_name not in selected_hp_names:
                    continue

                x += [hp_name]
                y += [results[0]]
                error_y += [results[1]]

            data[budget_id] = (np.array(x), np.array(y), np.array(error_y))

        # Sort by last fidelity now
        selected_budget_id = max(selected_budget_ids)
        idx = np.argsort(data[selected_budget_id][1], axis=None)[::-1]
        idx = idx[:n_hps]

        bar_data = []
        for budget_id, values in data.items():
            budget = run.get_budget(budget_id, human=True)

            x = values[0][idx]
            # new_x = []
            # for string in x:
            #    string = string.replace("center_optimizer:", "")
            #    string = string.replace(":__choice__", "")
            #    string = string.replace("AdamWOptimizer", "AdamW")
            #    string = string.replace("SGDOptimizer", "SGD")
            #    new_x += [string]
            # x = new_x

            bar_data += [
                go.Bar(
                    name=budget,
                    x=x,
                    y=values[1][idx],
                    error_y_array=values[2][idx],
                    marker_color=get_color(budget_id),
                )
            ]

        figure = go.Figure(data=bar_data)
        figure.update_layout(
            barmode="group",
            yaxis_title="Importance",
            legend={"title": "Budget"},
            margin=config.FIGURE_MARGIN,
            xaxis=dict(tickangle=-45),
        )
        save_image(figure, "importances.pdf")

        return figure

    @staticmethod
    def get_mpl_output_layout(register: Callable) -> html.Img:
        """
        Get an html container of the output layout.

        Parameters
        ----------
        register : Callable
            A function to get the id.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        html.Img
            An html container of the matplotlib output layout.
        """
        return html.Img(
            id=register("graph", "src"),
            className="img-fluid",
        )

    @staticmethod
    # Types dont match superclass
    def load_mpl_outputs(run, inputs: Dict[str, Any], outputs):
        """
        Load the importances and the corresponding layout of the matplotlib figure.

        Parameters
        ----------
        run
            The run to get the budget from.
        inputs : Dict[str, Any]
            Containing the hyperparameter (HPs) names, the budget ids
            and the number of hyperparameters (HPs).
        outputs
            Containing the budget id and importances.

        Returns
        -------
        The rendered matplotlib figure of the importances.
        """
        # First selected, should always be shown first
        selected_hp_names = inputs["hyperparameter_names"]
        selected_budget_ids = inputs["budget_ids"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate()
        else:
            n_hps = int(n_hps)

        if len(selected_hp_names) == 0 or len(selected_budget_ids) == 0:
            raise PreventUpdate()

        # Collect data
        data = {}
        for budget_id, importances in outputs.items():
            # Important to cast budget_id here because of json serialization
            budget_id = int(budget_id)
            # if budget_id not in selected_budget_ids:
            #    continue

            x = []
            y = []
            error_y = []
            for hp_name, results in importances.items():
                if hp_name not in selected_hp_names:
                    continue

                x += [hp_name]
                y += [results[0]]
                error_y += [results[1]]

            data[budget_id] = (np.array(x), np.array(y), np.array(error_y))

        # Sort by last fidelity now
        selected_budget_id = max(selected_budget_ids)
        idx = np.argsort(data[selected_budget_id][1], axis=None)[::-1]
        idx = idx[:n_hps]

        x_labels = []
        for hp_name in data[selected_budget_id][0][idx]:
            if len(hp_name) > 18:
                hp_name = "..." + hp_name[-18:]

            x_labels += [hp_name]
        x_values = np.arange(len(x_labels))

        plt.figure()
        for budget_id, values in data.items():
            if budget_id not in selected_budget_ids:
                continue

            y = values[1][idx]
            y_err = values[2][idx]

            budget = run.get_budget(budget_id, human=True)
            plt.bar(
                x_values,
                y,
                yerr=y_err,
                # plt is matplotlib object and has no function get_color, Issue opened
                color=plt.get_color(budget_id),
                label=budget,
                error_kw=dict(lw=1, capsize=2, capthick=1),
            )

        plt.legend(title="Budgets")

        # Rotate x ticks
        plt.xticks(x_values, x_labels, rotation=90)
        plt.ylabel("Importance")
        # plt is matplotlib object and has no function render, Issue opened
        return plt.render()
