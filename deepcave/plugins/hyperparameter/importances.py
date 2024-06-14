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
from ConfigSpace import ConfigurationSpace, Constant
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.evaluators.fanova import fANOVA as GlobalEvaluator
from deepcave.evaluators.lpi import LPI as LocalEvaluator
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.cast import optional_int
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plot import plt
from deepcave.utils.styled_plotty import get_color, save_image

logger = get_logger(__name__)


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
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layout for the input block.
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
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[html.Div]
            Layout for the filter block.
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
                    help_button(
                        "Budget refers to the multi-fidelity budget. "
                        "The hyperparameters are sorted by the highest budget."
                    ),
                    dbc.Checklist(id=register("budget_ids", ["value", "options"]), inline=True),
                ]
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        If the plugin is called for the first time, or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Content to be filled.
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

    def load_dependency_inputs(self, run, _: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore # noqa: E501
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs : Dict[str, Any]
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the changes.
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
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return raw data based on the run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs :  Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialzied dictionary.

        Raises
        ------
        RuntimeError
            If the number of trees is not specified.
            If the method is not found.
        """
        objective = run.get_objective(inputs["objective_id"])
        method = inputs["method"]
        n_trees = inputs["n_trees"]

        if n_trees is None:
            raise RuntimeError("Please specify the number of trees.")

        # Handle constant values in fANOVA: As the fANOVA implementation relies on pyrfr
        # and pyrfr cannot be applied to constant hyperparameters
        # (see https://github.com/automl/fanova/issues/81), as a workaround constant
        # hyperparameters are removed before calculation.
        # Note: This will break if there are conditions or forbiddens including
        # constant hyperparameters.
        hp_dict = run.configspace.get_hyperparameters_dict()
        if method == "global" and any([type(v) == Constant for v in hp_dict.values()]):
            hp_dict_wo_const = {k: v for k, v in hp_dict.items() if type(v) != Constant}
            configspace_wo_const = ConfigurationSpace()
            for k in hp_dict_wo_const.keys():
                configspace_wo_const.add_hyperparameter(hp_dict_wo_const[k])
            configspace_wo_const.add_conditions(run.configspace.get_conditions())
            configspace_wo_const.add_forbidden_clauses(run.configspace.get_forbiddens())
            run.configspace = configspace_wo_const

            configs_wo_const = []
            for n in range(len(run.configs)):
                configs_wo_const.append(
                    {k: v for k, v in run.configs[n].items() if k in hp_dict_wo_const.keys()}
                )
            run.configs = dict(enumerate(configs_wo_const))

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
            if any(np.isnan(val) for value in importances.values() for val in value):
                logger.warning(f"Nan encountered in importance values for budget {budget}.")
            data[budget_id] = importances

        return data  # type: ignore

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            Layout for the output block.
        """
        return dcc.Graph(
            register("graph", "figure"),
            style={"height": config.FIGURE_HEIGHT},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in raw data and prepare for layout.

        Note
        ----
        The passed inputs are cleaned and therefore differ
        compared to 'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        go.figure
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
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        save_image(figure, "importances.pdf")

        return figure

    @staticmethod
    def get_mpl_output_layout(register: Callable) -> html.Img:
        """
        Get the layout for the matplotlib output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        html.Img
            The layout for the matplotlib output block.
        """
        return html.Img(
            id=register("graph", "src"),
            className="img-fluid",
        )

    @staticmethod
    def load_mpl_outputs(run, inputs: Dict[str, Any], outputs):  # type: ignore
        """
        Read the raw data and prepare it for the layout.

        Parameters
        ----------
        run
            The selected run.
        inputs : Dict[str, Any]
            Input and filter values from the user.
        outputs
            Raw output from the run.

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
                color=plt.get_color(budget_id),  # type: ignore
                label=budget,
                error_kw=dict(lw=1, capsize=2, capthick=1),
            )

        plt.legend(title="Budgets")

        # Rotate x ticks
        plt.xticks(x_values, x_labels, rotation=90)
        plt.ylabel("Importance")

        return plt.render()  # type: ignore
