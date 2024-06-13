#  noqa: D400
"""
# ConfigurationCube

This module provides utilities for visualizing and creating a configuration cube.

The configuration cube displays configurations and their score on the objective.

## Classes
    - ConfigurationCube: A plugin for visualizing a configuration cube.
"""

from typing import Any, Callable, Dict, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.layout import (
    get_checklist_options,
    get_select_options,
    get_slider_marks,
    help_button,
)
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import (
    get_hovertext_from_config,
    get_hyperparameter_ticks,
    save_image,
)

logger = get_logger(__name__)


class ConfigurationCube(DynamicPlugin):
    """A plugin for visualizing a configuration cube."""

    id = "ccube"
    name = "Configuration Cube"
    icon = "fas fa-cube"
    activate_run_selection = True
    help = "docs/plugins/configuration_cube.rst"

    @staticmethod
    def get_input_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            Layouts for the input block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            help_button(
                                "Budget refers to the multi-fidelity budget. "
                                "Combined budget means that the trial on the highest"
                                " evaluated budget is used.  \n "
                                "Note: Selecting combined budget might be misleading if"
                                " a time objective is used. Often, higher budget take "
                                " longer to evaluate, which might negatively influence "
                                " the results."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
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
            Layouts for the filter block.
        """
        return [
            html.Div(
                [
                    dbc.Label("Number of Configurations"),
                    help_button(
                        "The number of configurations to show, these are ordered based on"
                        " the time at which they were evaluated."
                    ),
                    dcc.Slider(
                        id=register("n_configs", ["value", "min", "max", "marks"]), step=None
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    help_button("Which hyperparameters to show. Maxium 3 active."),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ]
            ),
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        So, if the plugin is called for the first time or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Any]
            The content to be filled.
        """
        return {
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Work like 'load_inputs' but called after inputs have changed.

        It is restricted to three Hyperparameters.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
           The dictionary with the changes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_select_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        # Prepare others
        hp_names = run.configspace.get_hyperparameter_names()

        # Get selected values
        n_configs_value = inputs["n_configs"]["value"]

        # Pre-set values
        if objective_value is None:
            objective_value = objective_ids[0]
        if budget_value is None:
            budget_value = budget_ids[-1]
        else:
            budget_value = inputs["budget_id"]["value"]

        budget = run.get_budget(budget_value)
        configs = run.get_configs(budget=budget)
        if n_configs_value == 0:
            n_configs_value = len(configs)
        else:
            if n_configs_value > len(configs):
                n_configs_value = len(configs)

        # Restrict to three hyperparameters
        selected_hps = inputs["hyperparameter_names"]["value"]
        n_selected = len(selected_hps)
        if n_selected > 3:
            del selected_hps[0]

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
            "n_configs": {
                "min": 0,
                "max": len(configs),
                "marks": get_slider_marks(list(range(0, len(configs) + 1))),
                "value": n_configs_value,
            },
            "hyperparameter_names": {
                "options": get_select_options(hp_names),
                "value": selected_hps,
            },
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Return raw data based on the run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The selected run.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, str]
            The serialized dictionary.
        """
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        df = run.get_encoded_data(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, include_config_ids=True
        )
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register: Callable) -> Tuple[dcc.Graph,]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register output.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        Tuple[dcc.Graph,]
            Layout for the output block.
        """
        return (
            dcc.Graph(
                register("graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """
        Read in the raw data and prepares them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
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
        go.Figure
            The output figure.
        """
        df = deserialize(outputs["df"], dtype=pd.DataFrame)
        hp_names = inputs["hyperparameter_names"]
        n_configs = inputs["n_configs"]
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)
        budget = run.get_budget(inputs["budget_id"])
        df = df.groupby(df.columns.drop(objective.name).to_list(), as_index=False).mean()
        df.index = df.index.astype("str")

        # Limit to n_configs
        idx = [str(i) for i in range(n_configs, len(df))]
        df = df.drop(idx)

        costs = df[objective.name].values.tolist()
        config_ids = df["config_id"].values.tolist()
        data = []

        # Specify layout kwargs
        layout_kwargs = {}
        if n_configs > 0 and len(hp_names) > 0:
            for i, (hp_name, axis_name) in enumerate(zip(hp_names, ["xaxis", "yaxis", "zaxis"])):
                hp = run.configspace.get_hyperparameter(hp_name)
                values = df[hp_name].values.tolist()

                tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)
                layout_kwargs[axis_name] = {
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                    "title": hp_name,
                }
                data.append(values)

        if len(data) == 0:
            raise PreventUpdate

        # Transforms data to values
        values = np.transpose(np.array(data)).tolist()

        # Now the duplicates are filtered
        filtered_data: Dict[Tuple[int, float], Tuple] = {}
        for config_id, cost, v in zip(config_ids, costs, values):
            v = tuple(v)  # Make hashable
            key = (config_id, cost)

            if v in filtered_data.values():
                old_key = list(filtered_data.keys())[list(filtered_data.values()).index(v)]
                old_cost = old_key[1]

                if objective.optimize == "lower":
                    if old_cost < cost:
                        continue

                if objective.optimize == "upper":
                    if old_cost > cost:
                        continue

                # Otherwise it is replaced
                del filtered_data[old_key]

            filtered_data[key] = v

        # Replace data
        config_ids = [k[0] for k in filtered_data.keys()]
        costs = [k[1] for k in filtered_data.keys()]

        # Specify scatter kwargs
        scatter_kwargs = {
            "mode": "markers",
            "marker": {
                "size": 5,
                "color": costs,
                "colorbar": {"thickness": 30, "title": objective.name},
            },
            "hovertext": [
                get_hovertext_from_config(run, config_id, budget) for config_id in config_ids
            ],
            "meta": {"colorbar": costs},
            "hoverinfo": "text",
        }

        if len(data) >= 1:
            x = [v[0] for v in filtered_data.values()]
        if len(data) >= 2:
            y = [v[1] for v in filtered_data.values()]
        if len(data) >= 3:
            z = [v[2] for v in filtered_data.values()]

        if len(data) == 3:
            trace = go.Scatter3d(x=x, y=y, z=z, **scatter_kwargs)
            layout = go.Layout(
                {"scene": {**layout_kwargs}},
                font=dict(size=config.FIGURE_FONT_SIZE),
            )
        else:
            if len(data) == 1:
                y = [0 for _ in x]
                trace = go.Scatter(x=x, y=y, **scatter_kwargs)
            elif len(data) == 2:
                trace = go.Scatter(x=x, y=y, **scatter_kwargs)
            else:
                trace = go.Scatter(x=[], y=[])
            layout = go.Layout(
                **layout_kwargs,
                font=dict(size=config.FIGURE_FONT_SIZE),
            )

        figure = go.Figure(data=trace, layout=layout)
        figure.update_layout(dict(margin=config.FIGURE_MARGIN))
        save_image(figure, "configuration_cube.pdf")

        return figure
