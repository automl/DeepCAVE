#  noqa: D400
"""
# Configurations

This module provides utilities to visualize the characteristics of a configuration within a run.

The module provides a corresponding dynamic plugin.

## Classes
    - Configurations: Visualize the characteristics of a configuration.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config
from deepcave.constants import VALUE_RANGE
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun
from deepcave.runs.status import Status
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.layout import create_table, get_slider_marks
from deepcave.utils.styled_plotty import (
    generate_config_code,
    get_color,
    get_hyperparameter_ticks,
    save_image,
)
from deepcave.utils.url import create_url


class Configurations(DynamicPlugin):
    """Visualize the characteristics of a configuration."""

    id = "configurations"
    name = "Configurations"
    icon = "fas fa-sliders-h"
    help = "docs/plugins/configurations.rst"
    activate_run_selection = True
    use_cache = False

    @staticmethod
    def get_link(run: AbstractRun, config_id: int) -> str:
        """
        Create a link to a specific configuration.

        Parameters
        ----------
        run : AbstractRun
            The selected run.
        config_id : int
            Configuration which should be visited.

        Returns
        -------
        str
            Link to the configuration.
        """
        # Create "fake" inputs to overwrite the selection.
        # Everything else will be taken from `load_inputs` and `load_dependency_inputs`.
        inputs = {
            "run": dict(value=run.id),
            "config_id": dict(value=config_id),
        }
        url = Configurations.get_base_url()

        return create_url(url, inputs)

    @staticmethod
    def get_input_layout(register: Callable) -> List[html.Div]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[html.Div]
            The layouts for the input block.
        """
        return [
            html.Div(
                [
                    dbc.Label("Configuration ID"),
                    dcc.Slider(id=register("config_id", ["value", "min", "max", "marks"])),
                ],
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
            The content to be filled
        """
        return {
            "config_id": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
        }

    def load_dependency_inputs(  # type: ignore
        self, run, previous_inputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run :
            The selected run.
        previous_inputs :
            Previous content of the inputs.
            Not used in this specific function
        inputs : Dict[str, Any]
            The current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the changes.
        """
        # Get selected values
        config_id_value = inputs["config_id"]["value"]
        configs = run.get_configs()
        if config_id_value > len(configs) - 1:
            config_id_value = len(configs) - 1

        return {
            "config_id": {
                "min": 0,
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(configs.keys()), access_all=True),
                "value": config_id_value,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and input data.

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
        run :
            The selected run.
        inputs :
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.
        """
        selected_config_id = int(inputs["config_id"])
        origin = run.get_origin(selected_config_id)
        objectives = run.get_objectives()
        budgets = run.get_budgets(include_combined=False)
        seeds = run.get_seeds(include_combined=False)

        overview_table_data = {
            "Key": ["Selected Configuration", "Origin"],
            "Value": [selected_config_id, origin],
        }

        if run.prefix == "group":
            original_run = run.get_original_run(selected_config_id)
            original_config_id = run.get_original_config_id(selected_config_id)
            overview_table_data["Key"] += ["Derived from"]
            overview_table_data["Value"] += [
                str(original_run.path) + f" (Configuration ID: {original_config_id})"
            ]

        performances: Dict[str, Dict[Union[int, float], Optional[Union[float, List[float]]]]] = {}
        performances_table_data: Dict[str, List[Any]] = {"Budget": []}
        for objective_id, objective in enumerate(objectives):
            if objective.name not in performances:
                performances[objective.name] = {}
                performances_table_data[objective.name] = []
            for budget in budgets:
                # Budget might not be evaluated
                seeds_evaluated = 0
                cost = []
                for seed in seeds:
                    all_costs = run.get_all_costs(
                        budget=budget,
                        seed=seed,
                        statuses=[Status.SUCCESS],
                    )
                    if selected_config_id in all_costs:
                        cost.append(all_costs[selected_config_id][seed][objective_id])
                        seeds_evaluated += 1
                    else:
                        continue

                # Add table data
                if budget not in performances_table_data["Budget"]:
                    performances_table_data["Budget"] += [budget]
                if seeds_evaluated > 0:
                    performances[objective.name][budget] = cost
                    if len(seeds) > 1:
                        performances_table_data[objective.name] += [
                            f"{np.mean(cost)} (\u00B1 {np.std(cost)})"
                        ]
                    else:
                        performances_table_data[objective.name] += [cost]
                else:
                    performances[objective.name][budget] = np.nan
                    if len(seeds) > 1:
                        performances_table_data[f"{objective.name}"] += ["No seed evaluated"]
                    else:
                        status = run.get_status(selected_config_id, seeds[0], budget)
                        performances_table_data[objective.name] += [status.to_text()]

        # Let's start with the configspace
        X = []
        cs_table_data: Dict[str, List[Any]] = {"Hyperparameter": [], "Value": []}
        for config_id, configuration in run.get_configs().items():
            x = run.encode_config(configuration)

            highlight = 0
            if config_id == selected_config_id:
                highlight = 1

                for k, v in configuration.items():
                    # Add accurate data for our table here
                    cs_table_data["Hyperparameter"] += [k]
                    cs_table_data["Value"] += [v]

            # Highlight is simply added as a new column
            x += [highlight]

            # And add it to the lists
            X.append(x)

        columns = run.configspace.get_hyperparameter_names()
        columns += ["highlighted"]

        cs_df = pd.DataFrame(data=X, columns=columns)

        return {
            "overview_table_data": overview_table_data,
            "performances": performances,
            "performances_table_data": performances_table_data,
            "cs_df": serialize(cs_df),
            "cs_table_data": cs_table_data,
        }

    @staticmethod
    def get_output_layout(
        register: Callable,
    ) -> List[Any]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method used to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the outputs.
        """
        return [
            html.Div(id=register("overview_table", "children"), className="mb-3"),
            html.Hr(),
            html.H3("Objectives"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        dcc.Graph(
                            id=register("performance_graph", "figure"),
                            style={"height": config.FIGURE_HEIGHT},
                            config={
                                "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                            },
                        ),
                        label="Graph",
                    ),
                    dbc.Tab(html.Div(id=register("performance_table", "children")), label="Table"),
                ]
            ),
            html.Hr(),
            html.H3("Configuration"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        dcc.Graph(
                            id=register("configspace_graph", "figure"),
                            style={"height": config.FIGURE_HEIGHT},
                            config={
                                "toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}
                            },
                        ),
                        label="Graph",
                    ),
                    dbc.Tab(html.Div(id=register("configspace_table", "children")), label="Table"),
                ]
            ),
            html.Hr(),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        html.Output(
                            generate_config_code(register, variables=["path", "config_dict"])
                        ),
                        title="See code",
                    ),
                ],
                start_collapsed=True,
            ),
        ]

    @staticmethod
    def _get_objective_figure(
        _: Any, outputs: Dict[str, Dict[str, Dict[Any, Any]]], run: AbstractRun
    ) -> go.Figure:
        """
        Get the figure for the visualization of the objectives.

        Parameters
        ----------
        outputs : Dict[str, Dict[str, Dict[Any, Any]]]
            Raw outputs from the run.
        run : AbstractRun
            The selected run.

        Returns
        -------
        go.Figure
            The figure of the objectives.
        """
        objective_data = []
        for i, (metric, values) in enumerate(outputs["performances"].items()):
            mean_objective_values, std_objective_values = [], []
            for objective_values in values.values():
                mean_objective_values.append(np.nanmean(objective_values))
                std_objective_values.append(np.nanstd(objective_values))

            trace_kwargs = {
                "x": list(values.keys()),
                "y": mean_objective_values,
                "error_y": dict(
                    type="data",
                    symmetric=True,
                    array=std_objective_values,
                ),
                "name": metric,
                "fill": "tozeroy",
            }

            if i > 0:
                trace_kwargs.update({"yaxis": f"y{i+1}"})

            trace = go.Scatter(**trace_kwargs)
            objective_data.append(trace)

        layout_kwargs: Dict[str, Any] = {
            "margin": config.FIGURE_MARGIN,
            "xaxis": {"title": "Budget", "domain": [0.05 * len(run.get_objectives()), 1]},
        }

        # An axis is created for each objective now
        for id, objective in enumerate(run.get_objectives()):
            yaxis = "yaxis"
            if id > 0:
                # yaxis, yaxis2, yaxis3, ...
                yaxis = f"yaxis{id+1}"

            layout_kwargs[yaxis] = {
                # "title": objective.name,
                "titlefont": {"color": get_color(id)},
                "tickfont": {"color": get_color(id)},
                "range": [objective.lower, objective.upper],
            }

            if id > 0:
                layout_kwargs[yaxis].update(
                    {
                        "anchor": "free",
                        "overlaying": "y",
                        "side": "left",
                        "position": 0.05 * id,
                    }
                )

        objective_layout = go.Layout(
            **layout_kwargs,
            font=dict(size=config.FIGURE_FONT_SIZE),
        )
        objective_figure = go.Figure(data=objective_data, layout=objective_layout)
        save_image(objective_figure, "configure.pdf")

        return objective_figure

    @staticmethod
    def _get_configspace_figure(
        inputs: Any, outputs: Dict[str, str], run: AbstractRun
    ) -> go.Figure:
        """
        Get the figure for the visualization of the configuration space.

        Parameters
        ----------
        outputs : Dict[str, str]
            Raw outputs from the run.
        run : AbstractRun
            The selected run.

        Returns
        -------
        go.Figure
            The figure of the configuration space.
        """
        df = deserialize(outputs["cs_df"], dtype=pd.DataFrame)

        highlighted = df["highlighted"].values
        hp_names = run.configspace.get_hyperparameter_names()

        # Get highlighted column
        highlighted_df = df[df["highlighted"] == 1]

        data: defaultdict = defaultdict(dict)
        for hp_name in hp_names:
            data[hp_name]["values"] = df[hp_name].values
            data[hp_name]["label"] = hp_name
            data[hp_name]["range"] = VALUE_RANGE

            hp = run.configspace.get_hyperparameter(hp_name)
            tickvals, ticktext = get_hyperparameter_ticks(
                hp, additional_values=highlighted_df[hp_name].values, ticks=4, include_nan=True
            )

            data[hp_name]["tickvals"] = tickvals
            data[hp_name]["ticktext"] = ticktext

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=highlighted,
                    showscale=False,
                    colorscale=["rgba(255,255,255,0.1)", "red"],
                ),
                dimensions=list([d for d in data.values()]),
                labelangle=45,
                labelside="top",
            ),
            layout=dict(
                margin=dict(
                    t=150,
                    b=50,
                ),
                font=dict(size=config.FIGURE_FONT_SIZE),
            ),
        )

        return fig

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[Any]:  # type: ignore
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
        inputs
            Input and filter values from the user.
        outputs
            Raw output from the run.

        Returns
        -------
        List[Any]
            A list of the created tables containing output information.
        """
        config_id = inputs["config_id"]
        config = run.get_config(config_id)

        if run.path is not None:
            path = run.path / "configspace.json"
        else:
            assert run.prefix == "group"
            original_run = run.get_original_run(config_id)
            path = original_run.path

        return [
            create_table(outputs["overview_table_data"], head=False, striped=False),
            Configurations._get_objective_figure(inputs, outputs, run),
            create_table(outputs["performances_table_data"]),
            Configurations._get_configspace_figure(inputs, outputs, run),
            create_table(outputs["cs_table_data"]),
            str(path),
            str(config.get_dictionary()),
        ]
