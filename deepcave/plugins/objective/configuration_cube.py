from typing import Dict, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import Status
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
    id = "ccube"
    name = "Configuration Cube"
    icon = "fas fa-cube"
    activate_run_selection = True
    help = "docs/plugins/configuration_cube.rst"

    @staticmethod
    def get_input_layout(register):
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
                                "Combined budget means that the trial on the highest"
                                " evaluated budget is used.\n\n"
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
    def get_filter_layout(register):
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

    def load_inputs(self):
        return {
            "n_configs": {"min": 0, "max": 0, "marks": get_slider_marks(), "value": 0},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
        }

    def load_dependency_inputs(self, run, _, inputs):
        # Prepare objetives
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
            budget_value = budget_ids[-1]
        else:
            budget_value = inputs["budget_id"]["value"]

        budget = run.get_budget(budget_value)
        configs = run.get_configs(budget=budget)
        if n_configs_value == 0:
            n_configs_value = len(configs) - 1
        else:
            if n_configs_value > len(configs) - 1:
                n_configs_value = len(configs) - 1

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
                "max": len(configs) - 1,
                "marks": get_slider_marks(list(range(len(configs)))),
                "value": n_configs_value,
            },
            "hyperparameter_names": {
                "options": get_select_options(hp_names),
                "value": selected_hps,
            },
        }

    @staticmethod
    def process(run, inputs):
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])

        df = run.get_encoded_data(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, include_config_ids=True
        )
        return {"df": serialize(df)}

    @staticmethod
    def get_output_layout(register):
        return (dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT}),)

    @staticmethod
    def load_outputs(run, inputs, outputs):
        df = deserialize(outputs["df"], dtype=pd.DataFrame)
        hp_names = inputs["hyperparameter_names"]
        n_configs = inputs["n_configs"]
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)

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

        # Now we want to filter duplicates
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

                # Otherwise we have to replace
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
            "hovertext": [get_hovertext_from_config(run, config_id) for config_id in config_ids],
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
            layout = go.Layout({"scene": {**layout_kwargs}})
        else:
            if len(data) == 1:
                y = [0 for _ in x]
                trace = go.Scatter(x=x, y=y, **scatter_kwargs)
            elif len(data) == 2:
                trace = go.Scatter(x=x, y=y, **scatter_kwargs)
            else:
                trace = go.Scatter(x=[], y=[])
            layout = go.Layout(**layout_kwargs)

        figure = go.Figure(data=trace, layout=layout)
        figure.update_layout(dict(margin=config.FIGURE_MARGIN))
        save_image(figure, "configuration_cube.pdf")

        return figure
