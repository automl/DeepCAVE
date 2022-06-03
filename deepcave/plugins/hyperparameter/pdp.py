import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from pyPDP.algorithms.pdp import PDP

from deepcave import config
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import Status
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, get_hyperparameter_ticks, save_image

GRID_POINTS_PER_AXIS = 20
SAMPLES_PER_HP = 10
MAX_SAMPLES = 10000
MAX_SHOWN_SAMPLES = 100


class PartialDependencies(StaticPlugin):
    id = "pdp"
    name = "Partial Dependencies"
    icon = "far fa-grip-lines"
    help = "docs/plugins/partial_dependencies.rst"
    activate_run_selection = True

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
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter #1"),
                            dbc.Select(
                                id=register("hyperparameter_name_1", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter #2"),
                            dbc.Select(
                                id=register("hyperparameter_name_2", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Show confidence"),
                                    help_button("Displays the confidence bands."),
                                    dbc.Select(
                                        id=register("show_confidence", ["value", "options"])
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Show ICE curves"),
                                    help_button(
                                        "Displays the ICE curves from which the PDP curve is "
                                        "derivied."
                                    ),
                                    dbc.Select(id=register("show_ice", ["value", "options"])),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    def load_inputs(self):
        return {
            "show_confidence": {"options": get_select_options(binary=True), "value": "true"},
            "show_ice": {"options": get_select_options(binary=True), "value": "true"},
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)

        hp_names = run.configspace.get_hyperparameter_names()

        # Get selected values
        objective_value = inputs["objective_id"]["value"]
        budget_value = inputs["budget_id"]["value"]
        hp1_value = inputs["hyperparameter_name_1"]["value"]

        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
            hp1_value = hp_names[0]

        return {
            "objective_id": {"options": objective_options, "value": objective_value},
            "budget_id": {"options": budget_options, "value": budget_value},
            "hyperparameter_name_1": {
                "options": get_checklist_options(hp_names),
                "value": hp1_value,
            },
            "hyperparameter_name_2": {
                "options": get_checklist_options([None] + hp_names),
            },
        }

    @staticmethod
    def process(run, inputs):
        # Surrogate
        hp_names = run.configspace.get_hyperparameter_names()
        objective = run.get_objective(inputs["objective_id"])
        budget = run.get_budget(inputs["budget_id"])
        hp1 = inputs["hyperparameter_name_1"]
        hp2 = inputs["hyperparameter_name_2"]

        if objective is None:
            raise RuntimeError("Objective not found.")

        # Encode data
        df = run.get_encoded_data(
            objective,
            budget,
            specific=True,
            statuses=Status.SUCCESS,
        )

        X = df[hp_names].to_numpy()
        Y = df[objective.name].to_numpy()

        # Let's initialize the surrogate
        surrogate_model = RandomForestSurrogate(run.configspace, seed=0)
        surrogate_model.fit(X, Y)

        # Prepare the hyperparameters
        selected_hyperparameters = [hp1]
        if hp2 is not None and hp2 != "":
            selected_hyperparameters += [hp2]

        num_samples = SAMPLES_PER_HP * len(X)
        # We limit the samples to max 10k
        if num_samples > MAX_SAMPLES:
            num_samples = MAX_SAMPLES

        # And finally call PDP
        pdp = PDP.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            seed=0,
            num_grid_points_per_axis=GRID_POINTS_PER_AXIS,
            num_samples=num_samples,
        )

        x = pdp.x_pdp.tolist()
        y = pdp.y_pdp.tolist()

        # We have to cut the ICE curves because it's too much data
        x_ice = pdp._ice.x_ice.tolist()
        y_ice = pdp._ice.y_ice.tolist()

        if len(x_ice) > MAX_SHOWN_SAMPLES:
            x_ice = x_ice[:MAX_SHOWN_SAMPLES]
            y_ice = y_ice[:MAX_SHOWN_SAMPLES]

        return {
            "x": x,
            "y": y,
            "variances": pdp.y_variances.tolist(),
            "x_ice": x_ice,
            "y_ice": y_ice,
        }

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    def load_outputs(run, inputs, outputs):
        # Parse inputs
        hp1_name = inputs["hyperparameter_name_1"]
        hp1_idx = run.configspace.get_idx_by_hyperparameter_name(hp1_name)
        hp1 = run.configspace.get_hyperparameter(hp1_name)

        hp2_name = inputs["hyperparameter_name_2"]
        hp2_idx = None
        hp2 = None
        if hp2_name is not None and hp2_name != "":
            hp2_idx = run.configspace.get_idx_by_hyperparameter_name(hp2_name)
            hp2 = run.configspace.get_hyperparameter(hp2_name)

        show_confidence = inputs["show_confidence"]
        show_ice = inputs["show_ice"]

        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        # Parse outputs
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        sigmas = np.sqrt(np.asarray(outputs["variances"]))

        x_ice = np.asarray(outputs["x_ice"])
        y_ice = np.asarray(outputs["y_ice"])

        traces = []
        if hp2_idx is None:  # 1D
            # Add ICE curves
            if show_ice:
                for x_, y_ in zip(x_ice, y_ice):
                    traces += [
                        go.Scatter(
                            x=x_[:, hp1_idx],
                            y=y_,
                            line=dict(color=get_color(1, 0.1)),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    ]

            if show_confidence:
                traces += [
                    go.Scatter(
                        x=x[:, hp1_idx],
                        y=y + sigmas,
                        line=dict(color=get_color(0, 0.1)),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                ]

                traces += [
                    go.Scatter(
                        x=x[:, hp1_idx],
                        y=y - sigmas,
                        fill="tonexty",
                        fillcolor=get_color(0, 0.2),
                        line=dict(color=get_color(0, 0.1)),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                ]

            traces += [
                go.Scatter(
                    x=x[:, hp1_idx],
                    y=y,
                    line=dict(color=get_color(0, 1)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            ]

            tickvals, ticktext = get_hyperparameter_ticks(hp1)
            layout = go.Layout(
                {
                    "xaxis": {
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "title": hp1_name,
                    },
                    "yaxis": {
                        "title": objective_name,
                    },
                }
            )
        else:
            z = y
            if show_confidence:
                z = sigmas
            traces += [
                go.Contour(
                    z=z,
                    x=x[:, hp1_idx],
                    y=x[:, hp2_idx],
                    colorbar=dict(
                        title=objective_name if not show_confidence else "Confidence (1-Sigma)",
                    ),
                    hoverinfo="skip",
                )
            ]

            x_tickvals, x_ticktext = get_hyperparameter_ticks(hp1)
            y_tickvals, y_ticktext = get_hyperparameter_ticks(hp2)

            layout = go.Layout(
                dict(
                    xaxis=dict(tickvals=x_tickvals, ticktext=x_ticktext, title=hp1_name),
                    yaxis=dict(tickvals=y_tickvals, ticktext=y_ticktext, title=hp2_name),
                    margin=config.FIGURE_MARGIN,
                )
            )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "pdp.pdf")

        return figure
