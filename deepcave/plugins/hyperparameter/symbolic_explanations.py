import signal

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from dash import dcc, html
from gplearn.genetic import SymbolicRegressor
from pyPDP.algorithms.pdp import PDP

from deepcave import config
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import Status
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, get_hyperparameter_ticks, save_image
from deepcave.utils.symbolic_regression import convert_symb, get_function_set

SR_TRAIN_POINTS_PER_AXIS = 20
SAMPLES_PER_HP = 10
MAX_SAMPLES = 10000
MAX_SHOWN_SAMPLES = 100


class SymbolicExplanations(StaticPlugin):
    id = "symbolic_explanations"
    name = "Symbolic Explanations"
    icon = "fas fa-subscript"
    help = "docs/plugins/partial_dependencies.rst"  # TODO
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
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Label("Parsimony coefficient"),
                                    help_button(
                                        "Penalizes the complexity of the resulting formulas."
                                    ),
                                    dcc.Slider(
                                        id=register("parsimony", "value", type=int),
                                        marks=dict([i, str(10**i)] for i in range(-8, 1)),
                                        min=-8,
                                        max=0,
                                        step=1,
                                        updatemode="drag",
                                    ),
                                ],
                            )
                        ],
                    )
                ],
                className="mb-3",
            ),
            html.Details(
                [
                    html.Summary("Additional options for symbolic regression configuration"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Generations"),
                                    help_button("The number of generations to evolve."),
                                    dbc.Input(
                                        id=register("generations", type=int),
                                        type="number",
                                        min=5,
                                        step=5,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Population Size"),
                                    help_button(
                                        "The number of formulas competing in each generation."
                                    ),
                                    dbc.Input(
                                        id=register("population_size", type=int),
                                        type="number",
                                        min=1000,
                                        step=1000,
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-3",
                        style={"marginTop": "0.8em"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Random seed"),
                                    help_button(
                                        "The random seed to be used in the symbolic regression."
                                    ),
                                    dbc.Input(
                                        id=register("random_seed", type=int), type="number", min=0
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Metric"),
                                    help_button(
                                        "The metric to evaluate the fitness of the formulas."
                                    ),
                                    dbc.Select(id=register("metric", ["value", "options"])),
                                ],
                                md=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            ),
        ]

    def load_inputs(self):
        return {
            "parsimony": {"value": "-4"},
            "generations": {"value": "10"},
            "population_size": {"value": "5000"},
            "random_seed": {"value": "0"},
            "metric": {
                "options": get_select_options(values=["rmse", "mse", "mean absolute error"]),
                "value": "rmse",
            },
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)

        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)

        hp_dict = run.configspace.get_hyperparameters_dict()
        hp_names_numerical = []
        for k, v in hp_dict.items():
            if not isinstance(v, CategoricalHyperparameter):
                hp_names_numerical.append(k)
        hp_names = hp_names_numerical

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
        parsimony = 10 ** inputs["parsimony"]
        generations = inputs["generations"]
        population_size = inputs["population_size"]
        random_seed = inputs["random_seed"]
        metric = inputs["metric"]

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
        idx1 = run.configspace.get_idx_by_hyperparameter_name(hp1)
        idxs = [idx1]
        if hp2 is not None and hp2 != "":
            selected_hyperparameters += [hp2]
            idx2 = run.configspace.get_idx_by_hyperparameter_name(hp2)
            idxs += [idx2]

        if len(selected_hyperparameters) < len(hp_names):
            num_samples = SAMPLES_PER_HP * len(X)
            # We limit the samples to max 10k
            if num_samples > MAX_SAMPLES:
                num_samples = MAX_SAMPLES

            # And finally call PDP
            pdp = PDP.from_random_points(
                surrogate_model,
                selected_hyperparameter=selected_hyperparameters,
                seed=0,
                num_grid_points_per_axis=SR_TRAIN_POINTS_PER_AXIS,
                num_samples=num_samples,
            )

            x = pdp.x_pdp[:, idxs].tolist()
            y = pdp.y_pdp.tolist()

        else:
            cs = surrogate_model.config_space
            random_samples = np.asarray(
                [
                    config.get_array()
                    for config in cs.sample_configuration(
                        SR_TRAIN_POINTS_PER_AXIS ** len(selected_hyperparameters)
                    )
                ]
            )
            x = random_samples.tolist()
            y = surrogate_model.predict(random_samples)[0]

        symb_params = dict(
            population_size=population_size,
            generations=generations,
            function_set=get_function_set(),
            metric=metric,
            parsimony_coefficient=parsimony,
            random_state=random_seed,
            verbose=1,
        )

        # run SR on samples
        symb_model = SymbolicRegressor(**symb_params)
        symb_model.fit(x, y)

        def handler(signo, frame):
            raise RuntimeError

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(6)  # seconds
        while True:
            try:
                conv_expr = convert_symb(
                    symb_model, n_decimals=3, hp_names=selected_hyperparameters
                )
            except:
                conv_expr = (
                    "The conversion of the expression failed. Please try another seed or increase "
                    "the parsimony hyperparameter to obtain a symbolic explanation."
                )

            y_symbolic = symb_model.predict(x).tolist()

            return {"x": x, "y": y_symbolic, "expr": str(conv_expr)}

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    def load_outputs(run, inputs, outputs):
        # Parse inputs
        hp1_name = inputs["hyperparameter_name_1"]
        hp1 = run.configspace.get_hyperparameter(hp1_name)

        hp2_name = inputs["hyperparameter_name_2"]
        hp2 = None
        if hp2_name is not None and hp2_name != "":
            hp2 = run.configspace.get_hyperparameter(hp2_name)

        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        # Parse outputs
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        expr = outputs["expr"]

        traces = []
        if hp2 is None:  # 1D
            traces += [
                go.Scatter(
                    x=x[:, 0],
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
                    "title": f"{objective.name} = {expr}",
                }
            )
        else:
            z = y
            traces += [
                go.Contour(
                    z=z,
                    x=x[:, 0],
                    y=x[:, 1],
                    colorbar=dict(
                        title=objective_name,
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
                    title=f"{objective.name} = {expr}",
                )
            )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "symbolic_explanation.pdf")

        return figure
