#  noqa: D400
"""
# SymbolicExplanations

This module provides utilities for generating Symbolic Explanations.

Provided utilities include getting input and output layout,
processing the data and loading the outputs.

## Classes
    - SymbolicExplanations: Leverage Symbolic Explanations to obtain a formula and plot it.

## Constants
    GRID_POINTS_PER_AXIS : int
    SAMPLES_PER_HP : int
    MAX_SAMPLES : int
    MAX_SHOWN_SAMPLES : int
"""

from typing import Any, Callable, Dict, List, Union

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from dash import dcc, html
from gplearn.genetic import SymbolicRegressor
from pyPDP.algorithms.pdp import PDP

from deepcave import config
from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
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
    """
    Generate Symbolic Explanations.

    Provided utilities include getting input and output layout,
    processing the data and loading the outputs.
    """

    id = "symbolic_explanations"
    name = "Symbolic Explanations"
    icon = "fas fa-subscript"
    help = "docs/plugins/symbolic_explanations.rst"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Union[dbc.Row, html.Details]]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Union[dbc.Row, html.Details]
            The layout for the input block.
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
                                        "Penalizes the complexity of the resulting formulas. The "
                                        "higher the value, the higher the penalty on the "
                                        "complexity will be, resulting in simpler formulas."
                                    ),
                                    dcc.Slider(
                                        id=register("parsimony", "value", type=int),
                                        marks=dict((i, str(10**i)) for i in range(-8, 1)),
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
                                        min=1,
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
                                        min=1,
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

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the content for the defined inputs in 'get_input_layout'.

        This method is necessary to pre-load contents for the inputs.
        If the plugin is called for the first time, or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Content to be filled.
        """
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

    def load_dependency_inputs(self, run, previous_inputs, inputs) -> Dict[str, Any]:  # type: ignore # noqa: E501
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned. The returned dictionary
        will be merged with the inputs.

        Parameters
        ----------
        run
            The selected run.
        inputs
            Current content of the inputs.
        previous_inputs
            Previous content of the inputs.
            Not used in this specific function.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the changes.
        """
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
        hp2_value = inputs["hyperparameter_name_2"]["value"]

        if objective_value is None:
            objective_value = objective_ids[0]
        if budget_value is None:
            budget_value = budget_ids[-1]
        if hp1_value is None:
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
                "value": hp2_value,
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and the input data.

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
        run
            The run to process.
        inputs
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.

        Raises
        ------
        RuntimeError
            If the objective is None.
        """
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

        num_samples = SAMPLES_PER_HP * len(X)
        # The samples are limited to max 10k
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

        x_pdp = pdp.x_pdp
        y_pdp = pdp.y_pdp.tolist()
        pdp_variances = pdp.y_variances.tolist()

        x_ice = pdp._ice.x_ice.tolist()
        y_ice = pdp._ice.y_ice.tolist()

        # The ICE curves have to be cut because it's too much data
        if len(x_ice) > MAX_SHOWN_SAMPLES:
            x_ice = x_ice[:MAX_SHOWN_SAMPLES]
            y_ice = y_ice[:MAX_SHOWN_SAMPLES]

        if len(selected_hyperparameters) < len(hp_names):
            # If number of hyperparameters to explain is smaller than number of hyperparameters
            # optimizes, use PDP to train the symbolic explanation
            x_symbolic = x_pdp
            y_train = y_pdp
        else:
            # Else, use random samples evaluated with the surrogate model to train the symbolic
            # explanation
            cs = surrogate_model.config_space
            random_samples = np.asarray(
                [
                    config.get_array()
                    for config in cs.sample_configuration(
                        SR_TRAIN_POINTS_PER_AXIS ** len(selected_hyperparameters)
                    )
                ]
            )
            x_symbolic = random_samples
            y_train = surrogate_model.predict(random_samples)[0]

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
        symb_model.fit(x_symbolic[:, idxs], y_train)

        try:
            conv_expr = (
                f"{objective.name} = "
                f"{convert_symb(symb_model, n_decimals=3, hp_names=selected_hyperparameters)}"
            )
        except Exception as e:
            conv_expr = (
                "Conversion of the expression failed. Please try another seed or increase "
                f"the parsimony hyperparameter: {e}"
            )

        if len(conv_expr) > 150:
            conv_expr = (
                "Expression is too long to display. Please try another seed or increase "
                "the parsimony hyperparameter."
            )

        y_symbolic = symb_model.predict(x_symbolic[:, idxs]).tolist()

        return {
            "x": x_pdp.tolist(),
            "x_symbolic": x_symbolic.tolist(),
            "y": y_pdp,
            "y_symbolic": y_symbolic,
            "expr": conv_expr,
            "variances": pdp_variances,
            "x_ice": x_ice,
            "y_ice": y_ice,
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[dcc.Graph]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dcc.Graph]
            Layout for the output block.
        """
        return [
            dcc.Graph(
                register("symb_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
            dcc.Graph(
                register("pdp_graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
        ]

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[go.Figure]:  # type: ignore
        """
        Read the raw data and prepare it for the layout.

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
        List[go.Figure]
            The figure of the Symbolic Explanation and the Partial Dependency Plot (PDP) leveraged
            for training in the case that the number of hyperparameters to be explained is smaller
            than the number of hyperparameters that was optimized, else, a Partial Dependency Plot
            (PDP) for comparison.
        """
        hp1_name = inputs["hyperparameter_name_1"]
        hp1_idx = run.configspace.get_idx_by_hyperparameter_name(hp1_name)
        hp1 = run.configspace.get_hyperparameter(hp1_name)
        selected_hyperparameters = [hp1]

        hp2_name = inputs["hyperparameter_name_2"]
        hp2_idx = None
        hp2 = None
        if hp2_name is not None and hp2_name != "":
            hp2_idx = run.configspace.get_idx_by_hyperparameter_name(hp2_name)
            hp2 = run.configspace.get_hyperparameter(hp2_name)
            selected_hyperparameters += [hp2]

        hp_names = run.configspace.get_hyperparameter_names()
        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        # Parse outputs
        x_symbolic = np.asarray(outputs["x_symbolic"])
        y_symbolic = np.asarray(outputs["y_symbolic"])
        expr = outputs["expr"]

        traces1 = []
        if hp2 is None:  # 1D
            traces1 += [
                go.Scatter(
                    x=x_symbolic[:, hp1_idx],
                    y=y_symbolic,
                    line=dict(color=get_color(0, 1)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            ]

            tickvals, ticktext = get_hyperparameter_ticks(hp1)
            layout1 = go.Layout(
                {
                    "xaxis": {
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "title": hp1_name,
                    },
                    "yaxis": {
                        "title": objective_name,
                    },
                    "title": expr,
                    "font": dict(size=config.FIGURE_FONT_SIZE - 4),
                }
            )
        else:
            z = y_symbolic
            traces1 += [
                go.Contour(
                    z=z,
                    x=x_symbolic[:, hp1_idx],
                    y=x_symbolic[:, hp2_idx],
                    colorbar=dict(
                        title=objective_name,
                    ),
                    hoverinfo="skip",
                )
            ]

            x_tickvals, x_ticktext = get_hyperparameter_ticks(hp1)
            y_tickvals, y_ticktext = get_hyperparameter_ticks(hp2)

            layout1 = go.Layout(
                dict(
                    xaxis=dict(tickvals=x_tickvals, ticktext=x_ticktext, title=hp1_name),
                    yaxis=dict(tickvals=y_tickvals, ticktext=y_ticktext, title=hp2_name),
                    margin=config.FIGURE_MARGIN,
                    title=expr,
                    font=dict(size=config.FIGURE_FONT_SIZE - 4),
                )
            )

        figure1 = go.Figure(data=traces1, layout=layout1)
        save_image(figure1, "symbolic_explanation.pdf")

        if len(selected_hyperparameters) < len(hp_names):
            pdp_title = "Partial Dependency leveraged for training of Symbolic Explanation:"
        else:
            pdp_title = "Partial Dependency for comparison:"

        figure2 = PartialDependencies.get_pdp_figure(
            run,
            inputs,
            outputs,
            show_confidence=False,
            show_ice=False,
            title=pdp_title,
            fontsize=config.FIGURE_FONT_SIZE - 4,
        )

        return [figure1, figure2]
