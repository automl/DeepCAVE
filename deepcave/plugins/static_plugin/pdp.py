import ConfigSpace as CS
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from pyPDP.algorithms.pdp import PDP
from pyPDP.utils.utils import config_list_to_array

from deepcave.evaluators.epm.surrogates import RandomForestSurrogate
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.runs import AbstractRun, Status
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options
from deepcave.utils.logs import get_logger

logger = get_logger("PDPPlugin")


class PDPPlugin(StaticPlugin):
    id = "pdp"
    name = "Partial Dependence Plot"
    icon = "far fa-grip-lines"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective", ["options", "value"]),
                        placeholder="Select objective ...",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Budget"),
                    dbc.Select(
                        id=register("budget", ["options", "value"]), placeholder="Select budget ..."
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
                                id=register("hyperparameter1", ["options", "value"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter #2"),
                            dbc.Select(
                                id=register("hyperparameter2", ["options", "value"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # html.Div(
            #    [
            #        dbc.Label("Number of Samples"),
            #        dbc.Input(id=register("num_samples", "value")),
            #    ]
            # ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Show confidence"),
                    dbc.Select(id=register("confidences", ["options", "value"])),
                ]
            )
        ]

    def load_inputs(self):
        return {
            "objective": {"value": None},
            "budget": {"value": None},
            "hyperparameter1": {"value": None},
            "hyperparameter2": {"value": None},
            "confidences": {"options": get_select_options(binary=True), "value": "true"},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):

        objective_names = selected_run.get_objective_names()
        objective_options = get_select_options(objective_names)

        budgets = selected_run.get_budgets(human=True)
        budget_options = get_select_options(budgets, range(len(budgets)))

        hp_names = selected_run.configspace.get_hyperparameter_names()

        # Get selected values
        objective_value = inputs["objective"]["value"]
        budget_value = inputs["budget"]["value"]
        hp1_value = inputs["hyperparameter1"]["value"]

        if objective_value is None:
            objective_value = objective_names[0]
            budget_value = budget_options[-1]["value"]
            hp1_value = hp_names[0]

        new_inputs = {
            "objective": {"options": objective_options, "value": objective_value},
            "budget": {"options": budget_options, "value": budget_value},
            "hyperparameter1": {"options": get_checklist_options(hp_names), "value": hp1_value},
            "hyperparameter2": {
                "options": get_checklist_options([None] + hp_names),
            },
        }

        update_dict(inputs, new_inputs)
        return inputs

    @staticmethod
    def process(run: AbstractRun, inputs):
        # Surrogate
        objective_name = inputs["objective"]["value"]
        objective = run.get_objective(objective_name)
        budget_id = int(inputs["budget"]["value"])
        budget = run.get_budget(budget_id)
        hp1 = inputs["hyperparameter1"]["value"]
        hp2 = inputs["hyperparameter2"]["value"]

        if objective is None:
            raise RuntimeError("Objective not found.")

        # Encode data
        X, Y, _ = run.get_encoded_configs(
            [objective], budget, statuses=[Status.SUCCESS], encode_y=False
        )

        # Let's initialize the surrogate
        surrogate_model = RandomForestSurrogate(run.configspace)
        surrogate_model.fit(X, Y)

        # Prepare the hyperparameters
        selected_hyperparameters = [hp1]
        if hp2 is not None:
            selected_hyperparameters += [hp2]

        # And finally call PDP
        pdp = PDP.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            seed=0,
        )

        return {
            "x": pdp.x_pdp.tolist(),
            "y": pdp.y_pdp.tolist(),
            "variances": pdp.y_variances.tolist(),
        }

    @staticmethod
    def get_output_layout(register):
        return [dcc.Graph(register("graph", "figure"))]

    def load_outputs(self, inputs, outputs, run):
        # Parse inputs
        hp1_name = inputs["hyperparameter1"]["value"]
        hp2_name = inputs["hyperparameter2"]["value"]
        hp1_idx = run.configspace.get_idx_by_hyperparameter_name(hp1_name)
        if hp2_name is not None:
            hp2_idx = run.configspace.get_idx_by_hyperparameter_name(hp2_name)
        else:
            hp2_idx = None
        show_confidences = inputs["confidences"]["value"] == "true"
        objective_name = inputs["objective"]["value"]

        # Parse outputs
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        sigmas = np.sqrt(np.asarray(outputs["variances"]))

        fig = go.Figure()
        if hp2_idx is None:  # 1D
            fig.add_trace(
                go.Scatter(
                    name="PDP",
                    x=x[:, hp1_idx],
                    y=y,
                )
            )
            if show_confidences:
                fig.add_trace(
                    go.Scatter(
                        name="1-Sigma",
                        x=x[:, hp1_idx],
                        y=y + sigmas,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        name="1-Sigma",
                        x=x[:, hp1_idx],
                        y=y - sigmas,
                    )
                )
            fig.update_layout(
                title="1D PDP" + " with confidences" * show_confidences,
                xaxis_title=hp1_name,
                yaxis_title=objective_name,
            )
        elif len(selected_hyperparameters) == 2:  # 2D
            num_grid_points_per_axis = inputs["num_grid_points_per_axis"]["value"]
            if show_confidences:
                fig.add_trace(
                    go.Contour(
                        z=np.reshape(sigmas, (num_grid_points_per_axis, num_grid_points_per_axis)),
                        # x=[-9, -6, -5, -3, -1],  # horizontal axis
                        # y=[0, 1, 4, 5, 7]  # vertical axis
                    )
                )
                fig.update_layout(
                    title="2D PDP (Confidences)",
                    xaxis_title=selected_hyperparameters[0],
                    yaxis_title=selected_hyperparameters[1],
                )
            else:
                fig.add_trace(
                    go.Contour(
                        z=np.reshape(y, (num_grid_points_per_axis, num_grid_points_per_axis)),
                        # x=[-9, -6, -5, -3, -1],  # horizontal axis
                        # y=[0, 1, 4, 5, 7]  # vertical axis
                    )
                )
                fig.update_layout(
                    title="2D PDP (Values)",
                    xaxis_title=selected_hyperparameters[0],
                    yaxis_title=selected_hyperparameters[1],
                )
        else:
            pass

        return [fig]
