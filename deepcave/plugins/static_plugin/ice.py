import ConfigSpace as CS
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from pyPDP.algorithms.ice import ICE
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.utils import config_list_to_array

from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.data_structures import update_dict
from deepcave.utils.layout import get_checklist_options, get_select_options
from deepcave.utils.logs import get_logger

logger = get_logger("ICEPlugin")


class ICEPlugin(StaticPlugin):
    id = "ice"
    name = "ICE"
    icon = "far fa-grip-lines"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register):
        return [
            html.Div(
                [
                    dbc.Label("Selected Hyperparameters"),
                    dbc.Select(id=register("hyperparameters", ["options", "value"])),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(id=register("objective", ["options", "value"])),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Number of Grid Points per Axis"),
                    dbc.Input(id=register("num_grid_points_per_axis", "value")),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Number of Samples"),
                    dbc.Input(id=register("num_samples", "value")),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register):
        return []

    def load_inputs(self):
        return {
            "hyperparameters": {"options": get_select_options(), "value": []},
            "objective": {
                "options": get_select_options(),
                "value": [],
            },
            "num_grid_points_per_axis": {"value": 20},
            "num_samples": {"value": 1000},
        }

    def load_dependency_inputs(self, previous_inputs, inputs, selected_run=None):
        hp_names = selected_run.configspace.get_hyperparameter_names()
        objective_names = selected_run.get_objective_names()
        objective_ids = list(range(len(objective_names)))

        current_objective = inputs["objective"]["value"]
        if current_objective is None:
            current_objective = objective_ids[0]

        current_hp = inputs["hyperparameters"]["value"]
        if current_hp is None:
            current_hp = hp_names[0]

        new_inputs = {
            "hyperparameters": {"options": get_select_options(hp_names), "value": current_hp},
            "objective": {
                "options": get_select_options(objective_names, objective_ids),
                "value": current_objective,
            },
        }
        update_dict(inputs, new_inputs)

        num_grid_points_per_axis = inputs["num_grid_points_per_axis"]["value"]
        num_samples = inputs["num_samples"]["value"]

        # Reset invalid values
        # Num Samples
        try:
            int(num_samples)
        except (TypeError, ValueError):
            inputs["num_samples"]["value"] = previous_inputs["num_samples"]["value"]

        # Num Grid Points per Axis
        try:
            int(num_grid_points_per_axis)
        except (TypeError, ValueError):
            inputs["num_grid_points_per_axis"]["value"] = previous_inputs[
                "num_grid_points_per_axis"
            ]["value"]

        return inputs

    @staticmethod
    def process(run: AbstractRun, inputs):
        # Surrogate
        objective_id = int(inputs["objective"]["value"])

        logger.debug("Initialize Surrogate")
        surrogate_model = GaussianProcessSurrogate(run.configspace)
        logger.debug("Get y")
        xy = [
            (
                CS.Configuration(run.configspace, values=run.get_config(trial.config_id)),
                trial.costs[objective_id],
            )
            for trial in run.get_trials()
            if trial.costs[objective_id] is not None
        ]
        logger.debug("Get configs")
        configs, y = list(zip(*xy))
        logger.debug("Get X")
        X = config_list_to_array(list(configs))
        logger.debug("Fit Surrogate")
        surrogate_model.fit(X, y)

        # ICE
        selected_hyperparameters = inputs["hyperparameters"]["value"]
        num_samples = inputs["num_samples"]["value"]
        num_grid_points_per_axis = inputs["num_grid_points_per_axis"]["value"]
        logger.debug(
            f"Create ice with {num_grid_points_per_axis} grid points per axis and {num_samples} "
            f"for hyperparameters: {selected_hyperparameters}"
        )
        ice = ICE.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            num_samples=int(num_samples),
            num_grid_points_per_axis=int(num_grid_points_per_axis),
        )
        logger.debug("Returning...")
        return {
            "x": ice.x_ice.tolist(),
            "y": ice.y_ice.tolist(),
            # "variances": ice.y_variances.tolist(),
        }

    @staticmethod
    def get_output_layout(register):
        return [dcc.Graph(register("graph", "figure"))]

    def load_outputs(self, inputs, outputs, run):
        # Parse inputs
        selected_hyperparameters = inputs["hyperparameters"]["value"]

        hyperparameter_idx = [
            run.configspace.get_idx_by_hyperparameter_name(hp) for hp in selected_hyperparameters
        ]

        # Parse outputs
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])

        fig = go.Figure()
        if len(selected_hyperparameters) == 1:  # 1D
            for x_row, y_row in zip(x, y):
                fig.add_trace(
                    go.Scatter(
                        name="ICE",
                        x=x_row[:, hyperparameter_idx[0]],
                        y=y_row,
                    )
                )
            fig.update_layout(
                title="1D ICE-Curves",
                xaxis_title=selected_hyperparameters[0],
                yaxis_title=run.get_objective(int(inputs["objective"]["value"]))["name"],
            )
        else:
            pass

        return [fig]
