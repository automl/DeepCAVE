from typing import Optional

import dash_bootstrap_components as dbc
import pandas as pd
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, NormalIntegerHyperparameter, \
    UniformFloatHyperparameter, NormalFloatHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter, Constant
from dash import html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


class Configurations(DynamicPlugin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def id() -> str:
        return "configurations"

    @staticmethod
    def name() -> str:
        return "Configurations"

    @staticmethod
    def position() -> int:
        return 5

    @staticmethod
    def category() -> Optional[str]:
        return "General"

    @staticmethod
    def activate_run_selection() -> bool:
        return True

    @staticmethod
    def process(run, inputs):

        configspace = {
            "Hyperparameter": [],
            "Possible Values": [],
            "Default": [],
            "Log": [],
        }

        for hp_name, hp in run.configspace.get_hyperparameters_dict().items():

            log = False
            value = None
            if isinstance(hp, UniformIntegerHyperparameter) or isinstance(hp,
                                                                          NormalIntegerHyperparameter) or isinstance(hp,
                                                                                                                     UniformFloatHyperparameter) or isinstance(
                hp, NormalFloatHyperparameter):
                value = str([hp.lower, hp.upper])
                log = hp.log
            elif isinstance(hp, CategoricalHyperparameter):
                value = ", ".join([str(i) for i in hp.choices])
            elif isinstance(hp, OrdinalHyperparameter):
                value = ", ".join([str(i) for i in hp.sequence])
            elif isinstance(hp, Constant):
                value = str(hp.value)

            default = str(hp.default_value)
            log = str(log)

            configspace["Hyperparameter"].append(hp_name)
            configspace["Possible Values"].append(value)
            configspace["Default"].append(default)
            configspace["Log"].append(log)

        # Get best cost across all objectives, highest budget
        cost, config = run.get_min_cost()

        best_config = {
            "Hyperparameter": [],
            "Value": []
        }

        for hp_name, value in config.items():
            best_config["Hyperparameter"].append(hp_name)
            best_config["Value"].append(value)

        return {
            "configspace": configspace,
            "best_config": best_config,
            "min_cost": cost,
        }

    @staticmethod
    def get_output_layout(register):
        return [
            html.H3("Configuration Space"),
            html.Div(id=register("configspace", "children")),
            html.Hr(),

            html.H3("Best Configuration"),
            html.Div(id=register("best_config", "children")),
            html.Div(id=register("min_cost", "children")),
        ]

    @staticmethod
    def load_outputs(inputs, outputs, _):
        run_name = inputs["run_name"]["value"]
        outputs = outputs[run_name]

        def create_table(output): return dbc.Table.from_dataframe(
            pd.DataFrame(output), striped=True, bordered=True
        )

        return [
            create_table(outputs["configspace"]),
            create_table(outputs["best_config"]),
            f"With normalized cost: {outputs['min_cost']}"
        ]
