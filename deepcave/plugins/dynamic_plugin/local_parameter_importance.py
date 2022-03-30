from typing import Any, Dict

import dash_bootstrap_components as dbc
from dash import html

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.runs import AbstractRun


class LocalParameterImportance(DynamicPlugin):
    id = "lpi"
    name = "Local Parameter Importance"
    icon = "fas fa-circle"
    description = """
        TODO
    """
    activate_run_selection = True

    # @staticmethod
    # def get_input_layout(register):
    #     return [
    #         html.Div(
    #             [
    #                 dbc.Label("Objective"),
    #                 dbc.Select(
    #                     id=register("objective", ["options", "value"]),
    #                     placeholder="Select objective ...",
    #                 ),
    #             ],
    #         ),
    #     ]

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]):
        pass
