from typing import Any, Dict, Type

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import app
from deepcave.evaluators.fanova import fANOVA as _fANOVA
from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.runs.run import Status
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)

"""
class Configspace(DynamicPlugin):
    @staticmethod
    def id():
        return "configspace"

    @staticmethod
    def name():
        return "Configspace"

    @staticmethod
    def position():
        return 2

    @staticmethod
    def category():
        return "Meta-Data Analysis"

    @staticmethod
    def description():
        return "especially lists all found configurations and which fidelities were used."

    @staticmethod
    def debug():
        return False

    def get_input_layout(self):
        return []

    def get_filter_layout(self):
        return []

    @staticmethod
    def process(run, inputs):

        # Config id | Fidelity #1 | Fidelity #2 | ...
        all_config_ids = []
        config_ids = {}
        for trial in run.history:
            if trial.status != Status.SUCCESS:
                continue

            if trial.config_id not in all_config_ids:
                all_config_ids.append(trial.config_id)

            if trial.budget not in config_ids:
                config_ids[trial.budget] = []

            if trial.config_id not in config_ids[trial.budget]:
                config_ids[trial.budget].append(trial.config_id)

        results = {}
        for config_id in all_config_ids:
            results[config_id] = []
            for fidelity in run.get_budgets():
                if config_id in config_ids[fidelity]:
                    results[config_id].append("YES")
                else:
                    results[config_id].append("")

        return {
            "fidelities": run.get_budgets(),
            "data": results
        }

    def get_output_layout(self):
        return [
            html.Div(id=self.register_output("output", "children"))
        ]

    def load_outputs(self, filters, raw_outputs):
        table_header = [
            html.Thead(html.Tr([
                html.Th("Config ID"),
                *[html.Th(fidelity) for fidelity in raw_outputs["fidelities"]]
            ]))
        ]

        rows = []
        for config_id, values in raw_outputs["data"].items():
            fidelity_cols = []
            for value in values:
                fidelity_cols.append(html.Td(value))

            rows.append(html.Tr([html.Td(config_id), *fidelity_cols]))

        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_header + table_body, bordered=True)

        return table
"""
