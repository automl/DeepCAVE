from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from src import app
from src.plugins.dynamic_plugin import DynamicPlugin
from src.plugins.static_plugin import StaticPlugin
from src.utils.logs import get_logger
from src.runs.run import Status

from src.evaluators.fanova import fANOVA as _fANOVA


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
