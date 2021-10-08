from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from deep_cave import app
from deep_cave.plugins.dynamic_plugin import DynamicPlugin
from deep_cave.plugins.static_plugin import StaticPlugin
from deep_cave.utils.logs import get_logger
from smac.tae import StatusType

from deep_cave.evaluators.fanova import fANOVA as _fANOVA


logger = get_logger(__name__)


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
        print("hi")
        rh = run.get_runhistory()

        # Config id | Fidelity #1 | Fidelity #2 | ...
        all_config_ids = []
        config_ids = {}
        for (config_id, _, _, budget), (_, _, status, _, _, _) in rh.data.items():
            if status != StatusType.SUCCESS:
                continue

            if config_id not in all_config_ids:
                all_config_ids.append(config_id)

            if budget not in config_ids:
                config_ids[budget] = []

            if config_id not in config_ids[budget]:
                config_ids[budget].append(config_id)

        results = {}
        for config_id in all_config_ids:
            results[config_id] = []
            for fidelity in run.get_fidelities():
                if config_id in config_ids[fidelity]:
                    results[config_id].append("YES")
                else:
                    results[config_id].append("")

        return {
            "fidelities": run.get_fidelities(),
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
