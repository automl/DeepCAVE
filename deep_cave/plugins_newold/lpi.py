from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from deep_cave import app
from deep_cave.plugins.plugin import Plugin
from deep_cave.utils.logs import get_logger

#from evaluators.lpi import LPI as _LPI
import itertools as it
import logging
from collections import OrderedDict

import ConfigSpace
import numpy as np
import pandas as pd
import pyrfr.regression as reg
import pyrfr.util
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, Constant, OrdinalHyperparameter

logger = get_logger(__name__)


class LPI(Plugin):
    @staticmethod
    def id():
        return "lpi"

    @staticmethod
    def name():
        return "LPI"

    @staticmethod
    def category():
        return "Hyperparameter Importance"

    def get_input_layout(self):
        return [
            dbc.Label("Hyperparameters"),
            dbc.Checklist(id=self.register_input(
                "hyperparameters", ["options", "value"]))
        ]

    def get_output_layout(self):
        return [
            dcc.Graph(self.register_output("graph", "figure"))
        ]

    def load_input(self, run):
        hp_names = run.cs.get_hyperparameter_names()

        return {
            "hyperparameters": {
                "options": [{"label": name, "value": idx} for idx, name in enumerate(hp_names)],
                "value": [idx for idx in range(len(hp_names))]
            }
        }

    def load_output(self, run, **inputs):
        fidelities = run.get_fidelities()
        hyperparameter_ids = sorted(inputs["hyperparameters"]["value"])
        hp_names = run.cs.get_hyperparameter_names()

        '''

        # Collect data
        data = []
        for fidelity in fidelities:

            X, y, mapping, id_mapping, _, _, _ = run.transform_configs(
                fidelity,
                hyperparameter_ids=hyperparameter_ids
            )

            lpi = _LPI(
                run.get_runhistory(),
                fidelity,
                selected_hyperparameters,
            )

            fanova = _fANOVA(
                pd.DataFrame(data=X),
                np.array(y),
                config_space=config_space,
                # n_trees=n_trees,
                # min_samples_split=min_samples_split
            )

            x = []
            y = []
            error_y = []
            for id in selected_hp_ids:
                new_id = hp_id_mapping[id]

                results = fanova.quantify_importance([new_id])
                results = list(results.values())[0]
                importance = results["total importance"]
                std = results["total std"]

                x += [hp_names[id]]
                y += [importance]
                error_y += [0]

            data += [go.Bar(name=fidelity, x=x, y=y, error_y_array=error_y)]
        '''
        fig = go.Figure(data={})
        fig.update_layout(barmode='group')

        return {
            "graph": fig,
        }
