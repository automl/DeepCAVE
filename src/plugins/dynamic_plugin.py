from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple
import os
import json
import copy
from collections import defaultdict

import pandas as pd

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from plotly.graph_objects import Figure
import dash_html_components as html
from dash.development.base_component import Component
import dash_table
from ConfigSpace import ConfigurationSpace
from dash.exceptions import PreventUpdate

from src import app, c, rc
from src.runs.handler import handler
from src.utils.logs import get_logger
from src.plugins.plugin import Plugin


logger = get_logger(__name__)


class DynamicPlugin(Plugin):
    def __init__(self):
        super().__init__()

    def register_callbacks(self):
        super().register_callbacks()

        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), 'n_clicks')]
        for id, attribute, _ in self.inputs:
            inputs.append(
                Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_output_update(state, *inputs_list):
            """
            Parameters:
                *inputs_list: Values from user.
            """

            # The results from the last run
            last_inputs = c.get("last_inputs", self.id())
            last_raw_outputs = {}
            for name in handler.get_run_names():
                last_raw_outputs[name] = rc[name].get(
                    self._dict_as_key(last_inputs, remove_filters=True))

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)

            # Check if inputs changed.
            inputs_changed, filters_changed = self._inputs_changed(
                inputs, last_inputs)

            logger.debug(f"Inputs changed: {inputs_changed}")
            logger.debug(f"Filters changed: {filters_changed}")

            # If inputs changed, we have to process again.
            if inputs_changed:
                return self._get_outputs(inputs)
            else:
                return self._get_outputs(inputs, last_raw_outputs)

    def _get_outputs(self, inputs, raw_outputs={}):
        for name, run in handler.get_runs().items():

            if name in raw_outputs:
                # If output is already set, we are good to go.
                if raw_outputs[name] is not None:
                    continue

            run_outputs = rc[name].get(
                self._dict_as_key(inputs, remove_filters=True))
            if run_outputs is None:
                logger.debug(f"Process {name}.")
                run_outputs = self.process(run, inputs)

                # Cache it
                rc[name].set(
                    self._dict_as_key(inputs, remove_filters=True),
                    value=run_outputs
                )
            else:
                logger.debug(f"Found outputs from {name} in cache.")

            raw_outputs[name] = run_outputs

        # Cache last inputs
        c.set("last_inputs", self.id(), value=inputs)

        return self._process_raw_outputs(inputs, raw_outputs)

    def __call__(self):
        return super().__call__(False)
