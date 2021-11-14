from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple
import os
import json
import copy
from collections import defaultdict

import pandas as pd

from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
from plotly.graph_objects import Figure
from dash import html
from dash.development.base_component import Component
from ConfigSpace import ConfigurationSpace
from dash.exceptions import PreventUpdate

from deepcave import app, c, rc
from deepcave.runs.handler import handler
from deepcave.utils.logs import get_logger
from deepcave.plugins.plugin import Plugin


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
        def plugin_output_update(_, *inputs_list):
            """
            Parameters:
                *inputs_list: Values from user.
            """

            # The results from the last run
            last_inputs = c.get("last_inputs", self.id())
            last_inputs_key = self._dict_as_key(
                last_inputs,
                remove_filters=True)
            last_raw_outputs = {}

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)

            # Special case: If run selection is active
            # Don't update anything if the inputs haven't changed
            if self.__class__.activate_run_selection():
                if "run_name" not in inputs or inputs["run_name"]["value"] is None:
                    raise PreventUpdate()

                run_names = [inputs["run_name"]["value"]]

                # Also:
                # Remove `run_name` from last_inputs_key because
                # we don't want the run names included.
                _last_inputs = c.get("last_inputs", self.id())
                if _last_inputs is not None:

                    del _last_inputs["run_name"]
                    last_inputs_key = self._dict_as_key(
                        _last_inputs,
                        remove_filters=True)
            else:
                run_names = self.runs.keys()

            # Get latest raw outputs
            for run_name in run_names:
                last_raw_outputs[run_name] = rc[run_name].get(last_inputs_key)

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
        inputs_key = self._dict_as_key(inputs, remove_filters=True)

        # Special case again
        # Only process the selected run
        if self.__class__.activate_run_selection():
            runs = {}
            run_name = inputs["run_name"]["value"]
            runs[run_name] = self.runs[run_name]

            # Also:
            # Remove `run_name` from last_inputs_key because
            # we don't want the run names included.
            _inputs = inputs.copy()
            del _inputs["run_name"]

            inputs_key = self._dict_as_key(_inputs, remove_filters=True)
        else:
            runs = self.runs

        for name, run in runs.items():
            if name in raw_outputs:
                # If output is already set, we are good to go.
                if raw_outputs[name] is not None:
                    continue

            run_outputs = rc[name].get(inputs_key)
            if run_outputs is None:
                logger.debug(f"Process {name}.")
                run_outputs = self.process(run, inputs)

                # Here's the thing:
                # We have to remove `run_name` from the inputs completely

                # Cache it
                rc[name].set(inputs_key, value=run_outputs)
            else:
                logger.debug(f"Found outputs from {name} in cache.")

            raw_outputs[name] = run_outputs

        # Cache last inputs
        c.set("last_inputs", self.id(), value=inputs)

        return self._process_raw_outputs(inputs, raw_outputs)

    def __call__(self):
        return super().__call__(False)
