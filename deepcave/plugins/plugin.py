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

from deepcave import app
from deepcave import c, rc
from deepcave.utils.logs import get_logger
from deepcave.layouts.layout import Layout
from deepcave.runs.handler import handler


logger = get_logger(__name__)


class Plugin(Layout):
    def __init__(self):
        self.inputs = []
        self.outputs = []

        # Processing right now?
        self.blocked = False

        # Alert texts
        self.alert_text = ""
        self.alert_color = "success"
        self.alert_update_required = False

        super().__init__()

    @staticmethod
    @abstractmethod
    def id():
        raise NotImplementedError()

    @staticmethod
    def category():
        return None

    @staticmethod
    def position():
        return 99999

    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()

    @staticmethod
    def description() -> str:
        return ''

    @staticmethod
    def button_caption():
        return "Process"

    @staticmethod
    def debug():
        """No caching is used."""
        return False

    @staticmethod
    def check_requirements(runs):
        """
        Returns either bool or str. If str, it is shown to the user.
        """
        return True

    def register_input(self, id, attributes=["value"], filter=False):
        if isinstance(attributes, str):
            attributes = [attributes]

        for attribute in attributes:
            key = (id, attribute, filter)
            if key not in self.inputs:
                self.inputs.append(key)

        # We have to rearrange the inputs because `State`
        # must follow all `Input`. Since all filters are `Input`, we have to
        # shift them to the front.
        self.inputs.sort(key=lambda x: x[2], reverse=True)

        return self.get_internal_input_id(id)

    def register_output(self, id, attribute="value", mpl=False):
        assert isinstance(attribute, str)

        if mpl:
            id = id + "-mpl"

        key = (id, attribute, mpl)
        if key not in self.outputs:
            self.outputs.append(key)

        return self.get_internal_output_id(id)

    def get_internal_id(self, id):
        return self.id() + "-" + id

    def get_internal_input_id(self, id):
        return self.id() + "-" + id + "-input"

    def get_internal_output_id(self, id):
        return self.id() + "-" + id + "-output"

    def register_callbacks(self):
        # We have to call the output layout one time to register
        # the values
        # Problem: Inputs/Outputs can't be changed afterwards anymore.

        self.__class__.get_input_layout(self.register_input)
        self.__class__.get_filter_layout(
            lambda a, b: self.register_input(a, b, filter=True))
        self.__class__.get_output_layout(self.register_output)
        self.__class__.get_mpl_output_layout(
            lambda a, b: self.register_output(a, b, mpl=True))

        # Handles the initial and the cashed input values
        outputs = []
        # [Input(self.get_internal_id("update-button"), 'n_clicks')]
        inputs = []

        # Define also inputs if they are declared as interactive
        for id, attribute, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        for id, attribute, _ in self.inputs:
            outputs.append(Output(self.get_internal_input_id(id), attribute))

        if len(outputs) > 0:
            @app.callback(outputs, inputs)
            def plugin_input_update(*inputs_list):
                init = True
                # Simple check if page was loaded for the first time
                for input in inputs_list:
                    if input is not None:
                        init = False
                        break

                # Reload our inputs
                if init:
                    inputs = c.get("last_inputs", self.id())

                    if inputs is None:
                        inputs = self.__class__.load_inputs(self.runs)

                        # Set not used inputs
                        for (id, attribute, _) in self.inputs:
                            if id not in inputs:
                                inputs[id] = {}

                            if attribute not in inputs[id]:
                                inputs[id][attribute] = None
                else:
                    # Map the list `inputs` to a dict.
                    inputs = self._list_to_dict(inputs_list)

                    if len(self.previous_inputs) == 0:
                        self.previous_inputs = inputs.copy()

                    # How to update only parameters which have a dependency?
                    user_dependencies_inputs = self.__class__.load_dependency_inputs(
                        self.runs, self.previous_inputs.copy(), inputs.copy())

                    # Update dict
                    # update() removes keys, so it's done manually
                    for k1, v1 in user_dependencies_inputs.items():
                        for k2, v2 in v1.items():
                            inputs[k1][k2] = v2

                # From dict to list
                inputs_list = self._dict_to_list(inputs, input=True)
                self.previous_inputs = inputs

                return inputs_list

        # Update internal alert state to divs
        @app.callback(
            Output(self.get_internal_id("alert"), 'children'),
            Output(self.get_internal_id("alert"), 'color'),
            Output(self.get_internal_id("alert"), 'is_open'),
            Input(self.get_internal_id("alert-interval"), 'n_intervals'))
        def update_alert(_):
            if self.alert_update_required:
                self.alert_update_required = False
                return self.alert_text, self.alert_color, True
            else:
                raise PreventUpdate()

    def update_alert(self, text, color="success"):
        self.alert_text = text
        self.alert_color = color
        self.alert_update_required = True

    def _inputs_changed(self, inputs, last_inputs):
        # Check if last_inputs are the same as the given inputs.
        inputs_changed = False
        filters_changed = False

        # If only filters changed, then we don't need to
        # calculate the results again.
        if last_inputs is not None:
            for (id, attribute, filter) in self.inputs:
                if inputs[id][attribute] != last_inputs[id][attribute]:
                    if not filter:
                        inputs_changed = True
                    else:
                        filters_changed = True

        return inputs_changed, filters_changed

    def _process_raw_outputs(self, inputs, raw_outputs):

        logger.debug("Process raw outputs.")
        # Use raw outputs to update our layout
        mpl_active = c.get("matplotlib-mode")
        if mpl_active:
            outputs = self.__class__.load_mpl_outputs(
                inputs, raw_outputs, self.groups)
        else:
            outputs = self.__class__.load_outputs(
                inputs, raw_outputs, self.groups)

        # Map outputs here because it may be that the outputs are
        # differently sorted than the values were registered.
        if isinstance(outputs, dict):
            outputs = self._dict_to_list(outputs, input=False)
        else:
            if not isinstance(outputs, list):
                outputs = [outputs]

        # We have to add no_updates here for the mode we don't want
        count_outputs = 0
        count_mpl_outputs = 0
        for (_, _, mpl_mode) in self.outputs:
            if mpl_mode:
                count_mpl_outputs += 1
            else:
                count_outputs += 1

        if mpl_active:
            outputs = [no_update for _ in range(count_outputs)] + outputs
        else:
            outputs = outputs + [no_update for _ in range(count_mpl_outputs)]

        return outputs

    def _list_to_dict(self, values: list, input=True):
        """
        Maps the given values to a dict, regarding the sorting from
        either self.inputs or self.outputs.

        Returns:
            dict
        """

        if input:
            order = self.inputs
        else:
            order = self.outputs

        mapping = {}
        for value, (id, attribute, *_) in zip(values, order):
            if id not in mapping:
                mapping[id] = {}

            mapping[id][attribute] = value

        return mapping

    def _dict_to_list(self, d, input=False):
        """
        Maps the given dict to a list, regarding the sorting from either
        self.inputs or self.outputs.

        Returns:
            list.
        """

        if input:
            order = self.inputs
        else:
            order = self.outputs

        result = []
        for (id, attribute, instance) in order:

            if not input:
                # Instance is mlp_mode in case of outputs
                # Simply ignore other outputs.
                if instance != c.get("matplotlib-mode"):
                    continue
            try:
                value = d[id][attribute]
                result += [value]
            except:
                result += [None]

        return result

    def _dict_as_key(self, d, remove_filters=False):
        """
        Converts a dictionary to a key. Only ids from self.inputs are considered.

        Parameters:
            d (dict): Dictionary to get the key from.
            remove_filters (bool): Option wheather the filters should be included or not.

        Returns:
            key (str): Key as string from the given dictionary.
        """

        if not isinstance(d, dict):
            return None

        new_d = copy.deepcopy(d)
        if remove_filters:
            for (id, _, filter) in self.inputs:
                if filter:
                    if id in new_d:
                        del new_d[id]

        return str(new_d)

    def __call__(self, render_button=False):
        """
        We overwrite the get_layout method here as we use a different
        interface compared to layout.
        """

        self.previous_inputs = {}

        self.runs = handler.get_runs()
        groups = handler.get_groups()
        if len(groups) == 0:
            # Create groups with run_name: run_name
            for run_name in handler.get_run_names():
                groups[run_name] = [run_name]

        self.groups = groups

        components = [html.H1(self.name())]

        if self.description() != '':
            components += [html.P(self.description())]

        # Register alerts
        components += [
            dcc.Interval(
                id=self.get_internal_id("alert-interval"),
                interval=1*500,
                n_intervals=5
            ),
            dbc.Alert(
                id=self.get_internal_id("alert"),
                is_open=False,
                dismissable=True,
                fade=True),
        ]

        status = self.check_requirements(self.runs)
        if isinstance(status, str):
            self.update_alert(status, color="danger")
            return components
        elif isinstance(status, bool):
            if not status:
                return components

        input_layout = self.get_input_layout(self.register_input)
        input_control_layout = html.Div(
            style={} if render_button else {"display": "none"},
            className="mt-3 clearfix",
            children=[
                dbc.Button(
                    children=self.button_caption(),
                    id=self.get_internal_id("update-button"),

                ),
                html.Span(
                    html.Em(id=self.get_internal_id("processing-info")),
                    className="ml-3 align-baseline",
                )
            ]
        )

        # We always have to render it because of the button.
        # Button tells us if the page was just loaded.
        components += [html.Div(
            id=f'{self.id()}-input',
            className="shadow-sm p-3 mb-3 bg-white rounded-lg",
            children=input_layout + [input_control_layout],
            style={} if render_button or input_layout else {"display": "none"}
        )]

        def register(a, b): return self.register_input(a, b, filter=True)
        filter_layout = self.__class__.get_filter_layout(register)
        if len(filter_layout) > 0:
            components += [html.Div(
                id=f'{self.id()}-filter',
                className="shadow-sm p-3 mb-3 bg-white rounded-lg",
                children=filter_layout
            )]

        output_layout = self.__class__.get_output_layout(self.register_output)
        if output_layout:
            components += [html.Div(
                id=f'{self.id()}-output',
                className="shadow-sm p-3 bg-white rounded-lg loading-container",
                children=output_layout,
                style={} if not c.get(
                    "matplotlib-mode") else {"display": "none"}
            )]

        def register(a, b): return self.register_input(a, b, mpl=True)
        output_layout = self.__class__.get_mpl_output_layout(register)
        if output_layout:
            components += [html.Div(
                id=f'{self.id()}-mpl-output',
                className="shadow-sm p-3 bg-white rounded-lg loading-container",
                children=output_layout,
                style={} if c.get(
                    "matplotlib-mode") else {"display": "none"}
            )]

        return components

    @staticmethod
    def load_inputs(runs):
        return {}

    @staticmethod
    def load_dependency_inputs(runs, previous_inputs, inputs):
        return inputs

    @staticmethod
    def get_input_layout(register):
        return []

    @staticmethod
    def get_filter_layout(register):
        return []

    @staticmethod
    def get_output_layout(register):
        return []

    @staticmethod
    def get_mpl_output_layout(register):
        return []

    @staticmethod
    def load_outputs(inputs, outputs, groups):
        return {}

    @staticmethod
    def load_mpl_outputs(inputs, outputs, groups):
        return {}

    @staticmethod
    @abstractmethod
    def process(run, inputs):
        pass
