from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple
import os
import json
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

from deep_cave.server import app
from deep_cave.data_manager import dm
from deep_cave.util.logs import get_logger
from deep_cave.layouts.layout import Layout


logger = get_logger(__name__)


class Plugin(Layout):
    @staticmethod
    @abstractmethod
    def name():
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def id():
        raise NotImplementedError()

    @staticmethod
    def description() -> str:
        return ''

    def _define_variables(self):
        self.inputs = {}
        self.outputs = {}

    def _register_callbacks(self):
        # We have to call the output layout one time to register
        # the values
        self._get_input_layout()
        self._get_output_layout()

        outputs = []
        for id, (attribute, _) in self.outputs.items():
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), 'n_clicks')]
        for id, attributes in self.inputs.items():
            for attribute in attributes:
                inputs.append(State(self.get_internal_input_id(id), attribute))

            # We also have to add the load attributes
            # as output
            #if isinstance(load_attrs, str):
            #    load_attrs = [load_attrs]
            
            #for load_attr in load_attrs:
            #    outputs.append(Output(self.get_internal_input_id(id), load_attr))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_output_update(n_clicks, *user_specified_inputs_list):
            # Make sure we have the run selected
            from deep_cave.run_manager import rm
            run = list(rm.get_runs().values())[0]

            # That's basically when the page reloads
            #if n_clicks is None:
            # We load the input first

            if n_clicks is None:
                outputs = dm.get(["plugins", self.id(), "outputs"])
                if outputs is None:
                    raise PreventUpdate()
            else:

                # Map the list `user_specified_inputs`
                # to a dict.
                user_specified_inputs = {}
                index = 0
                for id, attributes in self.inputs.items():
                    if id not in user_specified_inputs:
                        user_specified_inputs[id] = {}

                    for attribute in attributes:
                        if attribute not in user_specified_inputs[id]:
                            user_specified_inputs[id][attribute] = \
                                user_specified_inputs_list[index]

                        index += 1

                outputs = self.process(**user_specified_inputs)

                # Now we have to check if any funcs were
                # registered
                for id, (attribute, func) in self.outputs.items():
                    if func is not None:
                        outputs[id] = func(outputs[id])

                # Save inputs+outputs in cache
                dm.set(["plugins", self.id(), "user_specified_inputs"], user_specified_inputs)
                dm.set(["plugins", self.id(), "outputs"], outputs)

            return list(outputs.values())

        # Handles the initial and the cashed input values
        outputs = []
        inputs = [Input(self.get_internal_id("update-button"), 'n_clicks')]
        for id, attributes in self.inputs.items():
            for attribute in attributes:
                outputs.append(Output(self.get_internal_input_id(id), attribute))

        if len(outputs) > 0:
            @app.callback(outputs, inputs)
            def plugin_input_update(n_clicks):
                # Only load what we've got
                if n_clicks is None:

                    initial_inputs = dm.get(["plugins", self.id(), "initial_inputs"])
                    user_specified_inputs = dm.get(["plugins", self.id(), "user_specified_inputs"])

                    if initial_inputs is None or user_specified_inputs is None:
                        from deep_cave.run_manager import rm
                        run = list(rm.get_runs().values())[0]
                        initial_inputs = self._load_input(run)
                        dm.set(["plugins", self.id(), "initial_inputs"], initial_inputs)
                    else:
                        # We got the values which are used in the output
                        # and we got the values which are filled when the
                        # plugin is called.
                        # However, since the user specified the
                        # initial values, we need to overwrite them.
                        initial_inputs.update(user_specified_inputs)

                    # From dict to list
                    initial_inputs_list = []
                    for v1 in initial_inputs.values():
                        for v2 in v1.values():
                            initial_inputs_list.append(v2)

                    return initial_inputs_list
                
                raise PreventUpdate()

    def register_input(self, id, attributes=["value"]):
        if isinstance(attributes, str):
            attributes = [attributes]

        for attribute in attributes:
            if id not in self.inputs:
                self.inputs[id] = []

            if attribute not in self.inputs[id]:
                self.inputs[id].append(attribute)

        return self.get_internal_input_id(id)
    
    def register_output(self, id, attribute="value", func=None):
        assert isinstance(attribute, str)

        """
        If a func is used then our attr is always children.
        That's because the func should always return
        a list of dash html components.
        """
        if func is not None:
            attribute = "children"

        if id not in self.outputs:
            self.outputs[id] = (attribute, func)

        return self.get_internal_output_id(id)

    def get_internal_id(self, id):
        return self.id() + "-" + id

    def get_internal_input_id(self, id):
        return self.id() + "-" + id + "-input"

    def get_internal_output_id(self, id):
        return self.id() + "-" + id + "-output"        

    def get_layout(self):
        """
        We overwrite the get_layout method here as we use a different
        interface compared to layout.
        """

        return [
            html.H1(self.name()),

            # Input
            html.Div(id=f'{self.id()}-input', className="shadow p-3 mb-5 bg-white rounded", children=self.get_input_layout()),

            # Output
            html.Div(id=f'{self.id()}-output', className="shadow p-3 mb-5 bg-white rounded", children=self.get_output_layout())
        ]

    def process(self, **kwargs):
        """
        Calls the plugin for every selected run.
        """

        # TODO: Prepare for multiple runs.

        from deep_cave.run_manager import rm
        for name, run in rm.get_runs().items():
            return self._load_output(**kwargs)

    @abstractmethod
    def _load_output(self, *args, **kwargs):
        pass

    def get_input_layout(self):
        input_layout = self._get_input_layout()

        # Add process button here
        input_layout += [
            dbc.Button(children="hi", id=self.get_internal_id("update-button"))
        ]

        return input_layout

    def get_output_layout(self):
        return self._get_output_layout()

    @abstractmethod
    def _get_input_layout(self):
        pass
    
    @abstractmethod
    def _get_output_layout(self):
        pass
