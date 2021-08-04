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

from deep_cave import app
from deep_cave.cache import cache
from deep_cave.util.logs import get_logger
from deep_cave.layouts.layout import Layout
from deep_cave.runs import get_selected_run


logger = get_logger(__name__)


class Plugin(Layout):
    def __init__(self):
        self.inputs = {}
        self.interactive_inputs = {}
        self.outputs = {}

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
    def update_on_changes():
        return False

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

    def register_callbacks(self):
        # We have to call the output layout one time to register
        # the values
        self.get_input_layout()
        self.get_output_layout()

        outputs = []
        for id, (attribute, _) in self.outputs.items():
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), 'n_clicks')]
        for id, attributes in self.inputs.items():
            for attribute in attributes:
                if self.update_on_changes():
                    inputs.append(Input(self.get_internal_input_id(id), attribute))
                else:
                    inputs.append(State(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_output_update(n_clicks, *user_specified_inputs_list):
            if n_clicks is None and not self.update_on_changes():
                outputs = cache.get(["plugins", self.id(), "outputs"])
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

                outputs = self.load_output(get_selected_run(), **user_specified_inputs)

                # Now we have to check if any funcs were
                # registered
                for id, (attribute, func) in self.outputs.items():
                    if func is not None:
                        outputs[id] = func(outputs[id])

                # Save inputs+outputs in cache
                cache.set(["plugins", self.id(), "outputs"], outputs)

            return list(outputs.values())

        # Handles the initial and the cashed input values
        outputs = []
        inputs = [] #[Input(self.get_internal_id("update-button"), 'n_clicks')]

        # Define also inputs if they are declared as interactive
        for id, attributes in self.inputs.items():
            for attribute in attributes:
                inputs.append(Input(self.get_internal_input_id(id), attribute))

        for id, attributes in self.inputs.items():
            for attribute in attributes:
                outputs.append(Output(self.get_internal_input_id(id), attribute))

        if len(outputs) > 0:
            @app.callback(outputs, inputs)
            def plugin_input_update(*user_specified_inputs_list):
                init = True
                for input in user_specified_inputs_list:
                    if input is not None:
                        init = False
                        break

                # Only load what we've got
                if init:
                    user_specified_inputs = cache.get(["plugins", self.id(), "user_specified_inputs"])

                    if user_specified_inputs is None:
                        user_specified_inputs = self.load_input(get_selected_run())
                        cache.set(["plugins", self.id(), "user_specified_inputs"], user_specified_inputs)
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

                    # How to update only parameters which have a dependency?
                    user_dependencies_inputs = self.load_dependency_input(get_selected_run(), **user_specified_inputs)

                    # Update dict
                    # update() removes keys, so it's done manually
                    for k1, v1 in user_dependencies_inputs.items():
                        for k2, v2 in v1.items():
                            user_specified_inputs[k1][k2] = v2
                    
                    cache.set(["plugins", self.id(), "user_specified_inputs"], user_specified_inputs)

                # From dict to list
                user_specified_inputs_list = []
                for v1 in user_specified_inputs.values():
                    for v2 in v1.values():
                        user_specified_inputs_list.append(v2)

                return user_specified_inputs_list        

    def __call__(self):
        """
        We overwrite the get_layout method here as we use a different
        interface compared to layout.
        """
        components = [html.H1(self.name())]

        if self.description() != '':
            components += [html.P(self.description())]

        input_button = dbc.Button(
            children=self.button_caption(),
            className="mt-3",
            id=self.get_internal_id("update-button"),
            style={"display": "none"} if self.update_on_changes() else {}
        )

        input_layout = self.get_input_layout()
        if input_layout:
            components += [html.Div(
                id=f'{self.id()}-input',
                className="shadow-sm p-3 mb-3 bg-white rounded-lg",
                children=self.get_input_layout() + [input_button]
            )]
        
        output_layout = self.get_output_layout()
        if output_layout:
            components += [html.Div(
                id=f'{self.id()}-output',
                className="shadow-sm p-3 bg-white rounded-lg",
                children=self.get_output_layout()
            )]

        return components

    @abstractmethod
    def load_input(run):
        pass

    @abstractmethod
    def load_output(self, **kwargs):
        pass
    
    def load_dependency_input(self, run, **inputs):
        return inputs

    def get_input_layout(self):
        return []
    
    def get_output_layout(self):
        return []
