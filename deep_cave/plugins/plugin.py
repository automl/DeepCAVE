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

from deep_cave.server import app
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

    def register_input(self, id, attr):
        if id not in self.inputs:
            self.inputs[id] = attr

        return self.get_internal_input_id(id)
    
    def register_output(self, id, attr):
        if id not in self.outputs:
            self.outputs[id] = attr
        return self.get_internal_output_id(id)

    def get_internal_id(self, id):
        return self.id() + "-" + id

    def get_internal_input_id(self, id):
        return self.id() + "-" + id + "-input"

    def get_internal_output_id(self, id):
        return self.id() + "-" + id + "-output"

    def _register_callbacks(self):
        # We have to call the output layout one time to register
        # the values
        self._get_input_layout()
        self._get_output_layout()

        inputs = [Input(self.get_internal_id("update-button"), 'n_clicks')]
        for id, attr in self.inputs.items():
            inputs.append(State(self.get_internal_input_id(id), attr))

        outputs = []
        for id, attr in self.outputs.items():
            outputs.append(Output(self.get_internal_output_id(id), attr))

        # Register updates from inputs
        @app.callback(outputs, inputs, prevent_initial_call=True)
        def plugin_update(n_clicks, *args):
            return self.process(*args)

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

    def process(self, *args):
        """
        Calls the plugin for every selected run.
        """

        # TODO: Prepare for multiple runs.

        from deep_cave.run_manager import rm
        for name, run in rm.get_runs().items():
            return self._process(*args)

    @abstractmethod
    def _process(self, *args, **kwargs):
        pass

    def get_input_layout(self):
        input_layout = self._get_input_layout()

        # Add process button here
        input_layout += [
            dbc.Button(children="hi", id=self.get_internal_id("update-button"))
        ]

        return input_layout

    @abstractmethod
    def _get_input_layout(self):
        pass
    
    def get_output_layout(self):
        return self._get_output_layout()

    @abstractmethod
    def _get_output_layout(self):
        pass
