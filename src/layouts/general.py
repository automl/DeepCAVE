import os

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from src import app, c
from src.config import CONFIG
from src.runs.handler import handler
from src.layouts.layout import Layout
from src.utils.hash import string_to_hash


class GeneralLayout(Layout):
    def register_callbacks(self):
        outputs = [
            Output('general-working-directory-input', 'value'),
            Output('general-converter-label', 'children'),
            Output('general-runs-checklist', 'options'),
            Output('general-runs-checklist', 'value'),
        ]

        inputs = [
            Input('on-page-load', 'href'),
            Input('general-working-directory-input', 'value'),
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def general_update(load, working_dir):
            # `working_dir` is only none on page load
            if working_dir is None:
                working_dir = handler.get_working_dir()
                run_ids = handler.get_run_ids()
                converter = handler.get_converter()

                return \
                    working_dir, \
                    self.get_converter_text(converter), \
                    self.get_run_options(), \
                    list(run_ids.keys())

            # Check if working dir exists
            if working_dir is None or not os.path.isdir(working_dir):
                PreventUpdate()

            run_names = []
            handler.set_working_dir(working_dir)
            handler.set_run_ids(run_names)  # Reset run ids

            # Find converter name
            converter = handler.get_converter()
            if converter is None:
                PreventUpdate()

            return \
                working_dir, \
                self.get_converter_text(converter), \
                self.get_run_options(), \
                run_names

        input = Input('general-runs-checklist', 'value')
        output = Output('general-runs-output', 'value')

        # Save the run ids internally
        @app.callback(output, input)
        def general_register_runs(run_names):
            handler.set_run_names(run_names)
            return run_names

    @staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in handler.get_available_run_names()]

    @staticmethod
    def get_converter_text(converter):
        converter_text = ""
        if converter is not None:
            converter_text = [
                html.Span("Found compatible runs from "),
                html.I(converter.name()),
                html.Span(".")
            ]

        return converter_text

    def __call__(self):
        return [
            html.H1('General'),

            dbc.FormGroup([
                dbc.Label("Working Directory",
                          html_for="general-working-directory-input"),
                dbc.FormText("Absolute path to your runs."),
                dbc.Input(id="general-working-directory-input",
                          placeholder="", type="text"),
            ]),

            html.Div(id="general-converter-label"),

            html.Hr(),

            html.H2('Runs'),
            dbc.Input(id="general-runs-output", style={"display": "none"}),
            dbc.Checklist(id="general-runs-checklist"),

            html.Hr(),

            html.H2('Groups'),
            html.Div("In progress"),


            html.Hr(),

            html.H2('Additional'),
            dbc.Button("Clear Cache",
                       id="general-clear-cache-button", color="primary"),
        ]


layout = GeneralLayout()
