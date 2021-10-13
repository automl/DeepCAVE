import os

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from deep_cave import app, meta_cache, cache
from deep_cave.config import CONFIG
from deep_cave.runs.handler import handler
from deep_cave.layouts.layout import Layout
from deep_cave.utils.hash import string_to_hash


class GeneralLayout(Layout):
    def register_callbacks(self):
        outputs = [
            Output('general-working-directory-input', 'value'),
            Output('general-converter-label', 'children'),
            Output('general-runs-radiolist', 'options'),
            Output('general-runs-radiolist', 'value'),
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
                working_dir = meta_cache.get("working_dir")
                converter_name = meta_cache.get("converter_name")
                run_id = meta_cache.get("run_id")

                if converter_name is None:
                    converter_name = handler.find_compatible_converter(
                        working_dir)

                # Check here if working_dir is compatible
                meta_cache.set("converter_name", value=converter_name)

                return \
                    working_dir, \
                    self.get_converter_text(converter_name), \
                    self.get_run_options(), \
                    run_id

            # Check if working dir exists
            if working_dir is None or not os.path.isdir(working_dir):
                PreventUpdate()

            # Find converter name
            converter_name = handler.find_compatible_converter(working_dir)
            if converter_name is None:
                PreventUpdate()

            meta_cache.set("working_dir", value=working_dir)
            meta_cache.set("converter_name", value=converter_name)
            meta_cache.set("run_id", value=None)

            return \
                working_dir, \
                self.get_converter_text(converter_name), \
                self.get_run_options(), \
                meta_cache.get("run_id")

        input = Input('general-runs-radiolist', 'value')
        output = Output('general-runs-output', 'value')

        # Save the run ids internally
        # We have to inform the other plugins here as well
        @app.callback(output, input)
        def general_register_runs(run_id):
            if meta_cache.get("run_id") != run_id:
                meta_cache.set("run_id", value=run_id)

                return run_id

            raise PreventUpdate()

    @staticmethod
    def get_converter_options():
        return [{"label": adapter, "value": adapter} for adapter in handler.get_available_converters().keys()]

    @staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in handler.get_run_ids()]

    @staticmethod
    def get_converter_text(converter_name):
        converter_text = ""
        if converter_name is not None:
            converter_text = [
                html.Span("Found compatible runs from "),
                html.I(converter_name),
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
            dbc.RadioItems(id="general-runs-radiolist"),


            html.Hr(),

            html.H2('Additional'),
            dbc.Button("Clear Cache",
                       id="general-clear-cache-button", color="primary"),
        ]


layout = GeneralLayout()
