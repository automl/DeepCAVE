import os

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from deepcave import app, c, rc
from deepcave.config import CONFIG
from deepcave.runs.handler import handler
from deepcave.layouts.layout import Layout
from deepcave.utils.hash import string_to_hash


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
        def general_update(_, working_dir):
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
            handler.set_run_names(run_names)  # Reset run ids
            handler.set_groups({})  # Reset groups

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
        output = Output('general-run-names', 'value')

        # Save the run ids internally
        @app.callback(output, input)
        def general_register_runs(run_names):
            old_run_names = handler.get_run_names()

            # Reset groups here.
            # Alternatively: Remove all runs which are not selected anymore.
            if run_names != old_run_names:
                handler.set_groups({})

                # Also really important:
                # Reset last inputs
                c.set("last_inputs", value={})

            handler.set_run_names(run_names)
            return run_names

        outputs = [
            Output('general-group-container', 'children'),
            Output('general-add-group', 'n_clicks'),
        ]
        inputs = [
            Input('general-add-group', 'n_clicks'),
            Input('general-run-names', 'value'),
            State('general-group-container', 'children')
        ]

        # Let's take care of the groups here
        @app.callback(outputs, inputs)
        def general_display_groups(n_clicks, run_names, children):
            def get_layout(index, options, input_value="", dropdown_value=[]):
                return html.Div([
                    dbc.Input(
                        id={'type': 'group-name', 'index': index},
                        placeholder="Name",
                        type="text",
                        value=input_value,
                        style={"margin-bottom": "-1px"}),
                    dcc.Dropdown(
                        id={'type': 'group-dropdown', 'index': index},
                        options=[{'label': i, 'value': i} for i in options],
                        value=dropdown_value,
                        multi=True,
                    )
                ], className="mb-2")

            groups = handler.get_groups()
            index = 0

            # Load from cache if page is loaded
            children = []
            for group_name, selected_run_names in groups.items():
                if group_name is None:
                    continue

                children.append(
                    get_layout(index, run_names, group_name,
                               selected_run_names)
                )

                index += 1

            if n_clicks is not None and len(run_names) > 0:
                children.append(
                    get_layout(index, run_names)
                )

            return children, None

        outputs = Output('general-group-output', 'data')
        inputs = [
            Input({'type': 'group-name', 'index': ALL}, 'value'),
            Input({'type': 'group-dropdown', 'index': ALL}, 'value')
        ]

        @app.callback(outputs, inputs)
        def general_set_groups(group_names, all_run_names):
            # Abort on page load
            if self._refresh_groups:
                self._refresh_groups = False
                return

            groups = {}
            for group_name, run_names in zip(group_names, all_run_names):
                if group_name is None or group_name == "":
                    continue

                if run_names is None or len(run_names) == 0:
                    continue
                    # run_names = []

                groups[group_name] = run_names

            # Now save it
            handler.set_groups(groups)

            return

        output = Output('general-clear-cache-button', 'n_clicks')
        input = Input('general-clear-cache-button', 'n_clicks')

        @app.callback(output, input)
        def general_clear_cache(n_clicks):
            if n_clicks is not None:
                rc.clear_all()

            return None

    @ staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in handler.get_available_run_names()]

    @ staticmethod
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
        self._refresh_groups = True

        return [
            html.H1('General'),

            dbc.Label("Working Directory"),
            #html.Div("Working Directory"),
            #dbc.FormText("Absolute path to your runs."),
            dbc.Input(id="general-working-directory-input",
                      placeholder="", type="text"),

            dbc.FormText(id="general-converter-label"),

            html.Hr(),

            html.H2('Runs'),
            dbc.Input(id="general-run-names", style={"display": "none"}),
            dbc.Checklist(id="general-runs-checklist"),

            html.Hr(),

            html.H2('Groups'),
            html.Div(id="general-group-container", children=[]),
            dbc.Button("Add Group", id="general-add-group"),
            dcc.Store(id="general-group-output"),

            html.Hr(),

            html.H2('Caches'),
            dbc.Button("Clear Plugin Caches",
                       id="general-clear-cache-button", color="primary"),
        ]


layout = GeneralLayout()
