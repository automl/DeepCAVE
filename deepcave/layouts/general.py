from pathlib import Path
from typing import Type

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, rc
from deepcave.layouts import Layout
from deepcave.runs.handler import run_handler
from deepcave.runs.run import Run


class GeneralLayout(Layout):
    def register_callbacks(self):
        self._callback_working_directory_changed()
        self._callback_run_selection_changed()
        self._callback_group_selection_changed()
        self._callback_set_groups()
        self._callback_clear_cache()

    def _callback_working_directory_changed(self):
        outputs = [
            Output('general-working-directory-input', 'value'),  # Working directory input
            Output('general-converter-label', 'children'),  # Converter text
            Output('general-runs-checklist', 'options'),  # Runs options : ("labels": , "values":)
            Output('general-runs-checklist', 'value'),  # ???
        ]

        inputs = [
            Input('on-page-load', 'href'),
            Input('general-working-directory-input', 'value'),
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def callback(_, working_dir: str):
            # `working_dir` is only none on page load
            if working_dir is None:
                handler_working_dir = run_handler.get_working_dir()
                run_names = run_handler.get_run_names()
                converter = run_handler.available_run_classes

                return (str(handler_working_dir),
                        self.get_converter_text(converter),
                        self.get_run_options(),
                        run_names)

            # Check if working dir exists
            working_dir_path = Path(working_dir)
            if not working_dir_path.is_dir():
                raise PreventUpdate()

            empty_run_names = []
            run_handler.update_working_directory(working_dir_path)

            return (working_dir,
                    self.get_converter_text(run_handler.available_run_classes),
                    self.get_run_options(),
                    empty_run_names)

    def _callback_run_selection_changed(self):
        output = Output('general-run-names', 'value')
        input = Input('general-runs-checklist', 'value')

        # Save the run ids internally
        @app.callback(output, input)
        def callback(run_ids: list[str]):
            old_run_names = run_handler.get_run_names()
            print(f"Old run names: {old_run_names}, Run names: {run_ids}")

            # Reset groups here.
            # Alternatively: Remove all runs which are not selected anymore.
            if run_ids != old_run_names:
                run_handler.update_groups({})

                # Also really important:
                # Reset last inputs
                c.set("last_inputs", value={})

            run_handler.update_runs(run_ids)
            return run_ids

    def _callback_group_selection_changed(self):
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
        def callback(n_clicks:int, run_names, children):
            def get_layout(index, options, input_value="", dropdown_value=None):
                if dropdown_value is None:
                    dropdown_value = []
                return html.Div([
                    dbc.Input(
                        id={'type': 'group-name', 'index': index},
                        placeholder="Name",
                        type="text",
                        value=input_value,
                        style={"margin-bottom": "-1px"}),
                    dcc.Dropdown(
                        id={'type': 'group-dropdown', 'index': index},
                        options=[{'label': name, 'value': name} for name in options],
                        value=dropdown_value,
                        multi=True,
                    )
                ], className="mb-2")

            groups = run_handler.get_groups()
            index = 0

            # Load from cache if page is loaded
            children = []
            for group_name, grouped_run in groups.items():
                if group_name is None:
                    continue

                children.append(
                    get_layout(index, run_names, group_name,
                               grouped_run.run_names)
                )

                index += 1

            if n_clicks is not None and len(run_names) > 0:
                children.append(
                    get_layout(index, run_names)
                )

            return children, None

    def _callback_set_groups(self):
        outputs = Output('general-group-output', 'data')
        inputs = [
            Input({'type': 'group-name', 'index': ALL}, 'value'),
            Input({'type': 'group-dropdown', 'index': ALL}, 'value')
        ]

        @app.callback(outputs, inputs)
        def callback(group_names, all_run_names):
            # Abort on page load
            if self._refresh_groups:
                self._refresh_groups = False
                return

            groups = {}
            for group_name, run_names in zip(group_names, all_run_names):
                print(group_name, run_names)
                if group_name is None or group_name == "":
                    continue

                if run_names is None or len(run_names) == 0:
                    continue

                groups[group_name] = run_names

            # Now save it
            print("Groups:", groups)
            run_handler.update_groups(groups)

            return

    def _callback_clear_cache(self):
        output = Output('general-clear-cache-button', 'n_clicks')
        input = Input('general-clear-cache-button', 'n_clicks')

        @app.callback(output, input)
        def callback(n_clicks):
            if n_clicks is not None:
                rc.clear_all_caches()

            return None

    @staticmethod
    def get_run_options() -> list[dict[str, str]]:
        runs = [{"label": run_name, "value": run_name} for run_name in run_handler.get_available_run_names()]
        return runs

    @staticmethod
    def get_converter_text(converters: dict[Type[Run], int]) -> html.Div:
        converter_texts = []
        for converter in sorted(converters, key=converters.get, reverse=True):
            converter_texts += [
                html.Li([
                    html.I(converter.__name__),
                    html.Span(".")
                ])
            ]

        return html.Div([
            html.Span("Possible run converters:"),
            html.Ul(
                converter_texts
            )
        ], className="mt-2")

    def __call__(self) -> list[Component]:
        self._refresh_groups = True

        return [
            html.H1('General'),

            dbc.Label("Working Directory"),
            # html.Div("Working Directory"),
            # dbc.FormText("Absolute path to your runs."),
            dbc.Input(id="general-working-directory-input",
                      placeholder="",
                      type="text"),

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
