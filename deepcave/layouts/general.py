from typing import Type

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, rc, run_handler
from deepcave.layouts import Layout
from deepcave.runs import NotMergeableError, NotValidRunError
from deepcave.runs.run import Run


class GeneralLayout(Layout):
    def register_callbacks(self) -> None:
        self._callback_working_directory_changed()
        self._callback_run_selection_changed()
        self._callback_group_selection_changed()
        self._callback_set_groups()
        self._callback_clear_cache()

    def _callback_working_directory_changed(self) -> None:
        outputs = [
            # Working directory input
            Output("general-working-directory-input", "value"),
            # Converter text
            Output("general-converter-label", "children"),
            Output("general-runs-checklist", "options"),
            Output("general-runs-checklist", "value"),
        ]

        inputs = [
            Input("on-page-load", "href"),
            Input("general-working-directory-input", "value"),
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def callback(_, working_dir: str):
            # `working_dir` is only none on page load
            if working_dir is None:
                handler_working_dir = run_handler.get_working_dir()
                run_names = run_handler.get_run_names()
                converter = run_handler.available_run_classes

                return (
                    str(handler_working_dir),
                    self.get_converter_text(converter),
                    self.get_run_options(),
                    run_names,
                )

            empty_run_names = []
            run_handler.update_working_directory(Path(working_dir))

            return (
                working_dir,
                self.get_converter_text(run_handler.available_run_classes),
                self.get_run_options(),
                empty_run_names,
            )

    def _callback_run_selection_changed(self) -> None:
        output = Output("general-run-names", "data")
        input = Input("general-runs-checklist", "value")

        # Save the run ids internally
        @app.callback(output, input)
        def callback(run_ids: list[str]):
            old_run_names = run_handler.get_run_names()
            self.logger.debug(f"Old runs: {old_run_names}, Runs: {run_ids}")

            # Reset groups here.
            # Alternatively: Remove all runs which are not selected anymore.
            if run_ids != old_run_names:
                run_handler.update_groups({})

                # Also really important:
                # Reset last inputs
                c.set("last_inputs", value={})

            try:
                run_handler.update_runs(run_ids)
            except NotValidRunError:
                return old_run_names

            return run_ids

    def _callback_group_selection_changed(self) -> None:
        outputs = [
            Output("general-group-container", "children"),
            Output("general-add-group", "n_clicks"),
        ]
        inputs = [
            Input("general-add-group", "n_clicks"),
            Input("general-run-names", "data"),
            Input("general-group-trigger", "data"),
            State("general-group-container", "children"),
        ]

        # Let's take care of the groups here
        @app.callback(outputs, inputs)
        def callback(n_clicks: int, run_names, update, children):
            def get_layout(index, options, input_value="", dropdown_value=None):
                if dropdown_value is None:
                    dropdown_value = []
                return html.Div(
                    [
                        dbc.Input(
                            id={"type": "group-name", "index": index},
                            placeholder="Name",
                            type="text",
                            value=input_value,
                            style={"margin-bottom": "-1px"},
                        ),
                        dcc.Dropdown(
                            id={"type": "group-dropdown", "index": index},
                            options=[
                                {"label": name, "value": name} for name in options
                            ],
                            value=dropdown_value,
                            multi=True,
                        ),
                    ],
                    className="mb-2",
                )

            groups = run_handler.get_groups()
            index = 0

            # Load from cache if page is loaded
            children = []
            for group_name, grouped_run in groups.items():
                if group_name is None:
                    continue

                children.append(
                    get_layout(index, run_names, group_name, grouped_run.run_names)
                )

                index += 1

            if n_clicks is not None and len(run_names) > 0:
                children.append(get_layout(index, run_names))

            return children, None

    def _callback_set_groups(self) -> None:
        outputs = Output("general-group-trigger", "data")
        inputs = [
            Input({"type": "group-name", "index": ALL}, "value"),
            Input({"type": "group-dropdown", "index": ALL}, "value"),
        ]

        @app.callback(outputs, inputs)
        def callback(group_names, all_run_names):
            # Abort on page load
            if self._refresh_groups:
                self._refresh_groups = False
                raise PreventUpdate()

            groups = {}
            for group_name, run_names in zip(group_names, all_run_names):
                if group_name is None or group_name == "":
                    continue

                if run_names is None or len(run_names) == 0:
                    continue

                groups[group_name] = run_names

            try:
                # Now save it
                run_handler.update_groups(groups)
            except NotMergeableError:
                return True

            self.logger.debug(f"Groups: {groups}")

            raise PreventUpdate()

    def _callback_clear_cache(self) -> None:
        output = Output("general-clear-cache-button", "n_clicks")
        input = Input("general-clear-cache-button", "n_clicks")

        @app.callback(output, input)
        def callback(n_clicks):
            if n_clicks is not None:
                rc.clear_all_caches()

            return None

    @staticmethod
    def get_run_options() -> list[dict[str, str]]:
        runs = [
            {"label": run_name, "value": run_name}
            for run_name in run_handler.get_available_run_names()
        ]
        return runs

    @staticmethod
    def get_converter_text(converters: list[Type[Run]]) -> html.Div:
        converter_text = []
        for converter in converters:
            converter_text += [converter.prefix]

        return html.Div(
            html.Span(f"Available run converters: {', '.join(converter_text)}"),
            className="mt-2",
        )

    def __call__(self) -> list[Component]:
        self._refresh_groups = True

        return [
            html.H1("General"),
            dbc.Label("Working Directory"),
            # html.Div("Working Directory"),
            # dbc.FormText("Absolute path to your runs."),
            dbc.Input(
                id="general-working-directory-input", placeholder="", type="text"
            ),
            dbc.FormText(id="general-converter-label"),
            html.Hr(),
            html.H2("Runs"),
            dcc.Store(id="general-run-names"),
            dbc.Checklist(id="general-runs-checklist"),
            html.Hr(),
            html.H2("Groups"),
            html.Div(id="general-group-container", children=[]),
            dbc.Button("Add Group", id="general-add-group"),
            dcc.Store(id="general-group-trigger"),
            html.Hr(),
            html.H2("Caches"),
            dbc.Button(
                "Clear Plugin Caches", id="general-clear-cache-button", color="primary"
            ),
        ]
