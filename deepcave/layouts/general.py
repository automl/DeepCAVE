from typing import Dict, List, Optional, Type

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, notification, queue, rc, run_handler
from deepcave.layouts import Layout
from deepcave.runs import NotMergeableError
from deepcave.runs.run import Run
from deepcave.utils.layout import help_button
from deepcave.utils.util import short_string


class GeneralLayout(Layout):
    def register_callbacks(self) -> None:
        self._callback_working_directory_changed()
        self._callback_display_available_runs()
        self._callback_display_selected_runs()
        self._callback_manage_run()
        self._callback_display_groups()
        self._callback_manage_groups()
        self._callback_clear_cache()

    def _callback_working_directory_changed(self) -> None:
        inputs = [
            Input("on-page-load", "href"),
            Input("general-working-directory-input", "value"),
            # Those inputs are for changing working directory quickly
            # (if someone is pressing on a directory)
            # Only works with "ALL"
            Input({"type": "general-dynamic-change-directory", "index": ALL}, "n_clicks"),
            State({"type": "general-dynamic-available-run-path", "index": ALL}, "data"),
        ]

        outputs = [
            # Working directory input
            Output("general-working-directory-input", "value"),
            # Converter text
            Output("general-converter-label", "children"),
            Output("general-available-runs", "data"),
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)  # type: ignore
        def callback(  # type: ignore
            _,
            working_dir: str,
            dynamic_n_clicks: List[Optional[int]],
            dynamic_working_dirs: List[str],
        ):
            # `working_dir` is only none on page load
            if working_dir is None:
                # Handler working directory
                working_dir = run_handler.get_working_directory()
                converter = run_handler.available_run_classes
            else:
                for dir, n_clicks in zip(dynamic_working_dirs, dynamic_n_clicks):
                    if n_clicks is not None:
                        working_dir = dir
                        break

                run_handler.set_working_directory(working_dir)
                converter = run_handler.available_run_classes

            return (
                str(working_dir),
                self.get_converter_text(converter),
                run_handler.get_available_run_paths(),
            )

    def _callback_display_available_runs(self) -> None:
        output = Output("general-available-runs-container", "children")
        input = Input("general-available-runs", "data")

        @app.callback(output, input)  # type: ignore
        def callback(run_paths: List[str]):  # type: ignore
            children = []

            # Add text to go to parent directory
            new_element = html.Div(
                [
                    dbc.Button(
                        "+", id={"type": "general-dynamic-add-run", "index": -1}, disabled=True
                    ),
                    dbc.Button(
                        "..",
                        id={"type": "general-dynamic-change-directory", "index": -1},
                        color="link",
                    ),
                    dcc.Store(
                        id={"type": "general-dynamic-available-run-path", "index": -1},
                        data=str(Path(run_handler.get_working_directory()).parent.absolute()),
                    ),
                ],
                className="mb-1",
            )
            children.append(new_element)

            for i, run_path in enumerate(run_paths):
                run_name = run_handler.get_run_name(run_path)
                new_element = html.Div(
                    [
                        dbc.Button("+", id={"type": "general-dynamic-add-run", "index": i}),
                        dbc.Button(
                            run_name,
                            id={"type": "general-dynamic-change-directory", "index": i},
                            color="link",
                        ),
                        dcc.Store(
                            id={"type": "general-dynamic-available-run-path", "index": i},
                            data=run_path,
                        ),
                    ],
                    className="mb-1",
                )
                children.append(new_element)

            if len(children) == 0:
                return html.Div("No runs found.")

            return children

    def _callback_display_selected_runs(self) -> None:
        output = Output("general-selected-runs-container", "children")
        input = Input("general-selected-runs", "data")

        @app.callback(output, input)  # type: ignore
        def callback(run_paths: List[str]):  # type: ignore
            children = []

            for i, run_path in enumerate(run_paths):
                run_name = run_handler.get_run_name(run_path)

                shortened_run_path = short_string(run_path, 30, mode="prefix")

                new_element = html.Div(
                    [
                        dbc.Button(
                            "-",
                            id={"type": "general-dynamic-remove-run", "index": i},
                            color="danger",
                        ),
                        html.Span(
                            [
                                run_name,
                                html.I(
                                    f"({shortened_run_path})",
                                    style={"font-size": "10px", "margin-left": "6px"},
                                ),
                            ],
                            # color="light",
                            style={"margin-left": "12px"},
                        ),
                        dcc.Store(
                            id={"type": "general-dynamic-selected-run-path", "index": i},
                            data=run_path,
                        ),
                    ],
                    className="mb-1",
                )
                children.append(new_element)

            if len(children) == 0:
                return html.Div("No runs selected.")

            return children

    def _callback_manage_run(self) -> None:
        outputs = [
            Output({"type": "general-dynamic-add-run", "index": ALL}, "n_clicks"),
            Output({"type": "general-dynamic-remove-run", "index": ALL}, "n_clicks"),
            Output("general-selected-runs", "data"),
        ]

        inputs = [
            Input({"type": "general-dynamic-add-run", "index": ALL}, "n_clicks"),
            Input({"type": "general-dynamic-remove-run", "index": ALL}, "n_clicks"),
        ]

        states = [
            State({"type": "general-dynamic-available-run-path", "index": ALL}, "data"),
            State({"type": "general-dynamic-selected-run-path", "index": ALL}, "data"),
        ]

        @app.callback(outputs, inputs, states)  # type: ignore
        def callback(  # type: ignore
            add_n_clicks, remove_n_clicks, available_run_paths, selected_run_paths
        ):
            # Add run path
            for n_click, run_path in zip(add_n_clicks, available_run_paths):
                if n_click is not None:
                    success = run_handler.add_run(run_path)
                    if not success:
                        notification.update("The run could not be added.")

            # Remove run path
            for n_click, run_path in zip(remove_n_clicks, selected_run_paths):
                if n_click is not None:
                    run_handler.remove_run(run_path)

            # Reset clicks
            add_n_clicks = [None for _ in add_n_clicks]
            remove_n_clicks = [None for _ in remove_n_clicks]

            # Remove last inputs
            c.set("last_inputs", value={})

            return add_n_clicks, remove_n_clicks, run_handler.get_selected_run_paths()

    def _callback_display_groups(self) -> None:
        outputs = [
            Output("general-group-container", "children"),
            Output("general-add-group", "n_clicks"),
        ]
        inputs = [
            Input("general-add-group", "n_clicks"),
            Input("general-selected-runs-container", "children"),
            Input("general-group-trigger", "data"),
            State("general-group-container", "children"),
        ]

        # Let's take care of the groups here
        @app.callback(outputs, inputs)  # type: ignore
        def callback(n_clicks: int, _trigger1, _trigger2, children):  # type: ignore
            def get_layout(
                index: int,
                options: Dict[str, str],
                input_value: str = "",
                dropdown_value: Optional[List[str]] = None,
            ) -> Component:
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
                            options=[{"label": v, "value": k} for k, v in options.items()],
                            value=dropdown_value,
                            multi=True,
                        ),
                    ],
                    className="mb-2",
                )

            groups = run_handler.get_selected_groups()
            selected_run_paths = run_handler.get_selected_run_paths()
            selected_runs = {p: run_handler.get_run_name(p) for p in selected_run_paths}
            index = 0

            # Load from cache if page is loaded
            children = []
            for name, paths in groups.items():
                if name is None:
                    continue

                children.append(get_layout(index, selected_runs, name, paths))

                index += 1

            if n_clicks is not None and len(selected_runs) > 0:
                children.append(get_layout(index, selected_runs))

            return children, None

    def _callback_manage_groups(self) -> None:
        outputs = Output("general-group-trigger", "data")
        inputs = [
            Input({"type": "group-name", "index": ALL}, "value"),
            Input({"type": "group-dropdown", "index": ALL}, "value"),
            State("general-group-trigger", "data"),
        ]

        @app.callback(outputs, inputs)  # type: ignore
        def callback(group_names, all_run_paths, i):  # type: ignore
            # Abort on page load
            if self._refresh_groups:
                self._refresh_groups = False
                raise PreventUpdate()

            groups = {}
            for group_name, run_paths in zip(group_names, all_run_paths):
                if group_name is None or group_name == "":
                    continue

                if run_paths is None or len(run_paths) == 0:
                    continue

                valid_run_paths = []
                for run_path in run_paths:
                    if run_path in run_handler.get_selected_run_paths():
                        valid_run_paths.append(run_path)

                if len(valid_run_paths) == 0:
                    continue

                groups[group_name] = valid_run_paths

            try:
                # Now save it
                run_handler.update_groups(groups)
            except NotMergeableError:
                notification.update("The selected runs are not mergeable.")

                # This will automatically trigger the group display s.t. the selection is redo.
                return i + 1

            self.logger.debug(f"Groups: {groups}")

            raise PreventUpdate()

    def _callback_clear_cache(self) -> None:
        output = Output("general-clear-cache-button", "n_clicks")
        input = Input("general-clear-cache-button", "n_clicks")

        @app.callback(output, input)  # type: ignore
        def callback(n_clicks):  # type: ignore
            if n_clicks is not None:
                rc.clear()

                # Also remove last inputs
                c.set("last_inputs", value={})

                # Also: We have to stop all running jobs
                queue.delete_jobs()

                # We have to call `update_runs` now to create the run caches again
                run_handler.update_runs()
                notification.update("Plugin caches successfully cleared.", "success")

            return None

    @staticmethod
    def get_converter_text(converters: List[Type[Run]]) -> html.Div:
        converter_text = []
        for converter in converters:
            converter_text += [converter.prefix]

        return html.Div(
            html.Span(f"Available run converters: {', '.join(converter_text)}"),
            className="mt-2",
        )

    def __call__(self) -> List[Component]:
        self._refresh_groups = True

        return [
            html.H1("General"),
            # Working dir
            dbc.Label("Working Directory"),
            help_button("Relative and absolute paths are supported."),
            dbc.Input(id="general-working-directory-input", placeholder="", type="text"),
            dbc.FormText(id="general-converter-label"),
            dcc.Store(id="general-available-runs", data={}),
            html.Div(id="general-available-runs-container", className="mt-2"),
            html.Hr(),
            # Selected Runs
            html.H2("Selected Runs"),
            dcc.Store(id="general-selected-runs", data={}),
            html.Div(id="general-selected-runs-container", className="mt-2"),
            html.Hr(),
            # Groups
            html.H2("Groups"),
            html.Div(id="general-group-container", children=[]),
            dbc.Button("Add Group", id="general-add-group"),
            dcc.Store(id="general-group-trigger", data=0),
            html.Hr(),
            # Cache
            html.H2("Caches"),
            dbc.Button("Clear Plugin Caches", id="general-clear-cache-button", color="primary"),
        ]
