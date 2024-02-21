import os
import time

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from deepcave import app, c, queue
from deepcave.layouts import Layout


class HeaderLayout(Layout):
    def register_callbacks(self) -> None:
        super().register_callbacks()
        self._callback_update_matplotlib_mode()
        self._callback_delete_jobs()
        self._callback_terminate_deepcave()

    def _callback_update_matplotlib_mode(self) -> None:
        outputs = [
            Output("matplotlib-mode-toggle", "color"),
            Output("matplotlib-mode-badge", "children"),
            Output("matplotlib-mode-refresh", "href"),
        ]
        inputs = [
            Input("matplotlib-mode-toggle", "n_clicks"),
            Input("matplotlib-mode-refresh", "pathname"),
        ]

        @app.callback(outputs, inputs)
        def callback(n_clicks, pathname):
            update = None
            mode = c.get("matplotlib-mode")
            if mode is None:
                mode = False

            if n_clicks is not None:
                update = pathname
                mode = not mode
                c.set("matplotlib-mode", value=mode)

            if mode:
                return "primary", "on", update
            else:
                return "secondary", "off", update

    def _callback_delete_jobs(self) -> None:
        inputs = [Input("exit-deepcave", "n_clicks")]
        outputs = [
            Output("exit-deepcave", "color"),
            Output("exit-deepcave", "children"),
            Output("exit-deepcave", "disabled"),
        ]

        @app.callback(inputs, outputs)
        def callback(n_clicks):
            # When clicking the Exit button, we first want to delete existing jobs and update the button
            if n_clicks is not None:
                queue.delete_jobs()
                return "danger", "Terminated DeepCAVE", True
            else:
                return "primary", "Exit", False

    def _callback_terminate_deepcave(self) -> None:
        inputs = [Input("exit-deepcave", "n_clicks")]
        outputs = []

        @app.callback(inputs, outputs)
        def callback(n_clicks):
            # Then we want to terminate DeepCAVE
            if n_clicks is not None:
                time.sleep(1)
                os._exit(130)

    def __call__(self) -> html.Header:
        return html.Header(
            className="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow",
            children=[
                html.A("DeepCAVE", className="navbar-brand me-0 px-3", href="#"),
                html.Button(className="navbar-toggler position-absolute d-md-none collapsed"),
                dcc.Location(id="matplotlib-mode-refresh", refresh=True),
                dbc.Button(
                    [
                        "Matplotlib",
                        dbc.Badge(
                            "off",
                            color="light",
                            text_color="black",
                            className="ms-2",
                            id="matplotlib-mode-badge",
                        ),
                    ],
                    color="secondary",
                    className="me-2",
                    id="matplotlib-mode-toggle",
                ),
                dbc.Button(
                    "Exit", color="secondary", className="me-2", id="exit-deepcave", disabled=False
                ),
            ],
        )
