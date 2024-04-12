#  noqa: D400
"""
# Header

This module defines the layout for visualizing the header.

It handles different callbacks of the layout.

## Classes
    - HeaderLayout: This class provides the header and its layout.
"""


from typing import List, Literal, Optional, Tuple, Union

import os
import time

import dash_bootstrap_components as dbc
import requests
from dash import dcc, html
from dash.dependencies import Input, Output

from deepcave import app, c, config, queue
from deepcave.layouts import Layout


class HeaderLayout(Layout):
    """
    Provide the header and its layout.

    Register and handle callbacks.
    """

    def register_callbacks(self) -> None:
        """Register and handle the callbacks."""
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

        @app.callback(outputs, inputs)  # type: ignore
        def callback(
            n_clicks: int, pathname: str
        ) -> Union[
            Tuple[Literal["primary"], Literal["on"], str],
            Tuple[Literal["secondary"], Literal["off"], str],
        ]:
            """
            Update the matplotlib mode.

            Parameters
            ----------
            n_clicks : int
                Number of clicks.
            pathname : str
                Pathname.

            Returns
            -------
            Tuple[Literal["primary"], Literal["on"], str],
            Tuple[Literal["secondary"], Literal["off"], str]
                Tuple of either "primary", "on", pathname or "secondary", "off", pathname.
            """
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

        @app.callback(inputs, outputs)  # type: ignore
        def callback(n_clicks: Optional[int]) -> Tuple[str, str, bool]:
            # When clicking the Exit button, first existing jobs are deleted and then the button
            # is updated
            if n_clicks is not None:
                queue.delete_jobs()
                return "danger", "Terminated DeepCAVE", True
            else:
                return "primary", "Exit", False

    def _callback_terminate_deepcave(self) -> None:
        inputs = [Input("exit-deepcave", "n_clicks")]
        outputs: List[Output] = []

        @app.callback(inputs, outputs)  # type: ignore
        def callback(n_clicks: Optional[int]) -> None:
            # Then we want to terminate DeepCAVE
            if n_clicks is not None:
                time.sleep(1)
                requests.post(f"http://localhost:{config.DASH_PORT}/shutdown")
                os._exit(130)

    def __call__(self) -> html.Header:  # noqa: D102
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
