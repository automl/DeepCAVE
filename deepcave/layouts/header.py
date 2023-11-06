#  noqa: D400
"""
# Header

This module defines the layout for visualizing the header.

It handles different callbacks of the layout.

## Classes
    - HeaderLayout: This class provides the header and its layout.
"""


from typing import Literal, Tuple, Union

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from deepcave import app, c
from deepcave.layouts import Layout


class HeaderLayout(Layout):
    """
    Provide the header and its layout.

    Register and handle callbacks.
    """

    def register_callbacks(self) -> None:
        """Register and handle the callbacks."""
        super().register_callbacks()

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
        def update_matplotlib_mode(
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
            ],
        )
