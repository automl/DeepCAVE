"""
# Notification.

This module provides a notification layout.

It utilizes Dash and provides utilities for displaying notifications.
The here proivded NotificationLayout class inherits from Layout.
With a notification from the Notification module an alert component can be updated.
It can be updated in content, color as well as visibility.

## Contents
    - register_callbacks: Updates notification alert display
        - update_alert: Updates the notification
    - __call__: Generates notification alert
"""

from typing import List, Tuple, Union

import dash_bootstrap_components as dbc
from dash.dependencies import Output
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Trigger

from deepcave import app, notification
from deepcave.layouts import Layout


class NotificationLayout(Layout):
    """
    Layout class for displaying notifications.

    Extends Layout class, provides callback registering methods.

    Methods
    -------
    register_callbacks
        Register callback for updating notification alert.
    __call__
        Generate notification alert component.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()

    def register_callbacks(self) -> None:  # noqa: D102
        @app.callback(
            Output("alert", "children"),
            Output("alert", "color"),
            Output("alert", "is_open"),
            Trigger("global-update", "n_intervals"),
        )
        def update_alert() -> Tuple[str, str, bool]:
            if (result := notification.get_latest()) is not None:
                (message, color) = result
                return message, color, True
            else:
                raise PreventUpdate()

    def __call__(self) -> Union[List[Component], Component]:  # noqa: D102
        return dbc.Alert(
            id="alert",
            is_open=False,
            dismissable=True,
            fade=True,
        )
