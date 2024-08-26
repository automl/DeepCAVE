# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D400
"""
# Notification

This module provides a notification layout.

It utilizes Dash and provides utilities for displaying notifications.
With a notification from the Notification module an alert component can be updated.
Callbacks are registered and handled.

## Classes
    - NotificationLayout: Layout class for displaying notifications.
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

    Provide callback registering methods.
    """

    def __init__(self) -> None:
        super().__init__()

    def register_callbacks(self) -> None:
        """Register callbacks for updating notification alert."""

        @app.callback(
            Output("alert", "children"),
            Output("alert", "color"),
            Output("alert", "is_open"),
            Trigger("global-update", "n_intervals"),
        )  # type: ignore
        def update_alert() -> Tuple[str, str, bool]:
            """
            Update the notification alert.

            Returns
            -------
            Tuple[str, str, bool]
                The message, color and True.
            """
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
