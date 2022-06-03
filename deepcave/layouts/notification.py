from typing import List, Tuple, Union

import dash_bootstrap_components as dbc
from dash.dependencies import Output
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Trigger

from deepcave import app, notification
from deepcave.layouts import Layout


class NotificationLayout(Layout):
    def __init__(self) -> None:
        super().__init__()

    def register_callbacks(self) -> None:
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

    def __call__(self) -> Union[List[Component], Component]:
        return dbc.Alert(
            id="alert",
            is_open=False,
            dismissable=True,
            fade=True,
        )
