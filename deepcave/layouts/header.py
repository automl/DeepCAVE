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

#  noqa: D400
"""
# Header

This module defines the layout for visualizing the header.

It handles different callbacks of the layout.

## Classes
    - HeaderLayout: This class provides the header and its layout.
"""

from typing import List, Optional, Tuple

import os
import time

import dash_bootstrap_components as dbc
import requests
from dash import dcc, html
from dash.dependencies import Input, Output

from deepcave import app, config, queue
from deepcave.layouts import Layout


class HeaderLayout(Layout):
    """
    Provide the header and its layout.

    Register and handle callbacks.
    """

    def register_callbacks(self) -> None:
        """Register and handle the callbacks."""
        super().register_callbacks()
        self._callback_delete_jobs()
        self._callback_terminate_deepcave()

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
                    "Exit", color="secondary", className="me-2", id="exit-deepcave", disabled=False
                ),
            ],
        )
