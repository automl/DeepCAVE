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
# Dash

This module provides utilities to return and flash alerts.
"""

import dash_bootstrap_components as dbc


def flash(message: str, category: str = "info") -> dbc.Alert:
    """
    Flask style flash-message with Alerts.

    Parameters
    ----------
    message : str
        The message to be displayed.
    category : str, optional
        The category of the alert.
        Default is "info".

    Returns
    -------
    dbc.Alert
        The alert object.
    """
    return dbc.Alert(
        message,
        id=f"alert_{hash(message)}",
        is_open=True,
        dismissable=False,
        fade=True,
        color=category,
    )


def alert(message: str) -> dbc.Alert:
    """
    Return an alert message.

    Parameters
    ----------
    message : str
        The message of the alert.

    Returns
    -------
    dbc.Alert
        The alert object.
    """
    return flash(message, "danger")
