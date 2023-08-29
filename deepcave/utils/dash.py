#  noqa: D400
"""
# Dash

This module provides utilities to return and flash alerts.

## Contents
    - flash: Flask style flash-message with Alerts.
    - alert: Return an alert message.
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
        The message to be flashed.

    Returns
    -------
    dbc.Alert
        The alert object.
    """
    return flash(message, "danger")
