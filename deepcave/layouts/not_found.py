#  noqa: D400
"""
# Not_Found

This module defines a custom layout for displaying a "NotFound" page.

## Classes
    - NotFoundLayout: Define a custom "NotFound" layout.
"""
from typing import List

from dash import html
from dash.development.base_component import Component

from deepcave.layouts import Layout


class NotFoundLayout(Layout):
    """
    Define a custom "NotFound" layout.

    Properties
    ----------
    url : str
        The url that could not be reached.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def __call__(self) -> List[Component]:  # noqa: D102
        return [
            html.H2("This page does not exists."),
            html.Div(f"Tried to reach {self.url}."),
        ]
