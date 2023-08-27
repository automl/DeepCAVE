#  noqa: D400
"""
# Not_Found

This module defines a custom layout for displaying a "NotFound" page.

## Contents
    - __call__: Create the "NotFound" page.
"""
from typing import List

from dash import html
from dash.development.base_component import Component

from deepcave.layouts import Layout


class NotFoundLayout(Layout):
    """
    This class defines a custom "NotFound" layout.

    Methods
    -------
    __call__
        Create the "NotFound" page.
    """

    def __init__(self, url) -> None:  # noqa: D107
        super().__init__()
        self.url = url

    def __call__(self) -> List[Component]:  # noqa: D102
        return [
            html.H2("This page does not exists."),
            html.Div(f"Tried to reach {self.url}."),
        ]
