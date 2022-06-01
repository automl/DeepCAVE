from typing import List

from dash import html
from dash.development.base_component import Component

from deepcave.layouts import Layout


class NotFoundLayout(Layout):
    def __init__(self, url) -> None:
        super().__init__()
        self.url = url

    def __call__(self) -> List[Component]:
        return [
            html.H2("This page does not exists."),
            html.Div(f"Tried to reach {self.url}."),
        ]
