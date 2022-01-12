from dash import html
from dash.development.base_component import Component

from deepcave.layouts.layout import Layout


class NotFoundLayout(Layout):
    def __call__(self) -> Component:
        return html.H2('This page does not exists.')


layout = NotFoundLayout()
