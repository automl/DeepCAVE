import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from deep_cave.layouts.layout import Layout
from deep_cave import app
from deep_cave import cache


class HeaderLayout(Layout):
    def register_callbacks(self):
        super().register_callbacks()

        outputs = [
            Output('matplotlib-mode-toggle', 'color'),
            Output('matplotlib-mode-badge', 'children'),
            Output('matplotlib-mode-refresh', 'href'),
        ]
        inputs = [
            Input('matplotlib-mode-toggle', 'n_clicks'),
            Input('matplotlib-mode-refresh', 'pathname'),
        ]

        @app.callback(outputs, inputs)
        def update_matplotlib_mode(n_clicks, pathname):
            update = None
            mode = cache.get("matplotlib-mode")
            if mode is None:
                mode = False

            if n_clicks is not None:
                update = pathname
                mode = not mode
                cache.set("matplotlib-mode", value=mode)

            if mode:
                return "primary", "on", update
            else:
                return "secondary", "off", update

    def __call__(self):
        return html.Header(className='navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow', children=[
            html.A(
                'DeepCave', className='navbar-brand me-0 px-3', href='#'),
            html.Button(
                className='navbar-toggler position-absolute d-md-none collapsed'),

            dcc.Location(id="matplotlib-mode-refresh", refresh=True),
            dbc.Button(
                ["Matplotlib", dbc.Badge(
                    "off", color="light", className="ml-2", id="matplotlib-mode-badge")],
                color="secondary",
                className="mr-2",
                id="matplotlib-mode-toggle"
            ),
        ])


layout = HeaderLayout()
