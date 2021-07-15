import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from deep_cave.server import app
from deep_cave.layouts.layout import Layout


class HyperparameterLayout(Layout):

    def _register_callbacks(self):
        inputs = Input('on-page-load', 'href'),
        outputs = Output('hyperparameters-output', 'children'),

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def hyperparameters_update(_):
            return "hi2"

    def _get_layout(self):
        return [
            html.H1('Hyperparameters'),
            html.Div(id='hyperparameters-output')
        ]