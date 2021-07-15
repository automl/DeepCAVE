import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from deep_cave.server import app
from deep_cave.plugin_manager import pm
from deep_cave.layouts.layout import Layout
from deep_cave.layouts.header import HeaderLayout
from deep_cave.layouts.general import GeneralLayout
from deep_cave.layouts.not_found import NotFoundLayout
from deep_cave.layouts.sidebar import SidebarLayout
from deep_cave.layouts.hyperparameters import HyperparameterLayout


class MainLayout(Layout):
    def _define_variables(self):
        # We set up the plugins here
        pass

    def _register_callbacks(self):
        # It's important to call the sublayouts first
        # otherwise they are not registered
        # when visiting the page in the first place
        general_layout = GeneralLayout()
        not_found_layout = NotFoundLayout()
        hyperparameters_layout = HyperparameterLayout()

        # Register plugins here
        plugin_layouts = pm.get_plugin_layouts()

        output = Output('content', 'children')
        input = Input('on-page-load', 'pathname')
        @app.callback(output, input)
        def display_page(pathname):
            paths = pathname.split("/")[1:]

            if paths[0] == "":
                return general_layout
            elif paths[0] == "hyperparameters":
                return hyperparameters_layout
            elif paths[0] == "plugins":
                for name, layout in plugin_layouts.items():
                    if name == paths[1]:
                        return layout
            else:
                return not_found_layout
    

    def _get_layout(self):
        return \
            html.Div(children=[
                HeaderLayout(),
                html.Div(className='container-fluid', children=[
                    html.Div(className='row', children=[
                        SidebarLayout(),
                        html.Main(className='col-md-9 ms-sm-auto col-lg-10 px-md-4', children=[
                            html.Div(className='', children=[
                                dcc.Location(id='on-page-load', refresh=False),
                                html.Div(id='content'),
                            ])
                        ])
                    ]),
                ]),   
            ])
