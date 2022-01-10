from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import app, queue
from deepcave.layouts.general import layout as general_layout
from deepcave.layouts.header import layout as header_layout
from deepcave.layouts.layout import Layout
from deepcave.layouts.not_found import layout as not_found_layout
from deepcave.layouts.sidebar import layout as sidebar_layout
from deepcave.plugins import plugin_layouts
from deepcave.runs.handler import handler
from deepcave.utils.dash import alert


class MainLayout(Layout):
    def __init__(self):
        super().__init__()

    def register_callbacks(self):
        output = Output('content', 'children')
        input = Input('on-page-load', 'pathname')

        @app.callback(output, input)
        def display_page(pathname):
            paths = pathname.split("/")[1:]

            if paths[0] == "":
                return general_layout()
            else:
                if not queue.ready():
                    return alert("At least one worker has to be enabled.")
                if len(handler.get_run_names()) == 0:
                    return alert("Please select a run first.")
                else:
                    if paths[0] == "plugins":
                        for name, layout in plugin_layouts.items():
                            if name == paths[1]:
                                return layout()

            return not_found_layout

    def __call__(self) -> Component:
        return \
            html.Div(children=[
                header_layout(),
                html.Div(id='main-container', className='container-fluid', children=[
                    html.Div(className='row', children=[
                        sidebar_layout(),
                        html.Main(className='col-md-9 ms-sm-auto col-lg-10 px-md-4', children=[
                            html.Div(className='', children=[
                                dcc.Location(id='on-page-load', refresh=False),
                                html.Div(id='content'),
                                html.Div(id='general_message')
                            ])
                        ])
                    ]),
                ]),
            ])


layout = MainLayout()
