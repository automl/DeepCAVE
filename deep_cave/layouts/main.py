import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from deep_cave import app
from deep_cave.cache import cache
from deep_cave.layouts.layout import Layout
from deep_cave.layouts.header import layout as header_layout
from deep_cave.layouts.general import layout as general_layout
from deep_cave.layouts.not_found import layout as not_found_layout
from deep_cave.layouts.sidebar import layout as sidebar_layout
from deep_cave.plugins import plugin_layouts
from deep_cave.runs import get_selected_run


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
                if cache.get("run_id") is None:
                    return html.Div("Please select run first.")
                else:
                    # Cache run here
                    # check if new run is run in cache, otherwise empty it
                    
                    run = get_selected_run()
                    require_data = cache.get_required_data()

                    if cache.get("run") is not None:
                        if run != cache.get("run"):
                            cache.empty()
                            cache.set_dict(require_data)

                            # Print a message that the run changed and thus the cache was cleared

                    cache.set("run") = run



                    if paths[0] == "plugins":
                        for name, layout in plugin_layouts.items():
                            if name == paths[1]:
                                return layout()
            
            return not_found_layout

    def __call__(self):
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
