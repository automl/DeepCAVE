import dash_html_components as html

from deep_cave.layouts.layout import Layout
from deep_cave.plugins import plugin_names


class SidebarLayout(Layout):

    def __init__(self):
        super().__init__()

        nav_points = []
        for id, name in plugin_names.items():
            point = html.Li(
                className='nav-item',
                children=[html.A(name, className='nav-link active', href=f'/plugins/{id}')])
            nav_points.append(point)
        
        self.nav_points = nav_points

    def __call__(self):

        return \
            html.Nav(className='col-md-3 col-lg-2 d-md-block sidebar collapse', id='sidebarMenu', children=[
                html.Div(className='position-sticky pt-3', children=[
                    html.Ul(className='nav flex-column', children=[
                        html.A("General", className='nav-link active', href='/'),
                    ]),
                    html.H6(className='sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted', children=[
                        html.Span('Tools')
                    ]),
                    html.Ul(className='nav flex-column', children=self.nav_points)
                ]),
            ])

layout = SidebarLayout()