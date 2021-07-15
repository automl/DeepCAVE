import dash_html_components as html

from deep_cave.plugin_manager import pm
from deep_cave.layouts.layout import Layout


class SidebarLayout(Layout):

    def _define_variables(self):
        nav_points = []
        for id, name in pm.get_plugin_names().items():
            point = html.Li(
                className='nav-item',
                children=[html.A(name, className='nav-link active', href=f'/plugins/{id}')])
            nav_points.append(point)
        
        self.nav_points = nav_points
        

    def _get_layout(self):

        return \
            html.Nav(className='col-md-3 col-lg-2 d-md-block bg-light sidebar collapse', id='sidebarMenu', children=[
                html.Div(className='position-sticky pt-3', children=[
                    html.Ul(className='nav flex-column', children=[
                        html.A("General", className='nav-link active', href='/'),
                        html.A("Hyperparameters", className='nav-link active', href='/hyperparameters'),
                    ]),
                    html.H6(className='sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted', children=[
                        html.Span('Tools')
                    ]),
                    html.Ul(className='nav flex-column', children=self.nav_points)
                ]),
            ])