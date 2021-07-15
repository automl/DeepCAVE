import dash_html_components as html
from deep_cave.layouts.layout import Layout


class HeaderLayout(Layout):
    def _get_layout(self):
        return html.Header(className='navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow', children=[
            html.A('DeepCave', className='navbar-brand col-md-3 col-lg-2 me-0 px-3', href='#'),
            html.Button(className='navbar-toggler position-absolute d-md-none collapsed')
        ])