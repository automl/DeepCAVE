import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from deep_cave.server.layout.table import table
from deep_cave.server import helper
from deep_cave.server.plugins.plugin_manager import PluginManager


# dcc.Dropdown(id='list_of_studies', options=helper.dash_list_of_studies(studies_location)),

'''
# refresh button
    html.Div(id='some-placeholder', className='row', children=[
        html.Div(className='col-2 shadow p-4 mb-4 bg-white', children=[
            dcc.Checklist(id='refresh', options=[{'label': 'Refresh', 'value': 'refresh'}])
        ])
    ]),
'''

'''
# activate plugins sidebar
html.Div(className='col-1 shadow p-4 mb-4 bg-white', children=PluginManager().generate_side_bar_content()),
'''

layout = html.Div(className='container-fluid', children=[
    dcc.Location(id='on-page-load', refresh=False),

    html.H1(id='title', children='Deep CAVE'),

    html.Span(children=table),

    html.Div(className='container-fluid', children=[
        html.Div(className='row', children=[
            html.Div(className='col-12 shadow p-4 mb-4 bg-white', id='plugin-placeholder',
                     children=PluginManager().generate_uis()),
        ])
    ]),


])
