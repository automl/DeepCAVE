from dash import html

#from deepcave.server.plugins.plugin_manager import PluginManager
from deepcave.layouts.layout import Layout


class NotFoundLayout(Layout):
    def __call__(self):
        return html.H2('This page does not exists.')


layout = NotFoundLayout()
