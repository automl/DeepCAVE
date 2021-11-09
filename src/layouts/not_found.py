import dash_html_components as html

#from src.server.plugins.plugin_manager import PluginManager
from src.layouts.layout import Layout


class NotFoundLayout(Layout):
    def __call__(self):
        return html.H2('This page does not exists.')


layout = NotFoundLayout()
