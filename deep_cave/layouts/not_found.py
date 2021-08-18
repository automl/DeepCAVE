import dash_html_components as html

#from deep_cave.server.plugins.plugin_manager import PluginManager
from deep_cave.layouts.layout import Layout


class NotFoundLayout(Layout):
    def __call__(self):
        return html.H2('This page does not exists.')


layout = NotFoundLayout()
