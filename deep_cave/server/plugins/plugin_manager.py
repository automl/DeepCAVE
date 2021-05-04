from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import dash_bootstrap_components as dbc

from ..server import app
from ..helper import get_study_data
from deep_cave.util.logs import get_logger
from deep_cave.util.parsing import deep_cave_data_decoder
from ..analysis_tools import plugins


logger = get_logger(__name__)

# use singleton class from https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html


class PluginManager:
    # there should only be one PluginManager per Server
    class __PluginManager:
        """
        Inner Class
        """
        # implement PluginManager here
        plugins = {}
        # todo prettier
        # string to generate a boostrap question mark as a mouseover tooltip
        question_mark = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle" viewBox="0 0 16 16"' \
                        ' id={}>' \
                        '<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>' \
                            '<path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/>'\
                        '</svg>'

        def __init__(self):
            """
            Singleton Class. There can only be one PluginManager per Server.
            """
            # the plugins come from the auto import process in analysis tools
            self.plugins = plugins

        def generate_uis(self) -> List:
            """
            Create the layout for plugins. Calls every plugin to get the layout information.
            The layout is without any customizations.

            Returns
            -------
                A list of html.Div objects which itself contain lists.
            """
            # the layout bootstrap "container" for every plugin. In the container there multiple rows
            # the first rwo is the question mark icon with the name of the plugin
            # the w-100 is splitting between rows
            # the next row consists of two cols which take 4/12 and 8/12 of the available space (12 boostrap convention)
            # the cols are user input and plugin output
            # The user input is a list of dash Components from plugin.generate_ui()
            # The plugin output is currently only a placeholder from plugin.default_output()
            # The last row is the button that triggers execution
            output = []
            for name, plugin in self.plugins.items():
                output.append(
                    html.Div(className='row shadow-sm p-3 mb-5 bg-white rounded', id=plugin.id, children=[
                        html.Div(className='col', children=[
                            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                                self.question_mark.format(plugin.id + '_tooltip_place')),
                            html.H3(name),
                            dbc.Tooltip(plugin.tooltip, target=plugin.id + '_tooltip_place')
                        ]),
                        html.Div(className='w-100'),
                        html.Div(className='col-4', id=plugin.id + '-user_inputs', children=plugin.generate_ui()),
                        html.Div(className='col-8', id=plugin.get_output_id(), children=plugin.default_output),
                        html.Div(className='w-100'),
                        html.Div(className='col', children=[html.Button(plugin.trigger_ids[0],
                                                                        id=plugin.trigger_ids[0],
                                                                        n_clicks=0,
                                                                        className="btn btn-outline-secondary")])
                    ])
                )
            return output

        def generate_side_bar_content(self) -> dcc.Checklist:
            """
            Generates the overview of all available plugins.

            Generates a checklist which is added to one of the cols in the bootstrap layout.
            Should function as a side bar on the website

            Returns
            -------
                Checklist with all avaialble plugins.
            """
            return dcc.Checklist(id='active', options=[{'label': name, 'value': name} for name in self.plugins])

        def register_callbacks(self):
            """
            Wrapper function. Iterates at the start of the sever all plugins and registers all callbacks.
            Then the plugin for customization is registered, which listens to the meta data table for the selection
            of studies.

            Returns
            -------
                None. Adds callbacks to dash.
            """
            for plugin in self.plugins.values():
                plugin.register_callback()
            if self.plugins:
                # add an additional callback, when the study its data are known, then customize the input elements
                @app.callback(
                    [Output(plugin.id + '-user_inputs', 'children') for plugin in self.plugins.values()],
                    [Input('table_of_studies', 'selected_row_ids')],
                    prevent_inital_call=True
                )
                def customize_plugins(study_ids):
                    if not study_ids:
                        return [plugin.generate_ui() for plugin in self.plugins.values()]
                    output = []
                    data, meta, models = get_study_data(study_ids[0])

                    for plugin in self.plugins.values():
                        logger.debug(f'Customizing {plugin.name}')
                        try:
                            customizations = plugin.ui_customization(meta=meta, data=data, models=models)
                        except Exception as e:
                            logger.exception(f'Plugin {plugin.name} threw an exception: {str(e)}')
                            logger.warning(f'{plugin.name}: assuming empty dict as output of ui_customization')
                            customizations = {}
                        output.append(plugin.generate_ui(customizations))
                    return output

    instance = None

    def __init__(self):
        """
        Simple singleton design pattern. When the constructor of this wrapper class is called, check if a
        static instance of __PluginManager already exists. If not create it and assign it to this class.
        """
        if not PluginManager.instance:
            PluginManager.instance = PluginManager.__PluginManager()

    def __getattr__(self, name):
        """
        Part of the wrapper. The __getattr__ directs all attribute requests to the instance object.
        This makes PluginManager a standin for the static __PluginManager instance.

        Parameters
        ----------
        name
            str. Name of the attribute

        Returns
        -------
            The attribute (e.g. property, function) from instance of __PluginManager
        """
        return getattr(self.instance, name)
