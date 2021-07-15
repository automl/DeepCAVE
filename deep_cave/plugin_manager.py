import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from deep_cave.server import app
from deep_cave.util.logs import get_logger
from deep_cave.plugins.plugin import Plugin
from deep_cave.util.importing import auto_import_iter


logger = get_logger(__name__)


class PluginManager:

    instance = None

    @staticmethod 
    def getInstance():
        if PluginManager.instance == None:
            PluginManager.instance = PluginManager()

        return PluginManager.instance

    def __init__(self):
        if PluginManager.instance != None:
            raise Exception("This class is a singleton!")
        
        self._load_plugins()

    def _load_plugins(self):
        self.plugins = []

        paths = [os.path.join(os.path.dirname(__file__), 'plugins', '*')]
        for _, obj in auto_import_iter("plugins", paths):
            if not issubclass(obj, Plugin):
                continue

            # Plugin itself is a subclass, filter it out
            if obj == Plugin:
                continue

            self.plugins.append(obj)

    def get_plugin_layouts(self):
        layouts = {}

        for plugin in self.plugins:
            layouts[plugin.id()] = plugin()

        return layouts

    def get_plugin_names(self):
        names = {}

        for plugin in self.plugins:
            names[plugin.id()] = plugin.name()

        return names


pm = PluginManager.getInstance()
    
__all__ = ["pm"]
