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
from deep_cave.converter import converters
from deep_cave.data_manager import dm


logger = get_logger(__name__)


class RunManager:

    instance = None

    @staticmethod 
    def getInstance():
        if RunManager.instance == None:
            RunManager.instance = RunManager()

        return RunManager.instance

    def __init__(self):
        if RunManager.instance != None:
            raise Exception("This class is a singleton!")

    def update(self):
        # Get current converter first
        self.converter = converters[dm.get("converter_name")]()

    def get_selected_run(self, selected_only=True):
        self.update()

        # Get data from converter
        return self.converter.get_selected_run()

    def get_run_ids(self, selected_only=False):
        self.update()

        return self.converter.get_run_ids(selected_only=selected_only)


rm = RunManager.getInstance()
    
__all__ = ["rm"]
