from typing import Dict, Type, Any

import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from dash.development.base_component import Component

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger


logger = get_logger(__name__)


class PerformanceOverTime(Plugin):

    @property
    def wip(self):
        return False

    @property
    def has_single_output(self):
        return False

    @property
    def tooltip(self) -> str:
        return 'Testing Plugin. Visualize the distribution of the configuration with the corresponding metric value.'

    @staticmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        return {'objective': dcc.Dropdown}

    @staticmethod
    def ui_customization(meta, data, **kwargs) -> Dict[str, Dict[str, Any]]:
        return {
            'objective': {
                'options': [{'label': col, 'value': col} for col in meta['metrics']],
                'value': meta.get('objective', None) or meta['metrics'][0]
            }
        }

    def process(self, data, meta, objective, **kwargs):
        hover_data = None
        if 'trial_meta.tags' in data.columns:
            hover_data = data['trial_meta.tags']
        fig = px.scatter(data, x='trial.increment', y=objective, hover_data=hover_data)
        return dcc.Graph(figure=fig)
