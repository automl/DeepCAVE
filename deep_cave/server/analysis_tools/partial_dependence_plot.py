from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
from dash.development.base_component import Component

from pdpbox.info_plots import actual_plot, target_plot

import pandas as pd

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger
from deep_cave.util.util import matplotlib_to_html_image


logger = get_logger(__name__)


class PartialDependencePlot(Plugin):
    @property
    def wip(self):
        return False

    @staticmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        return {'objective': dcc.Dropdown, 'fidelity': dcc.Dropdown, 'feature': dcc.Dropdown}

    @property
    def tooltip(self) -> str:
        return 'Shows the marginal effect of one feature for the outcome variable. If a model was locked, the model' \
               'is queried for more data. Otherwise the graph is generated form the logged data alone.'

    @staticmethod
    def ui_customization(meta, data, models) -> Dict[str, Dict[str, Any]]:
        if models:
            # if models were logged, use the features supported in this model
            features = [feat for multi_feature in models['default']['mapping'].values() for feat in multi_feature]
        else:
            # if not, make all parameters in config available
            features = meta['config']
            models = {}
        # if models are registered only use the fidelities of the model
        # otherwise use the fidelities contained in the data
        supported_filities = models.keys() or data['trial.fidelity'].unique()
        return {
            'objective': {
                'options': [{'label': col, 'value': col} for col in meta['metrics']],
                'value': meta.get('objective', None) or meta['metrics'][0]
            },
            'fidelity': {
                'options': [{'label': col, 'value': col} for col in supported_filities],
                'value': 'default'
            },
            'feature': {
                'options': [{'label': col, 'value': col} for col in features],
                'value': features[0]
            }
        }

    def process(self, data, meta, objective, fidelity, feature, models):
        logger.debug(f'Process with objective={objective}, fidelity={fidelity}, feature={feature}')

        if fidelity is not None and fidelity != 'default':
            data = data[data['trial.fidelity'] == fidelity]
        if models:
            surrogate_model = self.get_surrogate_model(models, fidelity)
            # todo fix for models
            fig, axes, summary_df = actual_plot(surrogate_model, data, feature, feature, predict_kwds={})
        else:
            if 'search_space' in meta:
                if meta['search_space']:
                    data, mapping = self.encode_data(data, meta['search_space'])
                    fig, axes, summary_df = target_plot(data, mapping[feature], feature, target=objective)
                    # return matplotlib_to_html_image(fig)
            else:
                fig, axes, summary_df = target_plot(data, feature, feature, target=objective)
        return matplotlib_to_html_image(fig)
