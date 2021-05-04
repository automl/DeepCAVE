from typing import Dict, Type, Any

import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from dash.development.base_component import Component
import dash_table

from fanova import fANOVA

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger


logger = get_logger(__name__)


class FAnova(Plugin):

    @property
    def wip(self):
        return False

    @property
    def name(self):
        return 'fANOVA'

    @property
    def default_output(self):
        return dash_table.DataTable()

    @property
    def has_single_output(self):
        return False
    
    @property
    def tooltip(self) -> str:
        return 'Calculate global parameter importance'

    @staticmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        return {'objective': dcc.Dropdown, 'features': dcc.Dropdown, 'n_trees': dcc.Input,
                'min_samples_split': dcc.Input}

    @staticmethod
    def ui_customization(meta, data, **kwargs) -> Dict[str, Dict[str, Any]]:
        return {
            'objective': {
                'options': [{'label': col, 'value': col} for col in meta['metrics']],
                'value': meta.get('objective', None) or meta['metrics'][0]
            },
            'features': {
                'options': [{'label': col, 'value': col} for col in meta['config']],
                'value': meta['config'][0],
                'multi': True
            },
            'n_trees': {
                'type': 'number',
                'value': 30
            },
            'min_samples_split': {
                'type': 'number',
                'value': 3
            }
        }

    def process(self, data, meta, objective, features, n_trees, min_samples_split, **kwargs):
        search_space = None
        if 'search_space' in meta:
            if meta['search_space']:
                search_space = meta['search_space']
        if not isinstance(features, list):
            features = [features]
        # to allow match between features and config_space, modify col names

        if not search_space:
            # filter out everything that is constant
            cols_to_drop = []
            for col in meta['config']:
                if data[col].min() == data[col].max():
                    cols_to_drop.append(col.replace('.config', ''))

            config_cols = {feat: feat.replace('config.', '') for feat in set(data.columns) - set(cols_to_drop) if 'config.' in feat}
            # data = self.encode_data(data)
            for col in config_cols.keys():
                data[col] = pd.to_numeric(data[col])
        else:
            config_cols = {feat: feat.replace('config.', '') for feat in set(data.columns) if 'config.' in feat}
        input_data = data[list(config_cols.keys())].rename(columns=config_cols)
        f = fANOVA(input_data, data[objective].values, config_space=search_space,
                   n_trees=n_trees, min_samples_split=min_samples_split)
        if not search_space:
            output = {feat: f.quantify_importance(("x_%03i" % list.index(list(config_cols.keys()), feat), )) for feat in features}
        else:
            output = {feat: f.quantify_importance((config_cols[feat],)) for feat in features}

        return dash_table.DataTable(
            columns=[{"name": str(i), "id": str(i)} for i in list(list(output[list(output.keys())[0]].values())[0].keys())],
            data=[{**list(value.values())[0], **{'id': key}} for key, value in output.items()],
            style_table={'overflowY': 'auto', 'overflowX': 'auto'},
            fixed_rows={'headers': True},
            style_data_conditional=[
                {
                    'if': {'column_id': 'index'},
                    "backgroundColor": "#f7f7f7",
                    'color': 'black',
                    "fontWeight": "bold"
                }
            ],
            css=[{'selector': '.row', 'rule': 'margin: 0'}]
        )
