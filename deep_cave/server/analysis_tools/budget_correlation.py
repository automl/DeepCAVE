from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
from dash.development.base_component import Component
import dash_table

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations_with_replacement

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger

logger = get_logger(__name__)


class BudgetCorrelation(Plugin):
    @property
    def wip(self):
        return False

    @property
    def default_output(self):
        return dash_table.DataTable()

    @property
    def tooltip(self) -> str:
        return 'Spearman correlation between trials with the same config_id and a different fidelity.'

    @staticmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        return {'objective': dcc.Dropdown, 'fidelities': dcc.Dropdown}

    @staticmethod
    def ui_customization(meta, data, **kwargs) -> Dict[str, Dict[str, Any]]:
        return {
            'objective': {
                'options': [{'label': col, 'value': col} for col in meta['metrics']],
                'value': meta.get('objective', None) or meta['metrics'][0]
            },
            'fidelities': {
                'options': [{'label': col, 'value': col} for col in data['trial.fidelity'].unique()],
                'value': data['trial.fidelity'].unique(),
                'multi': True
            }
        }

    def process(self, data, meta, objective, fidelities, **kwargs):
        logger.debug(f'Process with objective={objective}')
        # get all available fidelities
        # fidelities = np.sort(data['trial.fidelity'].unique())
        if fidelities is None:
            raise ValueError('Fidelities not specified')
        # create a df to return
        output = pd.DataFrame([], columns=fidelities, index=fidelities)
        # generate pairwise combinations. Assumption changing the order doesn't effect the result
        for f1, f2 in combinations_with_replacement(fidelities, 2):
            if f1 == f2:
                output.loc[f1, f2] = '1'
                continue
            # spearman needs to equally sized arrays. Select all values belonging to fidelity f1 and f2
            # which have intersecting config_ids. Meaning a config which was run with 2 different config_ids
            intersection = pd.merge(data[data['trial.fidelity'] == f1][[objective, 'trial.config_id']],
                                    data[data['trial.fidelity'] == f2][[objective, 'trial.config_id']],
                                    on='trial.config_id', how='inner',
                                    suffixes=(f'_{f1}', f'_{f2}'))
            # inner guarantees, that the keys (on='config_id') are in both dfs
            rho, pval = spearmanr(intersection[f'{objective}_{f1}'], intersection[f'{objective}_{f2}'],
                                  nan_policy='omit')
            result_string = f'{rho:.2f} (pval={pval:.3f})'
            output.loc[f1, f2] = result_string
            output.loc[f2, f1] = result_string
        output = output.sort_index().T.sort_index().T.reset_index()
        return dash_table.DataTable(
            columns=[{"name": str(i) if i != 'index' else 'trial.fidelity', "id": str(i)} for i in output.columns],
            data=output.to_dict('records'),
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
