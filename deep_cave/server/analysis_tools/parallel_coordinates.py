from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go
from dash.development.base_component import Component

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.logs import get_logger


logger = get_logger(__name__)


class ParallelCoordinates(Plugin):
    @property
    def wip(self):
        return False

    @staticmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        return {'objective': dcc.Dropdown, 'features': dcc.Dropdown}

    @property
    def tooltip(self) -> str:
        return 'Multidimensional visualization of configurations. Colored by selected metric. The axis and can be' \
               'moved.'

    @staticmethod
    def ui_customization(meta, data, models, **kwargs) -> Dict[str, Dict[str, Any]]:
        if meta.get('objective', None):
            objective_ = meta['objective']
        else:
            objective_ = meta['metrics'][0]
        if models:
            features_ = models['default']['mapping']['X']
        else:
            features_ = meta['config']

        return {
            'objective': {
                'options': [{'label': col, 'value': col} for col in meta['metrics']],
                'value':  objective_
            },
            'features': {
                'options': [{'label': col, 'value': col} for col in meta['config']],
                'value':  features_,
                'multi': True
            }
        }

    def process(self, data, meta, objective, features, **kwargs):
        logger.debug(f'Process with objective={objective}')
        features.extend([objective])
        # todo fix parallel coordinates for categorical values

        dimensions = []
        for col in features:
            if data[col].dtype != 'object':
                dimensions.append({
                    'range': [data[col].min(), data[col].max()],
                    'label': col,
                    'values': data[col].values
                })
            else:
                cat_data = data[col].astype('category')
                num_cat = len(cat_data.cat.categories)
                dimensions.append({
                    'range': [0, num_cat],
                    'label': col,
                    'values': cat_data.cat.codes.values,
                    'ticktext': cat_data.cat.categories.values,
                    'tickvals': list(range(0, num_cat))
                })

        return go.Figure(go.Parcoords(dimensions=dimensions,
                                      line=dict(color=data[objective],
                                                colorscale=px.colors.diverging.Tealrose)))
        '''
        return px.parallel_coordinates(data,
                                       color=objective,
                                       dimensions=features,
                                       color_continuous_scale=px.colors.diverging.Tealrose)
        '''
