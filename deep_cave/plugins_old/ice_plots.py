from typing import Dict, Type, Any

import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go
from dash.development.base_component import Component

from pdpbox.pdp import pdp_isolate, pdp_plot
from sklearn.inspection import plot_partial_dependence, partial_dependence

import pandas as pd

from deep_cave.server.plugins.plugin import Plugin
from deep_cave.server.analysis_tools.partial_dependence_plot import PartialDependencePlot
from deep_cave.util.logs import get_logger
from deep_cave.util.util import matplotlib_to_html_image


logger = get_logger(__name__)


class IndividualConditionalExpectationPlot(PartialDependencePlot):

    @property
    def name(self):
        return 'Individual Conditional Expectation (ICE) plot'

    @property
    def tooltip(self) -> str:
        return 'Shows the marginal effect of one feature for the outcome variable. Only implemented for studies with' \
               'Surrogate models.'

    def process(self, data, meta, objective, fidelity, feature, models):
        logger.debug(f'Process with objective={objective}, fidelity={fidelity}, feature={feature}')

        if fidelity is not None and fidelity != 'default':
            data = data[data['trial.fidelity'] == fidelity]
        if models:
            surrogate_model = self.get_surrogate_model(models, fidelity)
            '''
            # sklearn implementation
            available_features = models[fidelity]['mapping']['X']
            result = plot_partial_dependence(surrogate_model, data[available_features],
                                        [list.index(available_features, feature)],
                                        kind='individual')

            return matplotlib_to_html_image(result.figure_)
            '''
            # '''
            # sklearn interactive implementation
            available_features = models[fidelity]['mapping']['X']
            result = partial_dependence(surrogate_model, data[available_features],
                                        feature,
                                        kind='both')
            df = pd.DataFrame(result.individual[0], columns=result['values'][0])

            # long format
            tmp = df.T.unstack().reset_index()
            tmp.columns = ['id', feature, objective]
            tmp['group'] = 'individual'

            '''
            # wide format
            df = df.T
            df.columns.name = 'configurations'
            df.index.name = feature
            df['group'] = 'individual
            '''
            available_features = models[fidelity]['mapping']['X']
            other_features = list(set(available_features) - set([feature, objective]))
            tmp = tmp.join(data[other_features], on='id')
            fig = px.line(tmp, x=feature, y=objective, color='group', line_group='id',
                          hover_data=other_features)
            fig.update_traces(line=dict(color="blue", width=0.2))
            fig.add_trace(go.Scattergl(x=result['values'][0], y=result['average'][0], mode='lines', name='average',
                                      line=go.scattergl.Line(width=3, color='red'), showlegend=True))
            return dcc.Graph(figure=fig)
            # '''
            '''
            # pdpbox implementation
            pdp_isolate_out = pdp_isolate(surrogate_model, data, models[fidelity]['mapping']['X'], feature)
            fig, axes = pdp_plot(pdp_isolate_out, feature, plot_lines=True, plot_pts_dist=True,
                                 plot_params={
                                     # plot title and subtitle
                                     'title': 'ICE plot for feature "%s"' % feature,
                                     'title_fontsize': 15,
                                     'subtitle_fontsize': 12,
                                     'font_family': 'Arial',
                                     # matplotlib color map for ICE lines
                                     'line_cmap': 'Blues',
                                     'xticks_rotation': 0,
                                     # pdp line color, highlight color and line width
                                     'pdp_color': '#1A4E5D',
                                     'pdp_hl_color': '#FEDC00',
                                     'pdp_linewidth': 1.5,
                                     # horizon zero line color and with
                                     'zero_color': '#E75438',
                                     'zero_linewidth': 1,
                                     # pdp std fill color and alpha
                                     'fill_color': '#66C2D7',
                                     'fill_alpha': 0.2,
                                     # marker size for pdp line
                                     'markersize': 3.5,
                                 })
            return matplotlib_to_html_image(fig)
            '''
        else:
            raise NotImplementedError('ICE plots are only implemented for studies with surrogate models')
