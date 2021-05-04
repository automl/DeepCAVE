import os

import pandas as pd
import numpy as np
import time

import plotly.express as px

from tests.api_tests import api_test

import deep_cave


def one_million_rendered():
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    np.random.seed(1)

    N = int(1e6)

    df = pd.DataFrame(dict(x=np.random.randn(N),
                           y=np.random.randn(N)))

    fig = px.scatter(df, x="x", y="y", render_mode='webgl')

    fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


def one_million_save():
    # todo make this run efficiently
    # for this test to succeed the backend needs to be more scalable to log all this data efficiently
    uri = os.path.join(os.path.join(os.path.dirname(__file__), '../studies/'))

    deep_cave.set_tracking_uri(uri)
    deep_cave.set_study('stress_test')
    # 100.000 configs * 200 params with a lot of budgets
    # should only use disk space in the realm of MB
    # RAM should be during processing at 1-2 GB
    # and running this simulation should only take 1-2 Minutes
    # running the analysis should be in the realm of seconds to a minute
    start_time = time.time()
    print(start_time)
    api_test.run_workflow(n_configs=100000, num_params=200)
    print(time.time()-start_time)
    # writing 1GB of Log data in 451 seconds (7.5 minutes)
    # loading this data takes about a minute and uses 12-15 GB
    # This done the visualization is super fast and relatively ressource efficient


if __name__ == '__main__':
    one_million_save()
