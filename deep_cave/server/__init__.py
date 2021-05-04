import json

from dash.dependencies import Input, Output, State
from dash_table.Format import Format
import plotly.express as px

import pandas as pd

from deep_cave.server.helper import get_type, get_study_data
from deep_cave.server.state import get_studies
from deep_cave.server.server import app
from deep_cave.util.logs import get_logger
from deep_cave.server.layout.layout import layout
from deep_cave.server.config import studies_location
from deep_cave.server.plugins.plugin_manager import PluginManager
from deep_cave.util.parsing import deep_cave_data_encoder


logger = get_logger(__name__)


def general_callback_init():
    @app.callback(
        [Output('table_of_studies', 'data'),
         Output('table_of_studies', 'columns'),
         Output('table_of_studies', 'row_selectable')],
        [Input('on-page-load', 'href')],
        prevent_inital_call=True
    )
    def set_dash_table_data_and_columns_with_trials_of_study(trigger):
        # logger.info('render_trials_for_study with study: ' + study)
        meta_records = [{**study, **{'id': study_name}} for study_name, study in get_studies().items()]
        base_cols = ['study_name', 'start_time', 'end_time', 'duration', 'objective']
        groups = []
        for study in meta_records:
            # get all column names
            groups.extend(list(study.keys()))
        # only retain the unique ones
        groups = list(set(groups))
        # remove all fixed column names defined in cols
        cols = base_cols.copy()
        groups = [x for x in groups if x not in cols]
        # add the groups to the end of cols, so the fixed columns always appear in the same order at the beginning
        cols.extend(groups)

        multi_column = [{"name": col,
                         "id": col,
                         "selectable": col not in base_cols,
                         "deletable": True
                         } for col in cols if col != 'id']
        return meta_records, multi_column, 'multi'


def main():
    general_callback_init()
    # init Plugins
    PluginManager().register_callbacks()
    app.layout = layout
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
