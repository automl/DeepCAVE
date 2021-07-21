
import os
import dash
import dash_bootstrap_components as dbc


from deep_cave.data_manager import dm
from deep_cave.util.logs import get_logger


logger = get_logger(__name__)


def get_app():
    app = dash.Dash(
        __name__,
        title='Deep CAVE',
        update_title='',
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    # Setup storage to be synced to the server
    dm.setup_storage(app.server)

    return app


app = get_app()


__all__ = ["app"]