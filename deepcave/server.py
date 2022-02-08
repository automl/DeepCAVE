import dash
import dash_bootstrap_components as dbc

from deepcave.config import Config


def get_app(config: Config = None):
    if config is None:
        config = Config()
    app = dash.Dash(
        __name__,
        title=config.TITLE,
        update_title="",
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
    )
    return app
