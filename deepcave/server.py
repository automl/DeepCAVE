import dash_bootstrap_components as dbc
from dash_extensions.enrich import (
    DashProxy,
    MultiplexerTransform,
    NoOutputTransform,
    TriggerTransform,
)

from deepcave.config import Config


def get_app(config: Config):
    app = DashProxy(
        __name__,
        title=config.TITLE,
        update_title="",
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        transforms=[
            # enable use of Trigger objects
            TriggerTransform(),
            # makes it possible to target an output multiple times in callbacks
            MultiplexerTransform(),
            # enable use of ServersideOutput objects
            # ServersideOutputTransform(),
            # enable callbacks without output
            NoOutputTransform(),
            # makes it possible to skip callback invocations while a callback is running
            # BlockingCallbackTransform(),
            # makes it possible to write log messages to a Dash component
            # LogTransform(),
        ],
    )
    return app
