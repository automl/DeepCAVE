import dash
import dash_bootstrap_components as dbc


def get_app():
    app = dash.Dash(
        __name__,
        title='Deep CAVE',
        update_title='',
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    return app


app = get_app()
