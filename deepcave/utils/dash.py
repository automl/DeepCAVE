import dash_bootstrap_components as dbc


def flash(message: str, category: str = "info"):
    """
    Flask style flash-message with Alerts
    https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/

    Possible categories:
        primary    -> blue
        secondary  -> light grey
        success    -> green
        warning    -> yellow
        danger     -> red
        info       -> cyan
        light      -> white
        dark       -> dark gray
    """
    return dbc.Alert(
        message,
        id=f"alert_{hash(message)}",
        is_open=True,
        dismissable=False,
        fade=True,
        color=category,
    )


def alert(message: str):
    return flash(message, "danger")
