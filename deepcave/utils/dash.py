import dash_bootstrap_components as dbc


def flash(message: str, category: str = "info") -> dbc.Alert:
    """Flask style flash-message with Alerts."""
    return dbc.Alert(
        message,
        id=f"alert_{hash(message)}",
        is_open=True,
        dismissable=False,
        fade=True,
        color=category,
    )


def alert(message: str) -> dbc.Alert:
    return flash(message, "danger")
