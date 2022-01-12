import plotly.express as px


def hex_to_rgb(hex_string: str) -> tuple[int, int, int]:
    """
    Converts a hex_string to a tuple of rgb values.
    Requires format including #, e.g.:
    #000000
    #ff00ff
    """
    if len(hex_string) != 7:
        raise ValueError(f"Invalid length for #{hex_string}")

    if any(c not in "0123456789ABCDEF" for c in hex_string.lstrip("#").upper()):
        raise ValueError(f"Invalid character in #{hex_string}")

    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)


def get_color(id_: int, alpha: float = 1) -> str:
    """
    Currently (Plotly version 5.3.1) there are 10 possible colors.
    """
    color = px.colors.qualitative.Plotly[id_]

    r, g, b = hex_to_rgb(color)

    return f"rgba({r}, {g}, {b}, {alpha})"
