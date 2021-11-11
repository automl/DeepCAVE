import plotly.express as px


def hex_to_rgb(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)


def get_color(id, alpha=1):
    colors = px.colors.qualitative.Plotly
    color = colors[id]

    R, G, B = hex_to_rgb(color)

    return f"rgba({R}, {G}, {B}, {alpha})"
