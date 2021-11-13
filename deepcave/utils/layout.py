import io
from dash import html
import base64


def render_table(df):
    pass


def render_mpl_figure(fig):
    # create a virtual file which matplotlib can use to save the figure
    buffer = BytesIO()
    # save the image to memory to display in the web
    fig.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    # display any kind of image taken from
    # https://github.com/plotly/dash/issues/71
    encoded_image = base64.b64encode(buffer.read())
    return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), className='img-fluid')


def display_figure(fig):
    buf = io.BytesIO()  # in-memory files
    fig.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode(
        "utf8")  # encode to html elements
    return html.Img(src="data:image/png;base64,{}".format(data))


def get_slider_marks(strings=None, steps=10):
    if strings is None:
        return {0: str("None")}

    if len(strings) < steps:
        steps = len(strings)

    marks = {}

    marks = {}
    for i, string in enumerate(strings):
        if i % (len(strings) / steps) == 0:
            marks[i] = str(string)

    # Also include the last mark
    marks[len(strings)-1] = str(strings[-1])

    return marks


def get_select_options(strings=None):
    if strings is None:
        return []

    options = []
    for string in strings:
        options.append({"label": string, "value": string})

    return options


def get_checklist_options(strings=None):
    return get_select_options(strings)
