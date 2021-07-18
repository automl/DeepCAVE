import io
import base64
import dash_html_components as html


def display_figure(fig):
    buf = io.BytesIO() # in-memory files
    fig.savefig(buf, format = "png") # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return html.Img(src="data:image/png;base64,{}".format(data))