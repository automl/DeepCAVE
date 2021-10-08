

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
