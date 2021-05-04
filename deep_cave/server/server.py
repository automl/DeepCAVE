import dash

from deep_cave.util.logs import get_logger

from deep_cave.server.config import external_stylesheets


logger = get_logger(__name__)


app = dash.Dash(__name__,
                title='Deep CAVE',
                update_title='Loading...',
                external_stylesheets=external_stylesheets)


