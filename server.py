from deep_cave import app
from deep_cave.layouts.main import layout


if __name__ == '__main__':
    app.layout = layout()
    app.run_server(debug=True)
