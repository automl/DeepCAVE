from src import app
from src.layouts.main import layout


if __name__ == '__main__':
    app.layout = layout()
    app.run_server(debug=True)
