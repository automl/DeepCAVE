from deepcave import app
from deepcave.layouts.main import layout


if __name__ == '__main__':
    app.layout = layout()
    app.run_server(debug=True)
