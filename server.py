from deepcave import app, config
from deepcave.layouts.main import MainLayout


if __name__ == '__main__':
    app.layout = MainLayout(config.PLUGINS)()
    app.run_server(debug=True)
