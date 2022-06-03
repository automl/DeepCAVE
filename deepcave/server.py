from deepcave import app, config
from deepcave.layouts.main import MainLayout

if __name__ == "__main__":
    print("\n-------------STARTING SERVER-------------")
    app.layout = MainLayout(config.PLUGINS)()
    app.run_server(debug=config.DEBUG, port=config.DASH_PORT, host=config.DASH_ADDRESS)
