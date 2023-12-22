from deepcave import app, config
from deepcave.layouts.main import MainLayout

if __name__ == "__main__":
    print("\n-------------STARTING SERVER-------------")
    app.layout = MainLayout(config.PLUGINS)()
    app.run_server(debug=config.DEBUG, dev_tools_ui=config.DEV_TOOLS, port=config.DASH_PORT, host=config.DASH_ADDRESS)
