from deep_cave.server import app
from deep_cave.layouts.main import MainLayout


if __name__ == '__main__':
    app.layout = MainLayout()
    app.run_server(debug=True)