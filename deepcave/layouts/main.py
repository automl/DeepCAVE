from typing import Dict, List

from urllib.parse import urlparse

from dash import dcc, html
from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import app, config, queue, run_handler
from deepcave.layouts import Layout
from deepcave.layouts.general import GeneralLayout
from deepcave.layouts.header import HeaderLayout
from deepcave.layouts.not_found import NotFoundLayout
from deepcave.layouts.notification import NotificationLayout
from deepcave.layouts.sidebar import SidebarLayout
from deepcave.plugins import Plugin
from deepcave.utils.dash import alert


class MainLayout(Layout):
    def __init__(self, categorized_plugins: Dict[str, List[Plugin]]):
        super().__init__()
        self.plugins = {}
        self.sidebar_layout = SidebarLayout(categorized_plugins)
        self.header_layout = HeaderLayout()
        self.general_layout = GeneralLayout()
        self.notification_layout = NotificationLayout()
        self.not_found_layout = NotFoundLayout  # Needs to be class here to add url
        for plugins in categorized_plugins.values():
            for plugin in plugins:
                self.plugins[plugin.id] = plugin

    def register_callbacks(self) -> None:
        output = Output("content", "children")
        input = Input("on-page-load", "pathname")

        @app.callback(output, input)  # type: ignore
        def display_page(pathname: str):  # type: ignore
            pathname = urlparse(pathname).path
            paths = pathname.split("/")[1:]

            if paths[0] == "":
                return self.general_layout()
            else:
                if not queue.ready():
                    return alert("At least one worker has to be enabled.")
                if len(run_handler.get_runs()) == 0:
                    return alert("Please select a run first.")
                else:
                    if paths[0] == "plugins":
                        plugin = self.plugins.get(paths[1], None)
                        if plugin is not None:
                            return plugin()

            return self.not_found_layout(pathname)()

    def __call__(self) -> Component:
        return html.Div(
            children=[
                dcc.Interval(id="global-update", interval=config.REFRESH_RATE),
                self.header_layout(),
                html.Div(
                    id="main-container",
                    className="container-fluid",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                self.sidebar_layout(),
                                html.Main(
                                    className="col-md-9 ms-sm-auto col-lg-10 px-md-4",
                                    children=[
                                        html.Div(
                                            className="",
                                            children=[
                                                dcc.Location(id="on-page-load", refresh=False),
                                                self.notification_layout(),
                                                html.Div(id="content"),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
