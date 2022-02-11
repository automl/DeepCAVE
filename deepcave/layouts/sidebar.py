from typing import Union

from dash import dcc, html
from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import app, queue
from deepcave.layouts import Layout
from deepcave.plugins import Plugin


class SidebarLayout(Layout):
    def __init__(self, categorized_plugins: dict[str, list[Plugin]]):
        super().__init__()
        self.plugins = categorized_plugins

        nav_points = {category: [] for category in categorized_plugins}
        for category, plugins in categorized_plugins.items():
            for plugin in plugins:
                nav_points[category].append((plugin.id, plugin.name, plugin.icon))

        self.nav_points = nav_points

    def register_callbacks(self):
        # Update queue information panel
        @app.callback(
            Output("queue-info", "children"),
            Input("queue-info-interval", "n_intervals"),
        )
        def update_queue_info(_):
            try:
                jobs = {}

                for job in queue.get_running_jobs():
                    display_name = job.meta["display_name"]
                    run_name = job.meta["run_name"]

                    if display_name not in jobs:
                        jobs[display_name] = []

                    jobs[display_name].append((run_name, "[R]"))

                for job in queue.get_pending_jobs():
                    display_name = job.meta["display_name"]
                    run_name = job.meta["run_name"]

                    if display_name not in jobs:
                        jobs[display_name] = []

                    jobs[display_name].append((run_name, "[P]"))

                items = []
                for display_name, run_names in jobs.items():
                    items += [
                        html.Li(
                            className="nav-item",
                            children=[html.A(f"{display_name}", className="nav-link")],
                        )
                    ]

                    for run_name, status in run_names:
                        items += [
                            html.Li(
                                className="nav-item",
                                children=[
                                    html.A(
                                        f"{status} {run_name}",
                                        className="nav-link disabled",
                                        style={"padding-top": 0, "padding-bottom": 0},
                                    )
                                ],
                            )
                        ]

                if len(jobs) > 0:
                    return [
                        html.Hr(),
                        html.H6(
                            className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted",
                            children=[html.Span("Queue Information")],
                        ),
                        html.Ul(className="nav flex-column", children=items),
                    ]

                return []
            except:
                return

    def __call__(self) -> Union[list[Component], Component]:

        layouts = []
        for category, points in self.nav_points.items():
            layouts += [
                html.H6(
                    className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted",
                    children=[html.Span(category)],
                )
            ]

            point_layouts = []
            for (id, name, icon) in points:
                point_layouts += [
                    html.Li(
                        className="nav-item",
                        children=[
                            html.A(
                                [html.I(className=icon), name],
                                className="nav-link",
                                href=f"/plugins/{id}",
                            )
                        ],
                    )
                ]

            layouts += [html.Ul(className="nav flex-column", children=point_layouts)]

        icon = {"data-feather": "file-text"}

        return html.Nav(
            className="col-md-3 col-lg-2 d-md-block sidebar collapse",
            id="sidebarMenu",
            children=[
                html.Div(
                    className="position-sticky pt-3",
                    children=[
                        html.Ul(
                            className="nav flex-column",
                            children=[
                                html.A(
                                    "General",
                                    className="nav-link active",
                                    href="/",
                                ),
                            ],
                        ),
                        *layouts,
                    ],
                ),
                dcc.Interval(id="queue-info-interval", interval=1000),
                html.Div(id="queue-info"),
            ],
        )
