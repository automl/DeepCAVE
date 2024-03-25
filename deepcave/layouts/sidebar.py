#  noqa: D400
"""
# SidebarLayout

This module defines layout for the sidebar.

Callbacks are registered and handled.

## Classes
    - SidebarLayout: Customize the Sidebar Layout.
"""

from typing import Dict, List, Tuple, Union

import dash_bootstrap_components as dbc
from dash import ALL, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash_extensions.enrich import Trigger

from deepcave import app, queue
from deepcave.layouts import Layout
from deepcave.plugins import Plugin


class SidebarLayout(Layout):
    """
    Customize the Sidebar Layout.

    Callbacks are registered and handled.

    Properties
    ----------
    plugins : Dict[str, List[Plugin]]
        A dictionary of all categorized plugins.
    nav_points : Dict[str, List[Tuple[str, str, str]]]
        A dictionary with plugins attributes corresponding to their category.
    """

    def __init__(self, categorized_plugins: Dict[str, List[Plugin]]) -> None:
        super().__init__()
        self.plugins = categorized_plugins

        nav_points: Dict[str, List[Tuple[str, str, str]]] = {
            category: [] for category in categorized_plugins
        }
        for category, plugins in categorized_plugins.items():
            for plugin in plugins:
                nav_points[category].append((plugin.id, plugin.name, plugin.icon))

        self.nav_points = nav_points

    def register_callbacks(self) -> None:
        """Register the callbacks for the sidebar layout."""
        # Update navigation items
        output = Output("navigation-items", "children")
        input = Input("on-page-load", "pathname")

        @app.callback(output, input)  # type: ignore
        def update_navigation_items(pathname: str) -> List[Component]:
            """
            Update the navigation items.

            Parameters
            ----------
            pathname : str
                The pathname.

            Returns
            -------
            List[Component]
                The navigation items.
            """
            layouts = []
            for category, points in self.nav_points.items():
                layouts += [
                    html.H6(
                        className="sidebar-heading d-flex justify-content-between "
                        "align-items-center px-3 mt-4 mb-1 text-muted",
                        children=[html.Span(category)],
                    )
                ]

                point_layouts = []
                for id, name, icon in points:
                    href = f"/plugins/{id}"
                    point_layouts += [
                        html.Li(
                            className="nav-item",
                            children=[
                                html.A(
                                    [html.I(className=icon), name],
                                    className=f"nav-link {'active' if href in pathname else ''}",
                                    href=href,
                                )
                            ],
                        )
                    ]

                layouts += [html.Ul(className="nav flex-column", children=point_layouts)]

            return html.Div(
                className="position-sticky pt-3",
                children=[
                    html.Ul(
                        className="nav flex-column",
                        children=[
                            html.A(
                                "General",
                                className=f"nav-link {'active' if pathname == '/' else ''}",
                                href="/",
                            ),
                        ],
                    ),
                    *layouts,
                ],
            )

        # Callback to cancel jobs
        @app.callback(
            # Output('dropdown-container-output', 'children'),
            Input({"type": "cancel-job", "index": ALL}, "n_clicks"),
            State({"type": "cancel-job", "index": ALL}, "name"),
        )
        def delete_job(n_clicks, job_ids):  # type: ignore
            """Delete the job from the queue."""
            for n_click, job_id in zip(n_clicks, job_ids):
                if n_click is not None:
                    queue.delete_job(job_id)

        # Update queue information panel
        output = Output("queue-info", "children")

        @app.callback(output, Trigger("global-update", "n_intervals"))  # type: ignore
        def update_queue_info() -> List[Component]:
            """Update the information of the queue."""
            try:
                all_jobs = [
                    queue.get_finished_jobs(),
                    queue.get_running_jobs(),
                    queue.get_pending_jobs(),
                ]

                job_stati = ["[FINISHED]", "[RUNNING]", "[PENDING]"]

                collect = []
                for jobs, status in zip(all_jobs, job_stati):
                    for job in jobs:
                        name = job.meta["display_name"]
                        job_id = job.id
                        link = job.meta["link"]

                        collect += [(name, job_id, status, link)]

                items = []
                for name, job_id, status, link in collect:
                    items += [
                        html.Li(
                            # className="nav-item",
                            children=[
                                html.A(
                                    f"{status} {name}",
                                    href=link,
                                    style={
                                        "width": "100%",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                    },
                                ),
                                html.Span(
                                    dbc.Button(
                                        "-",
                                        id={"type": "cancel-job", "index": name},
                                        name=job_id,
                                        color="danger",
                                        size="sm",
                                        disabled=True if status == job_stati[0] else False,
                                        style={
                                            "display": "inline-block",
                                            "padding": "0 0.4rem",
                                        },
                                        className="ms-1",
                                    ),
                                    style={
                                        "display": "table-cell",
                                        "vertical-align": "middle",
                                    },
                                ),
                            ],
                            className="nav-link",
                            style={"display": "flex", "flex-direction": "row"},
                        )
                    ]

                if len(collect) > 0:
                    return [
                        html.Hr(),
                        html.H6(
                            className="sidebar-heading d-flex justify-content-between "
                            "align-items-center px-3 mt-4 mb-1 text-muted",
                            children=[html.Span("Queue Information")],
                        ),
                        html.Ul(className="nav flex-column", children=items),
                    ]

                return []
            except Exception:
                return []

    def __call__(self) -> Union[List[Component], Component]:  # noqa: D102
        return html.Nav(
            className="col-md-3 col-lg-2 d-md-block sidebar collapse",
            id="sidebarMenu",
            children=[
                html.Div(id="navigation-items"),
                html.Div(id="queue-info"),
            ],
        )
