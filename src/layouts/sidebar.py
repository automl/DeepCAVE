import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc

from src import app, queue
from src.layouts.layout import Layout
from src.plugins import plugin_names, plugin_categories


class SidebarLayout(Layout):

    def __init__(self):
        super().__init__()

        nav_points = {}
        for id, name in plugin_names.items():

            category = plugin_categories[id]

            if category not in nav_points:
                nav_points[category] = []

            nav_points[category].append((id, name))

        self.nav_points = nav_points

    def register_callbacks(self):
        # Update queue information panel
        @app.callback(
            Output("queue-info", 'children'),
            Input("queue-info-interval", 'n_intervals'))
        def update_queue_info(_):
            try:
                # queue.get_finished_jobs()

                jobs = {}
                for job in queue.get_running_jobs():
                    display_name = job.meta['display_name']
                    run_name = job.meta['run_name']

                    if display_name not in jobs:
                        jobs[display_name] = []

                    jobs[display_name].append((run_name, "[R]"))

                for job in queue.get_pending_jobs():
                    display_name = job.meta['display_name']
                    run_name = job.meta['run_name']

                    if display_name not in jobs:
                        jobs[display_name] = []

                    jobs[display_name].append((run_name, "[P]"))

                items = []
                for display_name, run_names in jobs.items():
                    items += [html.Li(
                        className='nav-item',
                        children=[
                            html.A(f"{display_name}",
                                   className='nav-link disabled')
                        ]
                    )]

                    for run_name, status in run_names:
                        items += [
                            html.Li(
                                className='nav-item',
                                children=[
                                    html.A(f"{status} {run_name}",
                                           className='nav-link disabled')
                                ]
                            )
                        ]

                #reg = queue.finished_job_registry
                # for job_id in reg.get_job_ids():
                #    layouts += [html.Div(f"[FINISHED] {job_id[:10]}")]

                if len(jobs) > 0:
                    return [
                        html.Hr(),
                        html.H6(className='sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted', children=[
                            html.Span("Queue Information")
                        ]),
                        html.Ul(className='nav flex-column', children=items),
                    ]

                return []
            except:
                return

    def __call__(self):

        layouts = []
        for category, points in self.nav_points.items():
            layouts += [
                html.H6(className='sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted', children=[
                    html.Span(category)
                ])
            ]

            point_layouts = []
            for (id, name) in points:
                point_layouts += [html.Li(
                    className='nav-item',
                    children=[html.A(name, className='nav-link active', href=f'/plugins/{id}')])
                ]

            layouts += [html.Ul(className='nav flex-column',
                                children=point_layouts)]

        html.Li(
            className='nav-item',
            children=[html.A(name, className='nav-link active', href=f'/plugins/{id}')])

        return \
            html.Nav(className='col-md-3 col-lg-2 d-md-block sidebar collapse', id='sidebarMenu', children=[
                html.Div(className='position-sticky pt-3', children=[
                    html.Ul(className='nav flex-column', children=[
                        html.A("General", className='nav-link active', href='/'),
                    ]),

                    *layouts

                ]),

                dcc.Interval(id="queue-info-interval", interval=1000),
                html.Div(id="queue-info")
            ])


layout = SidebarLayout()
