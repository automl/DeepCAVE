import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from deep_cave.runs.converters import available_converters
from deep_cave import app
from deep_cave.cache import cache
from deep_cave.layouts.layout import Layout
from deep_cave.runs import get_run_ids


class GeneralLayout(Layout):
    def register_callbacks(self):
        outputs = [
            Output('general-working-directory-input', 'value'),
            Output('general-converter-select', 'value'),
            Output('general-runs-radiolist', 'options'),
            Output('general-runs-radiolist', 'value'),
            Output('general-alert', 'is_open'),
            Output('general-alert', 'children'),
        ]

        inputs = [
            Input('on-page-load', 'href'),
            Input('general-update-button', 'n_clicks'),
            State('general-working-directory-input', 'value'),
            State('general-converter-select', 'value')
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def general_update(_, n_clicks, working_dir, converter_name):
            alert_open = False
            alert_message = None

            if isinstance(n_clicks, int) and n_clicks > 0:
                cache.clear()
                cache.set("working_dir", working_dir)
                cache.set("converter_name", converter_name)
                cache.set("run_id", "")

                alert_open = True
                alert_message = "Successfully updated meta data."
            
            return \
                cache.get("working_dir"), \
                cache.get("converter_name"), \
                self.get_run_options(), \
                cache.get("run_id"), \
                alert_open, \
                alert_message

        input = Input('general-runs-radiolist', 'value')
        output = Output('general-runs-output', 'value')

        # Save the run ids internally
        # We have to inform the other plugins here as well
        @app.callback(output, input)
        def general_register_runs(run_id):
            if cache.get("run_id") != run_id:
                working_dir = cache.get("working_dir")
                converter_name = cache.get("converter_name")

                # Clear cache
                cache.clear()

                # Set everything
                cache.set("working_dir", working_dir)
                cache.set("converter_name", converter_name)
                cache.set("run_id", run_id)

                return run_id
            
            raise PreventUpdate()

    @staticmethod
    def get_converter_options():
        return [{"label": adapter, "value": adapter} for adapter in available_converters.keys()]

    @staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in get_run_ids()]

    def __call__(self):
        return [
            html.H1('General'),

            dbc.Alert("", color="success", id="general-alert", is_open=False, dismissable=True),

            dbc.FormGroup([
                dbc.Label("Working Directory", html_for="general-working-directory-input"),
                dbc.FormText("Absolute path to your studies."),
                dbc.Input(id="general-working-directory-input", placeholder="", type="text"),
            ]),

            dbc.FormGroup([
                dbc.Label("Converter", html_for="general-converter-select"),
                dbc.FormText("Which optimizer was used to receive the data?"),
                dbc.Select(
                    id="general-converter-select",
                    options=GeneralLayout.get_converter_options(),
                    placeholder="Select converter ..."
                ),
            ]),

            dbc.Button("Update", id="general-update-button", color="primary"),
            html.Hr(),

            html.H2('Runs'),
            dbc.Input(id="general-runs-output", style={"display": "none"}),
            dbc.RadioItems(id="general-runs-radiolist")
        ]


layout = GeneralLayout()