import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from deep_cave.converter import converters
from deep_cave.server import app
from deep_cave.data_manager import dm
from deep_cave.run_manager import rm
from deep_cave.layouts.layout import Layout


class GeneralLayout(Layout):
    def _register_callbacks(self):
        outputs = [
            Output('general-working-directory-input', 'value'),
            Output('general-converter-select', 'value'),
            Output('general-runs-checklist', 'options'),
            Output('general-runs-checklist', 'value'),
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
                dm.set("working_dir", working_dir)
                dm.set("converter", converter_name)
                dm.set("run_ids", [])

                alert_open = True
                alert_message = "Successfully updated meta data."
            
            return \
                GeneralLayout.get_working_dir(), \
                GeneralLayout.get_converter(), \
                GeneralLayout.get_run_options(), \
                GeneralLayout.get_run_ids(), \
                alert_open, \
                alert_message

        input = Input('general-runs-checklist', 'value')
        output = Output('general-runs-output', 'value')

        # Save the run ids internally
        # We have to inform the other plugins here as well
        @app.callback(output, input)
        def general_register_runs(run_ids):
            dm.set("run_ids", run_ids)

            return run_ids

    @staticmethod
    def get_converter_options():
        return [{"label": adapter, "value": adapter} for adapter in converters.keys()]

    @staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in rm.get_run_names()]

    @staticmethod
    def get_run_ids():
        run_ids = dm.get("run_ids")
        if run_ids is None:
            return []

        return run_ids

    @staticmethod
    def get_working_dir():
        return dm.get("working_dir")

    @staticmethod
    def get_converter():
        return dm.get("converter")

    def _get_layout(self):
        return [
            html.H1('General'),

            dbc.Alert("", color="success", id="general-alert", is_open=False, dismissable=True),

            html.Div("Working Directory"),
            html.Div(html.I("Absolute path to your studies.")),
            dbc.Input(id="general-working-directory-input", placeholder="", type="text", 
                #value=GeneralLayout.get_working_dir()
            ),

            html.Div("Converter"),
            html.Div(html.I("Which optimizer was used to receive the data?")),
            dbc.Select(
                id="general-converter-select",
                options=GeneralLayout.get_converter_options(),
                #value=GeneralLayout.get_converter(),
            ),

            dbc.Button("Update", id="general-update-button", color="primary", className="mt-3"),
            html.Hr(),

            html.H2('Runs'),
            dbc.Input(id="general-runs-output", style="display: none;"),
            dbc.Checklist(
                id="general-runs-checklist",
                #options=GeneralLayout.get_run_options(),
                #value=GeneralLayout.get_run_ids()
            )
        ]