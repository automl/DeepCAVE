from abc import ABC

from enum import Enum

from dash import dcc
from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from deepcave import app, c, queue, rc
from deepcave.plugins import Plugin


class PluginState(Enum):
    UNSET = -1
    READY = 0
    NEEDS_PROCESSING = 1
    PROCESSING = 2


class StaticPlugin(Plugin, ABC):
    """
    Calculation with queue. Made for time-consuming tasks.
    """

    def __init__(self):
        self._state: PluginState = PluginState.UNSET
        self._refresh_required = True

        super().__init__()

    def register_callbacks(self):
        super().register_callbacks()
        self._callback_inputs_changed()
        self._callback_loop_update_status_label()
        self._callback_loop_trigger_main_loop()

    def _callback_inputs_changed(self):
        # Plugin specific outputs
        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [
            Input(self.get_internal_id("update-button"), "n_clicks"),
            Input(self.get_internal_id("update-interval-output"), "data"),
        ]

        # Get other plugin specific inputs that might change
        for id, attribute, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_process(n_clicks, _, *inputs_list):
            """
            Parameters:
                n_clicks (int): From button.
                *inputs_list: Values from user.
            """

            self._blocked = True

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            last_inputs = c.get("last_inputs", self.id)

            runs = self.get_selected_runs(inputs)

            button_pressed = n_clicks is not None
            inputs_changed = inputs != last_inputs

            # Check current state
            raw_outputs = {}
            raw_outputs_available = True
            for run in runs:
                raw_outputs[run.name] = rc.get_run(run).get(self.id, inputs_key)

                if raw_outputs[run.name] is None:
                    raw_outputs_available = False

            # Process
            if raw_outputs_available:
                # Load raw outputs from cache
                self._state = PluginState.READY

                if inputs_changed or self._refresh_required:
                    c.set("last_inputs", self.id, value=inputs)

                    outputs = self._process_raw_outputs(inputs, raw_outputs)
                    self._refresh_required = False

                    return outputs
            else:
                # Load from process
                self._state = 1

                if button_pressed and self._state != PluginState.PROCESSING:
                    self.logger.debug("Button pressed.")

                    # Check if we need to process
                    for run in runs:
                        job_id = self._get_job_id(run.name, inputs_key)

                        # We already got our results or it was already processed
                        if raw_outputs[run.name] is not None or queue.is_processed(
                            job_id
                        ):
                            continue

                        job_meta = {
                            "display_name": self.name,
                            "run_name": run.name,
                            "run_cache_id": run.run_cache_id,
                            "inputs_key": inputs_key,
                        }

                        self.logger.debug(f"Enqueued {run.name}.")

                        # Start the task in rq
                        queue.enqueue(
                            self._process,
                            args=[self.process, run.run_cache_id, inputs],
                            job_id=job_id,
                            meta=job_meta,
                        )

                    # Reset button
                    self._reset_button = True
                    self._state = PluginState.PROCESSING
                else:
                    # Get finished jobs and save them
                    for job in queue.get_finished_jobs():
                        job_id = job.id
                        try:
                            job_run_outputs = job.result
                            job_meta = job.meta
                            job_inputs_key = job_meta["inputs_key"]
                            job_run_name = job_meta["run_cache_id"]

                            self.logger.debug(f"Job `{job_id}`")

                            # Save results in cache
                            rc.get(job_run_name).set(
                                self.id, job_inputs_key, value=job_run_outputs
                            )
                            self.logger.debug(f"... cached")

                            queue.delete_job(job_id)
                            self.logger.debug(f"... deleted")
                        except:
                            queue.delete_job(job_id)
                            self.logger.debug(f"... deleted")

                    # Check if queue is still running
                    queue_running = False
                    queue_pending = False
                    for run in runs:
                        job_id = self._get_job_id(run.name, inputs_key)
                        if queue.is_running(job_id):
                            queue_running = True

                        if queue.is_pending(job_id):
                            queue_pending = True

                    if queue_running or queue_pending:
                        self._state = PluginState.PROCESSING

            self._blocked = False
            raise PreventUpdate

    def _callback_loop_trigger_main_loop(self):
        output = Output(self.get_internal_id("update-interval-output"), "data")
        inputs = [
            Input(self.get_internal_id("update-interval"), "n_intervals"),
            State(self.get_internal_id("update-interval-output"), "data"),
        ]

        # Interval should not always run the main callback the whole time
        # Especially not if it's blocked because PreventUpdate
        # prevent output updates from previous callback calls.
        @app.callback(output, inputs)
        def plugin_check_blocked(_, data):
            if self._blocked:
                raise PreventUpdate

            # This will trigger the main loop
            return data + 1

    def _callback_loop_update_status_label(self):
        output = [
            Output(self.get_internal_id("processing-info"), "children"),
            Output(self.get_internal_id("update-button"), "n_clicks"),
        ]
        input = Input(self.get_internal_id("update-interval"), "n_intervals")

        # Update status label
        # Register updates from inputs
        @app.callback(output, input)
        def plugin_update_status(_):
            status_str = ""
            if self._state == PluginState.READY:
                status_str = "Ready."
            elif self._state == PluginState.NEEDS_PROCESSING:
                status_str = "Processing Necessary."
            elif self._state == PluginState.PROCESSING:
                status_str = "Processing ..."

            button = no_update
            if self._reset_button:
                self._reset_button = False
                button = None

            return status_str, button

    def _get_job_id(self, run_name, inputs_key) -> str:
        return f"{run_name}-{inputs_key}"

    def __call__(self):
        self._state = 1
        self._refresh_required = True
        self._reset_button = False
        self._blocked = False

        components = [
            dcc.Interval(id=self.get_internal_id("update-interval"), interval=200),
            dcc.Store(id=self.get_internal_id("update-interval-output"), data=0),
        ]
        components += super().__call__(True)

        return components
