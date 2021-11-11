from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple

from dash.dash import no_update
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from src import app, queue, c, rc
from src.runs.handler import handler
from src.utils.logs import get_logger
from src.plugins.plugin import Plugin


logger = get_logger(__name__)


class StaticPlugin(Plugin):
    """
    Calculation with queue. Made for time consuimg tasks.
    """

    def __init__(self):
        self._state = None
        self._refresh_required = True

        super().__init__()

    def register_callbacks(self):
        super().register_callbacks()

        outputs = [
            Output(self.get_internal_id("update-button"), 'n_clicks'),
            Output(self.get_internal_id("processing-info"), 'children')
        ]
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [
            Input(self.get_internal_id("update-button"), 'n_clicks'),
            Input(self.get_internal_id("update-outputs"), 'n_intervals')
        ]

        for id, attribute, _ in self.inputs:
            inputs.append(
                Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_process(n_clicks, n_intervals, *inputs_list):
            """
            Parameters:
                state (int): From button.
                *inputs_list: Values from user.
            """

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            last_inputs = c.get("plugins", self.id(), "last_inputs")

            raw_outputs = self._get_raw_outputs(inputs_key)
            raw_outputs_available = True
            for run_name in handler.get_run_names():
                if raw_outputs[run_name] is None:
                    raw_outputs_available = False

            button_pressed = n_clicks is not None
            inputs_changed = inputs != last_inputs

            # Check current state
            if raw_outputs_available:
                if inputs_changed or self._refresh_required:
                    c.set("plugins", self.id(), "last_inputs", value=inputs)

                    outputs = self._process_raw_outputs(inputs, raw_outputs)
                    self._refresh_required = False

                    return self._update(state=0, outputs=outputs)
                else:
                    return self._update(state=0)

            else:
                if button_pressed and self._state != 2:
                    logger.debug("Button pressed.")

                    # Check if we need to process
                    for run_name, run in handler.get_runs().items():
                        job_id = self._get_job_id(run_name, inputs_key)

                        # We already got our results or it was already processed
                        if raw_outputs[run_name] is not None or queue.is_processed(job_id):
                            continue

                        meta = {
                            "display_name": self.name(),
                            "run_name": run_name,
                            "inputs_key": inputs_key,
                        }

                        logger.debug(f"Enqueued {run_name}.")

                        # Start the task in rq
                        queue.enqueue(
                            self.process,
                            args=[run, inputs],
                            job_id=job_id,
                            meta=meta
                        )

                    # Reset button
                    return self._update(state=2, button_state=None)
                else:
                    # Get finished jobs and save them
                    for job in queue.get_finished_jobs():
                        try:
                            job_id = job.id
                            job_run_outputs = job.result
                            job_meta = job.meta
                            job_inputs_key = job_meta["inputs_key"]
                            job_run_name = job_meta["run_name"]

                            logger.debug(f"Job `{job_id}`")

                            # Save results in cache
                            rc[job_run_name].set(
                                job_inputs_key, value=job_run_outputs)
                            logger.debug(f"... cached")

                            queue.delete_job(job_id)
                            logger.debug(f"... deleted")
                        except:
                            queue.delete_job(job_id)
                            logger.debug(f"... deleted")

                    # Check if queue is still running
                    queue_running = False
                    queue_pending = False
                    for run_name in handler.get_run_names():
                        job_id = self._get_job_id(run_name, inputs_key)
                        if queue.is_running(job_id):
                            queue_running = True

                        if queue.is_pending(job_id):
                            queue_pending = True

                    if queue_running or queue_pending:
                        return self._update(state=2)

                    # Ready again?
                    raw_outputs = self._get_raw_outputs(inputs_key)
                    raw_outputs_available = True
                    for run_name in handler.get_run_names():
                        if raw_outputs[run_name] is None:
                            raw_outputs_available = False

                    if raw_outputs_available:
                        return self._update(state=0)
                    else:
                        return self._update(state=1)

    def _get_raw_outputs(self, inputs_key):
        raw_outputs = {}
        for run_name in handler.get_run_names():
            raw_outputs[run_name] = rc[run_name].get(inputs_key)

        return raw_outputs

    def _update(self, state=None, button_state=None, outputs=None):
        status = ""
        if self._state == 0:
            status = "Ready."
        if self._state == 1:
            status = "Processing Necessary."
        if self._state == 2:
            status = "Processing ..."

        if outputs is None:
            outputs = [no_update for _ in range(len(self.outputs))]

        if state is not None:
            self._state = state

        return [button_state, status] + outputs

    def _get_job_id(self, run_name, inputs_key):
        return run_name + "-" + inputs_key

    def __call__(self):
        self._state = None
        self._refresh_required = True

        components = [
            dcc.Interval(
                id=self.get_internal_id("update-outputs"), interval=500)
        ]
        components += super().__call__(True)

        return components
