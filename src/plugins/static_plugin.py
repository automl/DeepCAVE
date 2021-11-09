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
    Behaviour:

    A On page load:
    A1: last_inputs/last_outputs are available? Show them.
    A2: last_inputs/last_outputs not available? Leave everything as it is.

    B On filter+input change:
    B1: Don't do anything if button pressed.
    B1: Reset input
    B2: Print error message that input was reset.

    On input change:
    2.1: Don't do anything.

    On press button:
    3.1: Take input and process it if
    3.1.1: Input is not in the queue.
    3.1.2: Input is not found in the cache.
    3.2: Update output based on new results.



    On filter change:
    4.1: Update output without processing.

    Check for new results
    """

    def __init__(self):
        # How often the button was pressed.
        # It's important because we can't use `State` attributes
        # because of input dependencies.
        self._n_clicks = None

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

            button_pressed = False
            if n_clicks is not None and n_clicks != self._n_clicks:
                button_pressed = True

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            raw_outputs = self._get_raw_outputs(inputs_key)
            raw_outputs_available = True
            for run_name in handler.get_run_names():
                if raw_outputs[run_name] is None:
                    raw_outputs_available = False

            last_inputs = c.get("plugins", self.id(), "last_inputs")

            if button_pressed:
                logger.debug("Button pressed.")

                # Update state
                self._n_clicks = n_clicks

                if self.debug():
                    logger.debug("Debug mode: Reset raw outputs")
                    raw_outputs = {}

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

                return self._update_status("")

            # Now the interval comes in
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

            if queue_running:
                return self._update_status("Processing ...")
            elif queue_pending:
                return self._update_status("Pending ...")
            elif raw_outputs_available:
                if last_inputs == inputs and n_clicks is not None:
                    return self._update_status("Ready.")

                # Set new inputs as last_inputs
                c.set("plugins", self.id(), "last_inputs", value=inputs)

                return [0, ""] + self._process_raw_outputs(inputs, raw_outputs)
            else:
                return self._update_status("")

    def _get_raw_outputs(self, inputs_key):
        raw_outputs = {}
        for run_name in handler.get_run_names():
            raw_outputs[run_name] = rc[run_name].get(inputs_key)

        return raw_outputs

    def _update_status(self, status):
        return [no_update, status] + [no_update for _ in range(len(self.outputs))]

    def _get_job_id(self, run_name, inputs_key):
        return run_name + "-" + inputs_key

    def __call__(self):
        components = [
            # dcc.Store(id=self.get_internal_id("update-outputs")),
            dcc.Interval(id=self.get_internal_id(
                "update-outputs"), interval=500)
        ]
        components += super().__call__(True)

        return components
