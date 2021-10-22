from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple

from dash.dash import no_update
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deep_cave import app, queue, cache
from deep_cave.runs.handler import handler
from deep_cave.utils.logs import get_logger
from deep_cave.plugins.plugin import Plugin


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
            raw_outputs = cache.get(self.id(), inputs_key)

            last_inputs = cache.get(self.id(), "last_inputs")
            # last_inputs_key = self._dict_as_key(last_inputs, remove_filters=True)
            #last_raw_outputs = cache.get(self.id(), last_inputs_key)

            # Check if filter changed

            # Set new inputs as last_inputs
            cache.set(self.id(), "last_inputs", value=inputs)

            if button_pressed:
                # Update state
                self._n_clicks = n_clicks

                logger.debug("Button pressed.")

                if self.debug():
                    logger.debug("Debug mode: Reset raw outputs")
                    raw_outputs = None

                # Check if we need to process
                if raw_outputs is None and not queue.is_processed(inputs_key):

                    meta = {
                        "display_name": self.name(),
                        "keys": [self.id(), inputs_key]
                    }

                    logger.debug("Enqueued task.")

                    # Start the task in rq
                    queue.enqueue(
                        self.process,
                        args=[handler.get_run(), inputs],
                        job_id=inputs_key,
                        meta=meta
                    )

                    return self._update_status("Added task to queue.")
                else:
                    logger.debug("No process. Found task in cache.")
                    return self._update_status("Processing ...")

            # Now the interval comes in
            else:

                # Get finished jobs and save them
                for job in queue.get_finished_jobs():

                    try:
                        job_id = job.id
                        job_raw_outputs = job.result
                        meta = job.meta
                        job_inputs_key = meta["keys"][-1]

                        logger.debug(f"Job `{job_id}`")

                        # Save results in cache
                        cache.set(*meta["keys"], value=job_raw_outputs)
                        logger.debug(f"... cached")

                        queue.delete_job(job_id)
                        logger.debug(f"... deleted")

                        # Use the results directly
                        if job_inputs_key == inputs_key:
                            return [n_clicks, "Ready."] + self._process_raw_outputs(inputs, job_raw_outputs)
                    except:
                        queue.delete_job(job_id)
                        logger.debug(f"... deleted")

                if n_clicks is None and raw_outputs is not None:
                    return [0, "Ready."] + self._process_raw_outputs(inputs, raw_outputs)

                # Since we are updating every x seconds, prevent of
                # rewriting outputs over and over again.
                # Don't update if the inputs are the same.
                if inputs == last_inputs and raw_outputs is not None:
                    return self._update_status("Ready.")

                if raw_outputs is not None:
                    return [n_clicks, "Ready."] + self._process_raw_outputs(inputs, raw_outputs)

            if queue.is_running(inputs_key):
                return self._update_status("Processing ...")
            elif queue.is_pending(inputs_key):
                return self._update_status("Pending ...")
            else:
                return self._update_status("Not processed yet.")

    def _update_status(self, status):
        return [no_update, status] + [no_update for _ in range(len(self.outputs))]

    def __call__(self):
        components = [
            # dcc.Store(id=self.get_internal_id("update-outputs")),
            dcc.Interval(id=self.get_internal_id(
                "update-outputs"), interval=500)
        ]
        components += super().__call__(True)

        return components
