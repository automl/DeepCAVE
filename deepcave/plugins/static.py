#  noqa: D400
"""
# Static

This module provides a plugin class for a static plugin.

It provides an Enum used for the plugin state.

## Classes
    - PluginState: An Enum to define the state of the Plugin.
    - StaticPlugin: This class provides a static plugin object.
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple

import traceback
from enum import Enum

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import interactive
from deepcave.plugins import Plugin
from deepcave.runs import AbstractRun
from deepcave.utils.url import create_url


class PluginState(Enum):
    """An Enum to define the state of the Plugin."""

    UNSET = -1
    READY = 0
    NEEDS_PROCESSING = 1
    PROCESSING = 2
    FAILED = 3


def _process(
    process: Callable[[AbstractRun, Any], None], run: AbstractRun, inputs: Dict[str, Any]
) -> None:
    """
    Process the run with the input data if possible.

    Return raw data based on a run and input data.

    Parameters
    ----------
    process : Callable[[AbstractRun, Any], None]
        The process function.
    run : AbstractRun
        The run.
    inputs : Dict[str, Any]
        The inputs as a dictionary.

    Raises
    ------
    Exception
        If the process function fails.
    """
    try:
        return process(run, inputs)
    except Exception:
        traceback.print_exc()
        raise


class StaticPlugin(Plugin, ABC):
    """
    Provide a static plugin object.

    Calculation with queue. Made for time-consuming tasks.
    Register and handle callbacks.

    Properties
    ----------
    outputs : List[Tuple[str, str, bool]]
        The plugin specific outputs.
    inputs : List[Tuple[str, str, bool, Any]]
        The plugin specific inputs.
    id : str
        The plugin id.
    raw_outputs : dict[str, Any]
        The raw outputs of a run.
    logger : Logger
        The logger for the plugin.
    name : str
        The name of the plugin.
    process : Callable
        Return raw data based on a run and input data.
    button_caption : str
        The caption for the button.
    """

    def __init__(self) -> None:
        super().__init__()
        self._setup()

    def _setup(self) -> None:
        """Set up the plugin."""
        self._state = PluginState.UNSET  # Set in the main loop to track what's going on right now
        self._previous_state: Optional[PluginState] = None  # Used for updating status
        self._refresh_required = True
        self._reset_button = False
        self._blocked = False

    @interactive
    def register_callbacks(self) -> None:
        """Register different callbacks."""
        super().register_callbacks()
        self._callback_inputs_changed()
        self._callback_loop_update_status_label()
        self._callback_loop_trigger_main_loop()

    @interactive
    def _callback_inputs_changed(self) -> None:
        """Handle callback if the inputs changed."""
        from deepcave import app, c, queue, rc, run_handler

        # Plugin specific outputs
        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [
            Input(self.get_internal_id("update-button"), "n_clicks"),
            Input(self.get_internal_id("update-interval-output"), "data"),
        ]

        # Get other plugin specific inputs that might change
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)  # type: ignore
        def plugin_process(n_clicks: int, _: Any, *inputs_list: str) -> Optional[Any]:
            """Register updates from inputs."""
            self._blocked = True
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(list(inputs_list), input=True)
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            cleaned_inputs = self._clean_inputs(inputs)
            last_inputs = c.get("last_inputs", self.id)

            link = create_url(self.get_base_url(), inputs)
            runs = self.get_selected_runs(inputs)

            button_pressed = n_clicks is not None
            inputs_changed = inputs != last_inputs

            # Check current state
            raw_outputs = {}
            raw_outputs_available = True
            for run in runs:
                raw_outputs[run.id] = rc.get(run, self.id, inputs_key)

                if raw_outputs[run.id] is None:
                    raw_outputs_available = False

            # Process
            if raw_outputs_available:
                # Load raw outputs from cache
                self._state = PluginState.READY

                if inputs_changed or self._refresh_required:
                    c.set("last_inputs", self.id, value=inputs)

                    # Save for modal
                    self.raw_outputs = raw_outputs

                    outputs = self._process_raw_outputs(inputs, raw_outputs)
                    self._refresh_required = False

                    return outputs
            else:
                # Load from process
                self._state = PluginState.NEEDS_PROCESSING

                if button_pressed and self._state != PluginState.PROCESSING:
                    self.logger.debug("Button pressed.")

                    # Check if processing is needed
                    for run in runs:
                        job_id = self._get_job_id(run.id, inputs_key)

                        # Results are already achieved or it was already processed
                        if raw_outputs[run.id] is not None or queue.is_processed(job_id):
                            continue

                        job_meta = {
                            "display_name": self.name,
                            "run_name": run.name,
                            "run_id": run.id,
                            "plugin_id": self.id,
                            "inputs_key": inputs_key,
                            "link": link,
                        }

                        self.logger.debug(f"Enqueued {run.name} ({run.id}).")

                        # Start the task in rq
                        queue.enqueue(
                            _process,
                            args=[self.process, run, cleaned_inputs],
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
                            job_run_id = job_meta["run_id"]
                            job_plugin_id = job_meta["plugin_id"]

                            self.logger.debug(f"Job {job_id} for run_id {job_meta['run_id']}.")
                            run = run_handler.get_run(job_run_id)

                            # Save results in cache
                            # Same optional string problem
                            rc.set(run, job_plugin_id, job_inputs_key, job_run_outputs)
                            self.logger.debug(f"Job {job_id} cached.")

                            queue.delete_job(job_id)
                            self.logger.debug(f"Job {job_id} deleted.")
                        except Exception as e:
                            self.logger.error(f"Job {job_id} failed with exception {e}.")
                            queue.delete_job(job_id)
                            self.logger.debug(f"Job {job_id} deleted.")
                        except KeyboardInterrupt:
                            self.logger.error(f"Job {job_id} got interrupted by KeyboardInterrupt.")
                            queue.delete_job(job_id)
                            self.logger.debug(f"Job {job_id} deleted.")
                            raise

                    # Check if queue is still running
                    queue_running = False
                    queue_pending = False
                    for run in runs:
                        job_id = self._get_job_id(run.id, inputs_key)
                        if queue.is_running(job_id):
                            queue_running = True

                        if queue.is_pending(job_id):
                            queue_pending = True

                        if queue.has_failed(job_id):
                            self._state = PluginState.FAILED

                            # Delete it after it was detected
                            queue.delete_job(job_id)

                    if queue_running or queue_pending:
                        self._state = PluginState.PROCESSING

            self._blocked = False
            raise PreventUpdate

    @interactive
    def _callback_loop_trigger_main_loop(self) -> None:
        from deepcave import app

        output = Output(self.get_internal_id("update-interval-output"), "data")
        inputs = [
            Input(self.get_internal_id("update-interval"), "n_intervals"),
            State(self.get_internal_id("update-interval-output"), "data"),
        ]

        # Interval should not always run the main callback the whole time
        # Especially not if it's blocked because PreventUpdate
        # prevent output updates from previous callback calls.
        @app.callback(output, inputs)  # type: ignore
        def plugin_check_blocked(_: Any, data: Any) -> Any:
            """
            Check if blocked.

            Raises
            ------
            PreventUpdate
                If '_blocked' is True.
            """
            if self._blocked:
                raise PreventUpdate

            # This will trigger the main loop
            return data + 1

    @interactive
    def _callback_loop_update_status_label(self) -> None:
        from deepcave import app, notification

        output = [
            Output(self.get_internal_id("update-button"), "children"),
            Output(self.get_internal_id("update-button"), "n_clicks"),
            Output(self.get_internal_id("update-button"), "disabled"),
        ]
        input = Input(self.get_internal_id("update-interval"), "n_intervals")

        # Update status label
        # Register updates from inputs
        @app.callback(output, input)  # type: ignore
        def plugin_update_status(_: Any) -> Tuple[List[Any], Optional[Any], bool]:
            """Update the status of the plugin."""
            button_text = [html.Span(self.button_caption)]

            if self._state == PluginState.UNSET:
                # Disable and reset button
                return button_text, None, True

            # Important so the button is not updated every time (would result in an ugly spinner)
            if self._previous_state == self._state:
                raise PreventUpdate

            # This is a special case where the main loop goes into "needs processing"
            # although the result is already there. This is because the queue needs a second
            # to be updated.
            if (
                self._previous_state == PluginState.PROCESSING
                and self._state == PluginState.NEEDS_PROCESSING
            ):
                # However: The previous state has to be unset, so if the inputs are really changed
                # the visualizer will be updated.
                self._previous_state = PluginState.UNSET
                raise PreventUpdate

            if self._state == PluginState.FAILED:
                notification.update(
                    "The job failed. Check the logs or make sure the worker is still running. "
                    "Most of the times, a simple restart might help."
                )

            if self._state == PluginState.READY:
                disabled = True
            elif self._state == PluginState.PROCESSING:
                button_text += [dbc.Spinner(size="sm", spinner_class_name="ms-2")]
                disabled = True
            else:
                disabled = False

            button = no_update
            if self._reset_button:
                self._reset_button = False
                button = None

            # Update the previous state
            self._previous_state = self._state

            return button_text, button, disabled

    def _get_job_id(self, run_name: str, inputs_key: str) -> str:
        """
        Get the job id.

        Parameters
        ----------
        run_name : str
            The name of the run.
        inputs_key : str
            The inputs key.

        Returns
        -------
        str
            The job id.
        """
        return f"{run_name}-{inputs_key}"

    @interactive
    def __call__(self) -> List[Component]:  # type: ignore
        """
        Return the components for the plugin.

        Basically, all blocks and elements of the plugin are stacked-up here.

        Returns
        -------
        List[Component]
            Layout as list of components.
        """
        from deepcave import config

        self._setup()

        components = [
            dcc.Interval(id=self.get_internal_id("update-interval"), interval=config.REFRESH_RATE),
            dcc.Store(id=self.get_internal_id("update-interval-output"), data=0),
        ]
        components += super().__call__(True)

        return components
