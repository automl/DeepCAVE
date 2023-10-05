#  noqa: D400
"""
# Dynamic

This module provides a plugin class for a dynamic plugin.

Registers and handles callbacks.

## Classes
    - DynamicPlugin: This class provides a dynamic plugin object.
"""

from abc import ABC
from typing import List

from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import interactive
from deepcave.plugins import Plugin


class DynamicPlugin(Plugin, ABC):
    """
    Provide a dynamic plugin object.

    Registers and handles callbacks.

    Properties
    ----------
    outputs : List[Tuple[str, str, bool]]
        The registered outputs.
    inputs : List[Tuple[str, str, bool, Any]]
        The registered inputs.
    id : str
        The identifier of the Plugin.
    logger : Logger
        The logger for debugging information.
    use_cache : bool
        Defines whether to use the cache.
    raw_outputs : Dict[str, Any]
        A dictionary of the raw outputs.
    """

    use_cache = True

    def __init__(self) -> None:
        super().__init__()

    @interactive
    def register_callbacks(self) -> None:
        """Register the callbacks and update the outputs from the inputs."""
        super().register_callbacks()
        from deepcave import app, c, rc

        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), "n_clicks")]
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_output_update(_, *inputs_list):  # type: ignore
            """
            Update the outputs from the inputs.

            Parameters
            ----------
            *inputs_list
                Values from user.

            Returns
            -------
            The raw outputs.
            """
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(list(inputs_list), input=True)
            # We would need to change the Optional[str] to a str as return type annotation
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            cleaned_inputs = self._clean_inputs(inputs)
            runs = self.get_selected_runs(inputs)

            raw_outputs = {}
            for run in runs:
                run_outputs = rc.get(run, self.id, inputs_key)
                if run_outputs is None:
                    self.logger.debug(f"Process {run.name}.")
                    run_outputs = self.process(run, cleaned_inputs)

                    # Cache it
                    if self.use_cache:
                        rc.set(run, self.id, inputs_key, value=run_outputs)
                else:
                    self.logger.debug(f"Found outputs from {run.name} in cache.")

                raw_outputs[run.id] = run_outputs

            # Save for modal
            self.raw_outputs = raw_outputs

            # Cache last inputs
            c.set("last_inputs", self.id, value=inputs)

            return self._process_raw_outputs(inputs, raw_outputs)

    @interactive
    def __call__(self) -> List[Component]:  # type: ignore # noqa: D102
        return super().__call__(False)
