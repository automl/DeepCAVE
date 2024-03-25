#  noqa: D400
"""
# Dynamic

This module provides a plugin class for a dynamic plugin.

Register and handle callbacks.

## Classes
    - DynamicPlugin: This class provides a dynamic plugin object.
"""

from abc import ABC
from typing import Any, List

from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import interactive
from deepcave.plugins import Plugin


class DynamicPlugin(Plugin, ABC):
    """
    Provide a dynamic plugin object.

    Register and handle callbacks.

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
        """
        Register basic callbacks for the plugin.

        Following callbacks are registered:
        - If inputs changes, the changes are pasted back.
          This is in particular interest if input dependencies are used.
        - Raw data dialog to display raw data.
        - Callback to be redirected to the config if clicked on it.
        """
        super().register_callbacks()
        from deepcave import app, c, rc

        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), "n_clicks")]
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)  # type: ignore
        def plugin_output_update(_: Any, *inputs_list: str) -> Any:
            """
            Update the outputs.

            Parameters
            ----------
            *inputs_list
                Input values from user.

            Returns
            -------
            Any
                The raw outputs.
            """
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(list(inputs_list), input=True)
            # The Optional[str] would need tobe changed to a str as return type annotation
            # of the function _dict_as_key.
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            cleaned_inputs = self._clean_inputs(inputs)
            runs = self.get_selected_runs(inputs)

            raw_outputs = {}
            rc.clear()
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
    def __call__(self) -> List[Component]:  # type: ignore
        """
        Return the components for the plugin.

        Basically, all blocks and elements of the plugin are stacked-up here.

        Returns
        -------
        List[Component]
            Layout as list of components.
        """
        return super().__call__(False)
