from abc import ABC
from typing import List

from dash.dependencies import Input, Output
from dash.development.base_component import Component

from deepcave import interactive
from deepcave.plugins import Plugin


class DynamicPlugin(Plugin, ABC):
    use_cache = True

    def __init__(self) -> None:
        super().__init__()

    @interactive
    def register_callbacks(self) -> None:
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
            Parameters:
                *inputs_list: Values from user.
            """
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)
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
    def __call__(self) -> List[Component]:  # type: ignore
        return super().__call__(False)
