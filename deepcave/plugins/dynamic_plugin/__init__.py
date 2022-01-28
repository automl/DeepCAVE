from abc import ABC

from dash.dependencies import Input, Output

from deepcave import app, c, rc
from deepcave.plugins import Plugin


class DynamicPlugin(Plugin, ABC):
    def __init__(self):
        super().__init__()

    def register_callbacks(self):
        super().register_callbacks()
        self._callback_inputs_changed()

    def _callback_inputs_changed(self):
        outputs = []
        for id, attribute, _ in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), "n_clicks")]
        for id, attribute, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def plugin_output_update(_, *inputs_list):
            """
            Parameters:
                *inputs_list: Values from user.
            """

            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(inputs_list, input=True)
            inputs_key = self._dict_as_key(inputs, remove_filters=True)

            runs = self.get_selected_runs(inputs)

            raw_outputs = {}
            for run in runs:
                run_cache = rc.get_run(run)
                run_outputs = run_cache.get(self.id, inputs_key)
                if run_outputs is None:
                    self.logger.debug(f"Process {run.name}.")
                    run_outputs = self.process(run, inputs)

                    # Here's the thing:
                    # We have to remove `run_name` from the inputs completely

                    # Cache it
                    run_cache.set(self.id, inputs_key, value=run_outputs)
                else:
                    self.logger.debug(f"Found outputs from {run.name} in cache.")

                raw_outputs[run.name] = run_outputs

            # Cache last inputs
            c.set("last_inputs", self.id, value=inputs)

            return self._process_raw_outputs(inputs, raw_outputs)

    def __call__(self):
        return super().__call__(False)
