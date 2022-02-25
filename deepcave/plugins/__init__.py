from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Union, List

import copy

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import app, c, run_handler
from deepcave.layouts import Layout
from deepcave.runs import AbstractRun
from deepcave.runs.grouped_run import GroupedRun, NotMergeableError
from deepcave.runs.run import Run
from deepcave.utils.data_structures import update_dict
from deepcave.utils.hash import string_to_hash
from deepcave.utils.layout import get_select_options
from deepcave.utils.logs import get_logger
from deepcave.utils.util import add_prefix_to_dict

logger = get_logger(__name__)


class Plugin(Layout, ABC):
    """
    Base class for all plugins.

    Attributes
    ----------
    id : int
        Unique identifier for the plugin.
    name : str
        Name of the plugin. It is shown in the navigation and in the title.
    description : str, optional
        Description of the plugin. Displayed below the title.
    icon : str, optional
        FontAwesome icon. Shown in the navigation.
    button_caption : str, optional
        Caption of the button. Shown only, if `StaticPlugin` is used.
    activate_run_selection : bool, optional
        Shows a dropdown to select a run in the inputs layout.
        This feature is useful if only one run could be viewed at a time.
        Moreover, it prevents the plugin to calculate results across all runs.
    """

    id: str
    name: str
    description: Optional[str] = None
    icon: str = "far fa-file"
    button_caption: str = "Process"
    activate_run_selection: bool = False

    def __init__(self) -> None:
        self.inputs = []
        self.outputs = []

        # Processing right now?
        self.blocked = False

        # Alert texts
        self.alert_text = ""
        self.alert_color = "success"
        self.alert_update_required = False

        self.runs: Dict[str, AbstractRun] = {}  # Set in __call__: run_name -> AbstractRun

        super().__init__()

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """
        Checks if a run is compatible with this plugin. If a plugin is not compatible,
        you can not select the run.

        Notes
        -----
        This function is only called if `activate_run_selection` is True.

        Parameters
        ----------
        run : AbstractRun
            One of the selected runs/groups.

        Returns
        -------
        bool
            Returns True if the run is compatible.
        """

        return True

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """
        This function is needed if all selected runs need something in common
        (e.g. budget or objective). Since this function is called before the layout is created,
        it can be also used to set common values for the plugin.

        Parameters
        ----------
        runs : List[AbstractRun]
            Selected runs.

        Raises
        ------
        NotMergeableError
            If runs are not compatible, an error is thrown.
        """

        return True

    def register_input(
        self, id: str, attributes: Union[str, Iterable[str]] = ("value",), filter=False
    ) -> str:
        """
        Registers an input variable for the plugin. It is important to register the inputs
        because callbacks have to be defined before the server is started.
        After registering all inputs, an internal mapping is created.

        Parameters
        ----------
        id : str
            Specifies the id of the input.
        attributes : Union[str, Iterable[str]], optional
            Attributes which should be passed to the (dash) component, by default ("value",)
        filter : bool, optional
            Specifies if the input is a filter, by default False

        Returns
        -------
        str
            Unique id for the input and plugin. This is necessary because ids are defined globally.
        """

        if isinstance(attributes, str):
            attributes = [attributes]

        for attribute in attributes:
            key = (id, attribute, filter)
            if key not in self.inputs:
                self.inputs.append(key)

        # We have to rearrange the inputs because `State`
        # must follow all `Input`. Since all filters are `Input`, we have to
        # shift them to the front.
        self.inputs.sort(key=lambda x: x[2], reverse=True)

        return self.get_internal_input_id(id)

    def register_output(self, id: str, attribute: str = "value", mpl=False) -> str:
        """
        Registers an output variable for the plugin.

        Parameters
        ----------
        id : str
            Specifies the id of the output.
        attribute : str, optional
            Attribute, by default "value"
        mpl : bool, optional
            Specifies if the registration is for matplotlib or default, by default False

        Returns
        -------
        str
            Unique id for the output and plugin. This is necessary because ids are defined globally.
        """

        assert isinstance(attribute, str)

        if mpl:
            id += "-mpl"

        key = (id, attribute, mpl)
        if key not in self.outputs:
            self.outputs.append(key)

        return self.get_internal_output_id(id)

    def get_internal_id(self, id: str) -> str:
        return f"{self.id}-{id}"

    def get_internal_input_id(self, id: str) -> str:
        return f"{self.id}-{id}-input"

    def get_internal_output_id(self, id: str) -> str:
        return f"{self.id}-{id}-output"

    def register_callbacks(self) -> None:
        """
        Registers basic callbacks for the plugin. Following callbacks are registered:
        - If inputs changes, the changes are pasted back. This is in particular
        interest if input dependencies are used.
        - Alert messages.
        - Raw data dialog.
        """

        # We have to call the output layout one time to register
        # the values
        # Problem: Inputs/Outputs can't be changed afterwards anymore.

        if self.activate_run_selection:
            self.__class__.get_run_input_layout(self.register_input)

        self.__class__.get_input_layout(self.register_input)
        self.__class__.get_filter_layout(lambda a, b: self.register_input(a, b, filter=True))
        self.__class__.get_output_layout(self.register_output)
        self.__class__.get_mpl_output_layout(lambda a, b: self.register_output(a, b, mpl=True))

        # Handles the initial and the cashed input values
        outputs = []
        # [Input(self.get_internal_id("update-button"), 'n_clicks')]
        inputs = []

        # Define also inputs if they are declared as interactive
        for id, attribute, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        for id, attribute, _ in self.inputs:
            outputs.append(Output(self.get_internal_input_id(id), attribute))

        if len(outputs) > 0:

            @app.callback(outputs, inputs)
            def plugin_input_update(*inputs_list):
                init = True
                # Simple check if page was loaded for the first time
                for input in inputs_list:
                    if input is not None:
                        init = False
                        break

                # Reload our inputs
                if init:
                    inputs = c.get("last_inputs", self.id)

                    if inputs is None:
                        inputs = self.load_inputs()

                        # Also update the run selection
                        if self.activate_run_selection:
                            new_inputs = self.__class__.load_run_inputs(
                                self.runs,
                                self.groups,
                                self.__class__.check_run_compatibility,
                            )
                            update_dict(inputs, new_inputs)

                        # Set not used inputs
                        for (id, attribute, _) in self.inputs:
                            if id not in inputs:
                                inputs[id] = {}

                            if attribute not in inputs[id]:
                                inputs[id][attribute] = None
                else:
                    # Map the list `inputs` to a dict.
                    inputs = self._list_to_dict(inputs_list)

                    if len(self.previous_inputs) == 0:
                        self.previous_inputs = inputs.copy()

                    # Only work on copies.
                    # We don't want the inputs dict to be changed by the user.
                    _previous_inputs = self.previous_inputs.copy()
                    _inputs = inputs.copy()

                    selected_run: Optional[AbstractRun] = None
                    if self.activate_run_selection:
                        if "run_name" in _previous_inputs:
                            _previous_run_name = _previous_inputs["run_name"]["value"]
                        else:
                            _previous_run_name = None
                        _run_name = inputs["run_name"]["value"]

                        # Reset everything if run name changed.
                        if _previous_run_name is not None and _previous_run_name != _run_name:
                            # We can't use load_inputs here only
                            # because `run_name` would be removed.
                            # Also: We want to keep the current run name.
                            update_dict(_inputs, self.load_inputs())

                            # TODO: Reset only inputs which are not available in another ru.
                            # E.g. if options from budget in run_2 and run_3 are the same
                            # take the budget from run_2 if changed to run_3. Otherwise, reset budgets.

                        selected_run = self.all_runs[inputs["run_name"]["value"]]

                    # How to update only parameters which have a dependency?
                    user_dependencies_inputs = self.load_dependency_inputs(
                        _previous_inputs, _inputs, selected_run
                    )

                    # Update dict
                    # dict.update() remove keys, so it's done manually
                    update_dict(inputs, user_dependencies_inputs)  # inplace

                # From dict to list
                inputs_list = self._dict_to_list(inputs, input=True)
                self.previous_inputs = inputs

                return inputs_list

        # Update internal alert state to divs
        @app.callback(
            Output(self.get_internal_id("alert"), "children"),
            Output(self.get_internal_id("alert"), "color"),
            Output(self.get_internal_id("alert"), "is_open"),
            Input(self.get_internal_id("alert-interval"), "n_intervals"),
        )
        def update_alert(_):
            if self.alert_update_required:
                self.alert_update_required = False
                return self.alert_text, self.alert_color, True
            else:
                raise PreventUpdate()

        # Register modal here
        @app.callback(
            [
                Output(self.get_internal_id("raw_data"), "is_open"),
                Output(self.get_internal_id("raw_data_content"), "value"),
            ],
            Input(self.get_internal_id("show_raw_data"), "n_clicks"),
            State(self.get_internal_id("raw_data"), "is_open"),
        )
        def toggle_modal(n, is_open):
            code = ""
            if n:
                if (out := self.raw_outputs) is not None:
                    # Make list
                    code = str(out)

                return not is_open, code

            return is_open, code

    def update_alert(self, text: str, color: str = "success"):
        """
        Update the alert text and color. Will automatically trigger the alert callback.

        Parameters
        ----------
        text : str
            The text to display.
        color : str, optional
            The color to display, by default "success"
        """
        self.alert_text = text
        self.alert_color = color
        self.alert_update_required = True

    def _inputs_changed(self, inputs, last_inputs):
        # Check if last_inputs are the same as the given inputs.
        inputs_changed = False
        filters_changed = False

        # If only filters changed, then we don't need to
        # calculate the results again.
        if last_inputs is not None:
            for (id, attribute, filter) in self.inputs:

                if self.activate_run_selection:
                    if id == "run_name":
                        continue

                if inputs[id][attribute] != last_inputs[id][attribute]:
                    if not filter:
                        inputs_changed = True
                    else:
                        filters_changed = True

        return inputs_changed, filters_changed

    def _process_raw_outputs(self, inputs, raw_outputs):
        logger.debug("Process raw outputs.")

        # Use raw outputs to update our layout
        mpl_active = c.get("matplotlib-mode")

        passed_runs = self.all_runs
        passed_outputs = raw_outputs
        if self.activate_run_selection:
            passed_runs = self.all_runs[inputs["run_name"]["value"]]
            passed_outputs = raw_outputs[passed_runs.name]

        if mpl_active:
            outputs = self.load_mpl_outputs(inputs, passed_outputs, passed_runs)
        else:
            outputs = self.load_outputs(inputs, passed_outputs, passed_runs)

        if outputs == PreventUpdate:
            raise PreventUpdate()

        # Map outputs here because it may be that the outputs are
        # differently sorted than the values were registered.
        if isinstance(outputs, dict):
            outputs = self._dict_to_list(outputs, input=False)
        else:
            if not isinstance(outputs, list):
                outputs = [outputs]

        # We have to add no_updates here for the mode we don't want
        count_outputs = 0
        count_mpl_outputs = 0
        for (_, _, mpl_mode) in self.outputs:
            if mpl_mode:
                count_mpl_outputs += 1
            else:
                count_outputs += 1

        if mpl_active:
            outputs = [no_update for _ in range(count_outputs)] + outputs
        else:
            outputs = outputs + [no_update for _ in range(count_mpl_outputs)]

        return outputs

    def _list_to_dict(self, values: Iterable[str], input=True) -> Dict[str, Dict[str, str]]:
        """
        Maps the given values to a dict, regarding the sorting from
        either self.inputs or self.outputs.

        Returns:
            dict
        """

        if input:
            order = self.inputs
        else:
            order = self.outputs

        mapping = {}
        for value, (id, attribute, *_) in zip(values, order):
            if id not in mapping:
                mapping[id] = {}

            mapping[id][attribute] = value

        return mapping

    def _dict_to_list(self, d: Dict[str, Dict[str, str]], input=False) -> List[Optional[str]]:
        """
        Maps the given dict to a list, regarding the sorting from either
        self.inputs or self.outputs.

        Returns:
            list.
        """

        if input:
            order = self.inputs
        else:
            order = self.outputs

        result = []
        for (id, attribute, instance) in order:

            if not input:
                # Instance is mlp_mode in case of outputs
                # Simply ignore other outputs.
                if instance != c.get("matplotlib-mode"):
                    continue
            try:
                value = d[id][attribute]
                result += [value]
            except:
                result += [None]

        return result

    def _dict_as_key(self, d: Dict[str, Any], remove_filters=False) -> Optional[str]:
        """
        Converts a dictionary to a key. Only ids from self.inputs are considered.

        Parameters:
            d (dict): Dictionary to get the key from.
            remove_filters (bool): Option wheather the filters should be included or not.

        Returns:
            key (str): Key as string from the given dictionary.
        """

        if not isinstance(d, dict):
            return None

        new_d = copy.deepcopy(d)
        if remove_filters:
            for (id, _, filter) in self.inputs:
                if filter:
                    if id in new_d:
                        del new_d[id]

        return string_to_hash(str(new_d))

    def __call__(self, render_button: bool = False) -> List[Component]:
        """
        Returns the components for the plugin. Basically, all blocks and elements of the plugin
        are stacked-up here

        Returns
        -------
        List[Component]
            Layout as list of components.
        """

        self.previous_inputs = {}
        self.raw_outputs = None
        self.runs = run_handler.get_runs()
        groups = run_handler.get_groups()

        self.groups = {
            name: GroupedRun(name, [self.runs[run_id] for run_id in run_ids])
            for name, run_ids in groups.items()
        }

        self.all_runs = {}
        self.all_runs.update(add_prefix_to_dict(self.runs, "run:"))
        self.all_runs.update(add_prefix_to_dict(self.groups, "group:"))

        components = [html.H1(self.name)]
        if self.description is not None:
            components += [html.P(self.description)]

        # Register alerts
        components += [
            dcc.Interval(
                id=self.get_internal_id("alert-interval"),
                interval=1 * 500,
                n_intervals=5,
            ),
            dbc.Alert(
                id=self.get_internal_id("alert"),
                is_open=False,
                dismissable=True,
                fade=True,
            ),
        ]

        try:
            self.check_runs_compatibility(list(self.all_runs.values()))
        except NotMergeableError as message:
            self.update_alert(str(message), color="danger")
            return components

        if self.activate_run_selection:
            run_input_layout = [self.__class__.get_run_input_layout(self.register_input)]
        else:
            run_input_layout = []

        input_layout = self.__class__.get_input_layout(self.register_input)

        separator_layout = []
        if input_layout and run_input_layout:
            separator_layout.append(html.Hr())

        input_control_layout = html.Div(
            style={} if render_button else {"display": "none"},
            className="mt-3 clearfix",
            children=[
                dbc.Button(
                    children=self.button_caption,
                    id=self.get_internal_id("update-button"),
                ),
                html.Span(
                    html.Em(id=self.get_internal_id("processing-info")),
                    className="ms-3 align-baseline",
                ),
            ],
        )

        # We always have to render it because of the button.
        # Button tells us if the page was just loaded.
        components += [
            html.Div(
                id=f"{self.id}-input",
                className="shadow-sm p-3 mb-3 bg-white rounded-lg",
                children=run_input_layout
                + separator_layout
                + input_layout
                + [input_control_layout],
                style={}
                if render_button or input_layout or run_input_layout
                else {"display": "none"},
            )
        ]

        def register_in(a, b):
            return self.register_input(a, b, filter=True)

        filter_layout = self.__class__.get_filter_layout(register_in)
        if len(filter_layout) > 0:
            components += [
                html.Div(
                    id=f"{self.id}-filter",
                    className="shadow-sm p-3 mb-3 bg-white rounded-lg",
                    children=filter_layout,
                )
            ]

        output_layout = self.__class__.get_output_layout(self.register_output)
        if output_layout:
            components += [
                html.Div(
                    id=f"{self.id}-output",
                    className="shadow-sm p-3 bg-white rounded-lg loading-container",
                    children=output_layout,
                    style={} if not c.get("matplotlib-mode") else {"display": "none"},
                )
            ]

        def register_out(a, b):
            return self.register_output(a, b, mpl=True)

        output_layout = self.__class__.get_mpl_output_layout(register_out)
        if output_layout:
            components += [
                html.Div(
                    id=f"{self.id}-mpl-output",
                    className="shadow-sm p-3 bg-white rounded-lg loading-container",
                    children=output_layout,
                    style={} if c.get("matplotlib-mode") else {"display": "none"},
                )
            ]

        modal = html.Div(
            [
                dbc.Button(
                    "Raw Data",
                    id=self.get_internal_id("show_raw_data"),
                    className="mt-3",
                    n_clicks=0,
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(
                            [
                                dbc.ModalTitle("Raw Data"),
                                dcc.Clipboard(
                                    target_id=self.get_internal_id("raw_data_content"),
                                    style={
                                        "fontSize": 20,
                                        "marginLeft": "0.5rem",
                                    },
                                ),
                            ]
                        ),
                        dbc.ModalBody(
                            [
                                dbc.Textarea(
                                    id=self.get_internal_id("raw_data_content"),
                                    placeholder="",
                                    readonly=True,
                                    rows=20,
                                ),
                            ]
                        ),
                    ],
                    id=self.get_internal_id("raw_data"),
                    size="lg",
                    scrollable=True,
                    is_open=False,
                ),
            ]
        )

        components += [modal]

        return components

    @staticmethod
    def get_run_input_layout(register: Callable[[str, Union[str, List[str]]], str]) -> Component:
        """
        Generates the run selection input.
        This is only the case if `activate_run_selection` is True.

        Parameters
        ----------
        register : Callable[[str, Union[str, List[str]]], str]
            The register method to register (user) variables.

        Returns
        -------
        Component
            The layout of the run selection input.
        """
        return html.Div(
            [
                dbc.Select(
                    id=register("run_name", ["options", "value"]),
                    placeholder="Select run ...",
                ),
            ]
        )

    @staticmethod
    def load_run_inputs(
        runs: Dict[str, AbstractRun],
        groups: Dict[str, GroupedRun],
        check_run_compatibility: Callable[[AbstractRun], bool],
    ) -> Dict[str, Any]:
        """
        Loads the options for `get_run_input_layout`.
        Both runs and groups are displayed.

        Parameters
        ----------
        runs : Dict[str, Run]
            The runs to display.
        groups : Dict[str, GroupedRun]
            The groups to display.
        check_run_compatibility : Callable[[AbstractRun], bool]
            If a single run is compatible. If not, the run is not shown.

        Returns
        -------
        Dict[str, Any]
            Both runs and groups, separated by a separator.
        """

        labels = []
        values = []
        disabled = []

        for id, run in runs.items():
            try:
                values.append(f"run:{id}")
                labels.append(id)
                disabled.append(False)
            except:
                pass

        added_group_label = False
        for id, run in groups.items():
            if check_run_compatibility(run):
                if not added_group_label:
                    values.append("")
                    labels.append("Groups")
                    disabled.append(True)
                    added_group_label = True

                values.append(f"group:{id}")
                labels.append(id)
                disabled.append(False)

        return {
            "run_name": {
                "options": get_select_options(labels=labels, values=values, disabled=disabled),
                "value": None,
            }
        }

    def get_selected_runs(self, inputs: Dict[str, Any]) -> List[AbstractRun]:
        """
        Parses selected runs from inputs.
        If self.activate_run_selection is set, return only selected run. Otherwise, return all
        possible runs.

        Parameters
        ----------
        inputs : Dict[str, Any]
            The inputs to parse.

        Returns
        -------
        List[AbstractRun]
            The selected runs.

        Raises
        ------
        PreventUpdate
            If `activate_run_selection` is set but `run_name` is not available.
        """

        # Special case: If run selection is active
        # Don't update anything if the inputs haven't changed
        if self.activate_run_selection:
            if inputs["run_name"]["value"] is None:
                raise PreventUpdate()

            # Update runs
            run = run_handler.from_run_id(inputs["run_name"]["value"])

            # Also:
            # Remove `run_name` from inputs_key because
            # we don't want the run names included.
            _inputs = inputs.copy()
            del _inputs["run_name"]

            return [run]
        else:
            return list(self.all_runs.values())

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in `get_input_layout` and `get_filter_layout`.
        This method is necessary to pre-load contents for the inputs. So, if the plugin is
        called for the first time or there are no results in the cache, the plugin gets its
        content from this method.

        Returns
        -------
        Dict[str, Any]
            Content to be filled.
        """
        return {}

    def load_dependency_inputs(
        self,
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
        selected_run: Optional[Union[AbstractRun, List[AbstractRun]]] = None,
    ) -> Dict[str, Any]:
        """
        Same as `load_inputs` but called after inputs have changed. Provides a lot of flexibility.

        Parameters
        ----------
        previous_inputs : Dict[str, Any]
            Previous content of the inputs.
        inputs : Dict[str, Any]
            Current content of the inputs.
        selected_run : Optional[Union[AbstractRun, List[AbstractRun]]], optional
            The selected run from the user. In case of `activate_run_selection`, a list of runs
            are passed. Defaults to None.

        Returns
        -------
        Dict[str, Any]
            Content to be filled.
        """

        return inputs

    @staticmethod
    def get_input_layout(register: Callable[[str, Union[str, List[str]]], str]) -> List[Component]:
        """
        Layout for the input block.

        Parameters
        ----------
        register : Callable[[str, Union[str, List[str]]], str]
            The register method to register (user) variables.

        Returns
        -------
        List[Component]
            Layouts for the input block.
        """

        return []

    @staticmethod
    def get_filter_layout(register: Callable[[str, Union[str, List[str]]], str]):
        """
        Layout for the filter block.

        Parameters
        ----------
        register : Callable[[str, Union[str, List[str]]], str]
            The register method to register (user) variables.

        Returns
        -------
        List[Component]
            Layouts for the filter block.
        """

        return []

    @staticmethod
    def get_output_layout(register: Callable[[str, Union[str, List[str]]], str]):
        """
        Layout for the output block.

        Parameters
        ----------
        register : Callable[[str, Union[str, List[str]]], str]
            The register method to register outputs.

        Returns
        -------
        List[Component]
            Layouts for the output block.
        """

        return []

    @staticmethod
    def get_mpl_output_layout(register: Callable[[str, Union[str, List[str]]], str]):
        """
        Layout for the matplotlib output block.

        Parameters
        ----------
        register : Callable[[str, Union[str, List[str]]], str]
            The register method to register outputs.

        Returns
        -------
        List[Component]
            Layout for the matplotlib output block.
        """

        return []

    def load_outputs(
        self,
        inputs: Dict[str, Dict[str, str]],
        outputs: Dict[str, Union[str, Dict[str, str]]],
        runs: Union[AbstractRun, Dict[str, AbstractRun]],
    ) -> List[Component]:
        """
        Reads in the raw data and prepares them for the layout.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, str]]
            Input and filter values from the user.
        outputs : Dict[str, Union[str, Dict[str, str]]]
            Raw outputs from the runs. If `activate_run_selection` is set,
            a Dict[str, str] is returned.
        runs : Union[AbstractRun, Dict[str, AbstractRun]]
            All selected runs. If `activate_run_selection` is set, only the selected run is
            returned.

        Returns
        -------
        List[Component]
            The components must be in the same position as defined in `get_output_layout`.
        """

        return []

    def load_mpl_outputs(
        self,
        inputs: Dict[str, Dict[str, str]],
        outputs: Dict[str, Union[str, Dict[str, str]]],
        runs: Union[AbstractRun, Dict[str, AbstractRun]],
    ) -> List[Component]:
        """
        Reads in the raw data and prepares them for the layout.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, str]]
            Input and filter values from the user.
        outputs : Dict[str, Union[str, Dict[str, str]]]
            Raw outputs from the runs. If `activate_run_selection` is set,
            a Dict[str, str] is returned.
        runs : Union[AbstractRun, Dict[str, AbstractRun]]
            All selected runs. If `activate_run_selection` is set, only the selected run is
            returned.

        Returns
        -------
        List[Component]
            The components must be in the same position as defined in `get_output_layout`.
        """

        return []

    @staticmethod
    @abstractmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]):
        """
        Returns raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs : Dict[str, Any]
            Input data.
        """

        pass

    @staticmethod
    def _process(process: Callable[[AbstractRun, Any], None], run_cache_id: str, inputs):
        run = run_handler.from_run_cache_id(run_cache_id)
        return process(run, inputs)
