#  noqa: D400
"""
# Plugins

This module provides a base class for all the available plugins.
It provides different utilities to handle the plugins and check for compatibility in the runs.

## Classes
    - Plugin: Base class for all plugins.
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy
import re
import webbrowser
from collections import defaultdict

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import ROOT_DIR, interactive
from deepcave.layouts import Layout
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group, NotMergeableError
from deepcave.utils.data_structures import update_dict
from deepcave.utils.docs import rst_to_md
from deepcave.utils.hash import string_to_hash
from deepcave.utils.layout import get_select_options
from deepcave.utils.logs import get_logger
from deepcave.utils.url import parse_url

logger = get_logger(__name__)


class Plugin(Layout, ABC):
    """
    Base class for all plugins.

    Provides different utilities to handle the plugins and check for compatibility in the runs.

    Properties
    ----------
    inputs : List[Tuple[str, str, bool, Any]]
        The registered inputs.
    outputs : List[Tuple[str, str, bool]]
        The registered outputs.
    previous_inputs : Dict[str, Dict[str, str]]
        The previous inputs.
    raw_outputs : Optional[Dict[str, Any]]
        The raw outputs.
    activate_run_selection : bool
        Shows a dropdown to select a run in the inputs layout.
        This feature is useful if only one run could be viewed at a time.
        Moreover, it prevents the plugin to calculate results across all runs.
    id : str
        The unique identifier for the plugin.
    runs : List[AbstractRun]
        A list of the abstract runs.
    groups : List[Group]
        A list of the groups.
    help : str
        The path to the documentation.
    name : str
        The name of the plugin.
        It is shown in the navigation and title.
    button_caption : str
        Caption of the button. Shown only, if `StaticPlugin` is used.
    """

    id: str
    name: str
    description: Optional[str] = None
    icon: str = "far fa-file"
    help: Optional[str] = None
    button_caption: str = "Process"
    activate_run_selection: bool = False

    def __init__(self) -> None:
        # Registered inputs and outputs
        self.inputs: List[Tuple[str, str, bool, Any]] = []
        self.outputs: List[Tuple[str, str, bool]] = []

        # For runtime
        self.previous_inputs: Dict[str, Dict[str, str]] = {}
        self.raw_outputs: Optional[Dict[str, Any]] = None

        # The output layout has to be called one time to register
        # the values
        # Problem: Inputs/Outputs can't be changed afterwards anymore.

        if self.activate_run_selection:
            self.__class__.get_run_input_layout(self.register_input)

        self.__class__.get_input_layout(self.register_input)
        self.__class__.get_filter_layout(lambda a, b: self.register_input(a, b, filter=True))
        self.__class__.get_output_layout(self.register_output)
        self.__class__.get_mpl_output_layout(lambda a, b: self.register_output(a, b, mpl=True))

        super().__init__()

    @classmethod
    @interactive
    def get_base_url(cls) -> str:
        """
        Generate the url for the plugin.

        Returns
        -------
        str
            Url for the plugin as string.
        """
        from deepcave import config

        return f"http://{config.DASH_ADDRESS}:{config.DASH_PORT}/plugins/{cls.id}"

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """
        Check if a run is compatible with this plugin.

        If a plugin is not compatible, you can not select the run.

        Note
        ----
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
        Needed if all selected runs need something in common.

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
        return

    def register_input(
        self,
        id: str,
        attributes: Union[str, List[str]] = "value",
        filter: bool = False,
        type: Any = None,
    ) -> str:
        """
        Register an input variable for the plugin.

        It is important to register the inputs.
        This is, because callbacks have to be defined before the server is started.
        After registering all inputs, an internal mapping is created.

        Parameters
        ----------
        id : str
            Specifies the id of the input.
        attributes : Union[str, List[str]], optional
            Attributes which should be passed to the (dash) component, by default ("value",).
        filter : bool, optional
            Specifies if the input is a filter. By default False.
        type : Any, optional
            Type to which the *first* attribute should be casted to. By default str.

        Returns
        -------
        id : str
            Unique id for the input and plugin. This is necessary because ids are defined globally.
        """
        if isinstance(attributes, str):
            attributes = [attributes]

        for i, attribute in enumerate(attributes):
            # Only cast the first attribute
            type_ = None
            if i == 0:
                type_ = type

            key = (id, attribute, filter, type_)
            if key not in self.inputs:
                self.inputs.append(key)

        # The inputs have to be rearranged, because `State`
        # must follow all `Input`. Since all filters are `Input`, they have to
        # be shifted to the front.
        self.inputs.sort(key=lambda x: x[2], reverse=True)

        return self.get_internal_input_id(id)

    def register_output(
        self, id: str, attributes: Union[str, List[str]] = "value", mpl: bool = False
    ) -> str:
        """
        Register an output variable for the plugin.

        Parameters
        ----------
        id : str
            Specifies the id of the output.
        attributes : Union[str, List[str]], optional
            Attribute, by default "value"
        mpl : bool, optional
            Specifies if the registration is for matplotlib or default, by default False

        Returns
        -------
        id : str
            Unique id for the output and plugin. This is necessary because ids are defined globally.
        """
        if isinstance(attributes, str):
            attributes = [attributes]

        if mpl:
            id += "-mpl"

        for attribute in attributes:
            key = (id, attribute, mpl)
            if key not in self.outputs:
                self.outputs.append(key)

        return self.get_internal_output_id(id)

    def get_internal_id(self, id: str) -> str:
        """Get the internal id."""
        return f"{self.id}-{id}"

    def get_internal_input_id(self, id: str) -> str:
        """Get the internal input id."""
        return f"{self.id}-{id}-input"

    def get_internal_output_id(self, id: str) -> str:
        """Get the internal output id."""
        return f"{self.id}-{id}-output"

    @interactive
    def register_callbacks(self) -> None:
        """
        Register basic callbacks for the plugin.

        Following callbacks are registered:
        - If inputs changes, the changes are pasted back. This is in particular
        interest if input dependencies are used.
        - Raw data dialog to display raw data.
        - Callback to be redirected to the config if clicked on it.

        Raises
        ------
        RuntimeError
            If no run id is found.
        """
        from deepcave import app, c, run_handler

        # Handles the initial and the cashed input values
        outputs = []
        inputs = [Input("on-page-load", "href")]

        # Define also inputs if they are declared as interactive
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        for id, attribute, _, _ in self.inputs:
            outputs.append(Output(self.get_internal_input_id(id), attribute))

        if len(outputs) > 0:

            @app.callback(outputs, inputs)  # type: ignore
            def plugin_input_update(pathname: str, *inputs_list: str) -> List[Optional[str]]:
                """Update the input of the plugin."""
                # Simple check if page was loaded for the first time
                init = all(input is None for input in inputs_list)

                # Reload our inputs
                if init:
                    inputs = c.get("last_inputs", self.id)
                    passed_inputs = parse_url(pathname)

                    if passed_inputs is not None:
                        # First get normal inputs
                        inputs = self.load_inputs()

                        # Overwrite/set the passed inputs
                        update_dict(inputs, passed_inputs)

                        # Then the run_selection has to be taken care of
                        selected_run: Optional[AbstractRun] = None
                        if self.activate_run_selection:
                            # If run_selection is active and the id is not known, then
                            # the passed inputs have no use.
                            try:
                                run_id = passed_inputs["run"]["value"]
                            except Exception:
                                raise RuntimeError("No run id found.")
                            selected_run = run_handler.get_run(run_id)

                            # Update run_selection
                            new_inputs = self.__class__.load_run_inputs(
                                self.runs,
                                self.groups,
                                self.__class__.check_run_compatibility,
                            )

                            # Overwrite `run_id` and update the whole dict.
                            new_inputs["run"]["value"] = run_id
                            update_dict(inputs, new_inputs)

                        # And lastly update with the dependencies here
                        user_dependencies_inputs = self.load_dependency_inputs(
                            selected_run, inputs, inputs
                        )
                        update_dict(inputs, user_dependencies_inputs)

                    elif inputs is None:
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
                        for id, attribute, _, _ in self.inputs:
                            if id not in inputs:
                                inputs[id] = {}

                            if attribute not in inputs[id]:
                                inputs[id][attribute] = None
                    elif inputs is not None:
                        # The options of the run selection have to be updated here.
                        # This is important if the user have added/removed runs.
                        if self.activate_run_selection:
                            run_value = inputs["run"]["value"]
                            new_inputs = self.__class__.load_run_inputs(
                                self.runs,
                                self.groups,
                                self.__class__.check_run_compatibility,
                            )
                            update_dict(inputs, new_inputs)

                            # Keep the run value
                            inputs["run"]["value"] = run_value
                else:
                    # Map the list `inputs` to a dict.
                    # inputs_list_as_list is necessary as new variable,
                    # because inputs_list is a tuple and cant be passed to _list_to_dict.
                    inputs_list_as_list = list(inputs_list)
                    inputs = self._list_to_dict(inputs_list_as_list)

                    if len(self.previous_inputs) == 0:
                        self.previous_inputs = inputs.copy()

                    # Only work on copies.
                    # The inputs dict should not be changed by the user.
                    _previous_inputs = self.previous_inputs.copy()
                    _inputs = inputs.copy()

                    selected_run = None
                    if self.activate_run_selection:
                        if "run" in _previous_inputs:
                            _previous_run_id = _previous_inputs["run"]["value"]
                        else:
                            _previous_run_id = None
                        _run_id = inputs["run"]["value"]

                        # Reset everything if run name changed.
                        if _previous_run_id is not None and _previous_run_id != _run_id:
                            # load_inputs cannot be used here, only
                            # because `run` would be removed.
                            # Also: The current run name does not need to be kept.
                            update_dict(_inputs, self.load_inputs())
                            # Reset inputs
                            if "objective_id" in _inputs.keys():
                                update_dict(_inputs, {"objective_id": {"value": None}})
                            if "budget_id" in _inputs.keys():
                                update_dict(_inputs, {"budget_id": {"value": None}})
                            if "hyperparameter_name_1" in _inputs.keys():
                                update_dict(_inputs, {"hyperparameter_name_1": {"value": None}})
                            if "hyperparameter_name_2" in _inputs.keys():
                                update_dict(_inputs, {"hyperparameter_name_2": {"value": None}})

                        if _run_id:
                            selected_run = run_handler.get_run(_run_id)

                    if selected_run is not None:
                        # How to update only parameters which have a dependency?
                        user_dependencies_inputs = self.load_dependency_inputs(
                            selected_run, _previous_inputs, _inputs
                        )

                        # Update dict
                        # dict.update() removes keys, so our own method is used to do so
                        update_dict(inputs, user_dependencies_inputs)  # inplace operation

                # Let's cast the inputs
                inputs = self._cast_inputs(inputs)

                # From dict to list
                inputs_list_from_dict = self._dict_to_list(inputs, input=True)
                self.previous_inputs = inputs

                return inputs_list_from_dict

        # Register modal for raw data here
        @app.callback(  # type: ignore
            [
                Output(self.get_internal_id("raw_data"), "is_open"),
                Output(self.get_internal_id("raw_data_content"), "value"),
            ],
            Input(self.get_internal_id("show_raw_data"), "n_clicks"),
            State(self.get_internal_id("raw_data"), "is_open"),
        )
        def toggle_raw_data_modal(n: Optional[int], is_open: bool) -> Tuple[bool, str]:
            """Toggle the raw data modal."""
            code = ""
            if n:
                if (out := self.raw_outputs) is not None:
                    # Make list
                    code = str(out)

                return not is_open, code

            return is_open, code

        # Register modal for help here
        @app.callback(  # type: ignore
            [
                Output(self.get_internal_id("help"), "is_open"),
            ],
            Input(self.get_internal_id("show_help"), "n_clicks"),
            State(self.get_internal_id("help"), "is_open"),
        )
        def toggle_help_modal(n: Optional[int], is_open: bool) -> bool:
            """Toggle the help modal."""
            if n:
                return not is_open
            return is_open

        # Register callback to click on configurations
        for id, *_ in self.outputs:
            internal_id = self.get_internal_output_id(id)

            @app.callback(
                Output(internal_id, "clickData"),
                Input(internal_id, "clickData"),
            )  # type: ignore
            def go_to_configuration(click_data: Any):
                """Open link from hovertext."""
                if click_data is not None:
                    # Get hovertext
                    try:
                        hovertext = click_data["points"][0]["hovertext"]

                        # Now extract the link from href
                        match = re.search("<a href='(.+?)'", hovertext)
                        if match:
                            link = match.group(1)
                            webbrowser.open(link, new=0)
                    except Exception:
                        pass

                return None

    @interactive
    def _inputs_changed(
        self, inputs: Dict[str, Dict[str, str]], last_inputs: Dict[str, Dict[str, str]]
    ) -> Tuple[bool, bool]:
        """
        Check if the inputs have changed.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, str]]
            Current inputs.
        last_inputs : Dict[str, Dict[str, str]]
            Last inputs.

        Returns
        -------
        Tuple[bool, bool]
            Whether input and filter inputs have changed.
        """
        # Check if last_inputs are the same as the given inputs.
        inputs_changed = False
        filters_changed = False

        # If only filters changed, there is no need to
        # calculate the results again.
        if last_inputs is not None:
            for id, attribute, filter, _ in self.inputs:
                if self.activate_run_selection:
                    if id == "run":
                        continue

                if inputs[id][attribute] != last_inputs[id][attribute]:
                    if not filter:
                        inputs_changed = True
                    else:
                        filters_changed = True

        return inputs_changed, filters_changed

    @interactive
    def _process_raw_outputs(
        self, inputs: Dict[str, Dict[str, str]], raw_outputs: Dict[str, Any]
    ) -> Any:
        """
        Process the raw outputs and update the layout.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, str]]
            The inputs for the passed runs.
        raw_outputs : Dict[str, Any]
            The raw outputs to process.

        Returns
        -------
        Any
            The processed outputs.
        """
        from deepcave import c, run_handler

        # Use raw outputs to update our layout
        mpl_active = c.get("matplotlib-mode")
        passed_runs: Union[List[AbstractRun], AbstractRun]

        if self.activate_run_selection:
            passed_runs = run_handler.get_run(inputs["run"]["value"])
            passed_outputs = raw_outputs[passed_runs.id]
        else:
            passed_runs = self.all_runs
            passed_outputs = raw_outputs

        # Clean inputs
        cleaned_inputs = self._clean_inputs(inputs)

        # passed runs could be a list, but load mpl outputs and load outputs do not
        # accept lists, but expect single runs
        if mpl_active:
            outputs = self.__class__.load_mpl_outputs(passed_runs, cleaned_inputs, passed_outputs)  # type: ignore # noqa: E501
        else:
            outputs = self.__class__.load_outputs(passed_runs, cleaned_inputs, passed_outputs)  # type: ignore # noqa: E501

        logger.debug("Raw outputs processed successfully.")

        if outputs == PreventUpdate:
            raise PreventUpdate()

        # Map outputs here because it may be that the outputs are
        # differently sorted than the values were registered.
        if isinstance(outputs, dict):
            outputs = self._dict_to_list(outputs, input=False)
        else:
            if not isinstance(outputs, list):
                outputs = [outputs]

        # no_updates has to be added here for the mode that is not wanted
        count_outputs = 0
        count_mpl_outputs = 0
        for _, _, mpl_mode in self.outputs:
            if mpl_mode:
                count_mpl_outputs += 1
            else:
                count_outputs += 1

        if mpl_active:
            outputs = [no_update for _ in range(count_outputs)] + outputs
        else:
            outputs = outputs + [no_update for _ in range(count_mpl_outputs)]

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    @interactive
    def _list_to_dict(self, values: List[str], input: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Map the given values to a dict.

        Regarding the sorting from either self.inputs or self.outputs.

        Parameters
        ----------
        values : Iterable[str]
            Values to map.
        input : bool, optional
            Whether the data should be linked to the input or outputs. By default True.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary containing the mapping information.
        """
        # This is necessary, because of the conditional type of order
        order: Union[List[Tuple[str, str, bool]], List[Tuple[str, str, bool, Any]]]

        if input:
            order = self.inputs
        else:
            order = self.outputs

        mapping: Dict[str, Any] = {}
        for value, (id, attribute, *_) in zip(values, order):
            if id not in mapping:
                mapping[id] = {}

            mapping[id][attribute] = value

        return mapping

    @interactive
    def _dict_to_list(
        self, d: Dict[str, Dict[str, str]], input: bool = False
    ) -> List[Optional[str]]:
        """
        Map the given dict to a list.

        Respecting the sorting from either self.inputs or self.outputs.

        Parameters
        ----------
        d : Dict[str, Dict[str, str]]
            Dictionary to transform.
        input : bool, optional
            Whether the data should be linked to the input or outputs. By default False.

        Returns
        -------
        List[Optional[str]]
            Sorted list from the given dict.
        """
        from deepcave import c

        # This is necessary, because of the conditional type of order
        order: Union[List[Tuple[str, str, bool]], List[Tuple[str, str, bool, Any]]]

        if input:
            order = self.inputs
        else:
            order = self.outputs

        result: List[Optional[str]] = []
        for id, attribute, instance, *_ in order:
            if not input:
                # Instance is mlp_mode in case of outputs
                # Simply ignore other outputs.
                if instance != c.get("matplotlib-mode"):
                    continue

            try:
                value = d[id][attribute]
                result += [value]
            except Exception:
                result += [None]

        return result

    @interactive
    def _dict_as_key(self, d: Dict[str, Any], remove_filters: bool = False) -> str:
        """
        Convert a dictionary to a key. Only ids from self.inputs are considered.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary to get the key from.
        remove_filters : bool, optional
            Option whether the filters should be included or not. By default False.

        Returns
        -------
        Optional[str]
            Key as string from the given dictionary. Returns none if `d` is not a dictionary.

        Raises
        ------
        TypeError
            If `d` is not a dictionary.
        """
        if not isinstance(d, dict):
            raise TypeError("d must be a dictionary.")

        new_d = copy.deepcopy(d)
        if remove_filters:
            for id, _, filter, _ in self.inputs:
                if filter:
                    if id in new_d:
                        del new_d[id]

        return string_to_hash(str(new_d))

    def _cast_inputs(self, inputs: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Cast the inputs based on `self.inputs`.

        Background is that dash always casts integers/booleans to strings.
        This method ensures that the correct types are returned.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, str]]
            Inputs, which should be casted.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Casted inputs.
        """
        casted_inputs: Dict[str, Dict[str, str]] = defaultdict(dict)
        for id, attributes in inputs.items():
            for attribute in attributes:
                # Find corresponding input
                type = None
                for id_, attribute_, _, type_ in self.inputs:
                    if id == id_ and attribute == attribute_:
                        type = type_
                        break

                value = inputs[id][attribute]
                if not (type is None or value is None):
                    value = type(value)

                casted_inputs[id][attribute] = value

        return casted_inputs

    def _clean_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the given inputs s.t. only the first value is used.

        Also, boolean values are cast to booleans.

        Example
        -------
        You register the following input:
        ```
        dbc.Select(id=register("objective_name", ["value", "options"]))
        ```
        However, in the `process` or `load_outputs` method you don't need `options`.
        Instead of writing `inputs["objective_name"]["value"]` you can simply write
        `inputs["objective_name"]`.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Inputs to clean.

        Returns
        -------
        Dict[str, Any]
            Cleaned inputs.
        """
        used_ids = []
        cleaned_inputs = {}
        for id, attribute, *_ in self.inputs:
            # Since self.inputs is ordered, the first occurring attribute is used and
            # the id is added so it is not used again.
            if id not in used_ids:
                i = inputs[id][attribute]

                if i == "true":
                    i = True

                if i == "false":
                    i = False

                cleaned_inputs[id] = i
                used_ids += [id]

        return cleaned_inputs

    @property
    @interactive
    def runs(self) -> List[AbstractRun]:
        """
        Get the runs as a list.

        Returns
        -------
        List[AbstractRun]
            The list with the runs.
        """
        from deepcave import run_handler

        return run_handler.get_runs()

    @property
    @interactive
    def groups(self) -> List[Group]:
        """
        Get the groups as a list.

        Returns
        -------
        List[Group]
            The list with the groups.
        """
        from deepcave import run_handler

        return run_handler.get_groups()

    @property
    @interactive
    def all_runs(self) -> List[AbstractRun]:
        """
        Get all runs and include the groups as a list.

        Returns
        -------
        List[AbstractRun]
            The list with all runs and included groups.
        """
        from deepcave import run_handler

        return run_handler.get_runs(include_groups=True)

    @interactive
    def __call__(self, render_button: bool = False) -> List[Component]:
        """
        Return the components for the plugin.

        Basically, all blocks and elements of the plugin are stacked-up here.

        Parameters
        ----------
        render_button : bool, optional
            Whether to render the button or not. By default False.

        Returns
        -------
        List[Component]
            Layout as list of components.

        Raises
        ------
        NotMergeableError
            If runs are not compatible.
        FileNotFoundError
            If the help file can not be found.
        """
        from deepcave import c, notification

        # Reset runtime variables
        self.previous_inputs = {}
        self.raw_outputs = None

        components = []

        if self.help is not None:
            doc_path = ROOT_DIR / self.help
            if not doc_path.exists():
                raise FileNotFoundError(doc_path)

            if doc_path.name.endswith(".rst"):
                data = rst_to_md(doc_path)
            else:
                with doc_path.open("r") as file:
                    data = file.read()

            modal = html.Div(
                [
                    dbc.Modal(
                        [
                            dbc.ModalBody([dcc.Markdown(data)]),
                        ],
                        id=self.get_internal_id("help"),
                        size="xl",
                        scrollable=True,
                        is_open=False,
                    ),
                ]
            )

            components += [
                html.H1(
                    [
                        html.Span(self.name),
                        dbc.Button(
                            [html.I(className="far fa-question-circle")],
                            id=self.get_internal_id("show_help"),
                            style={"float": "right"},
                            color="primary",
                            outline=True,
                            n_clicks=0,
                        ),
                    ]
                ),
                modal,
            ]
        else:
            components += [html.H1(self.name)]
        try:
            self.check_runs_compatibility(self.all_runs)
        except NotMergeableError as message:
            notification.update(str(message))
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
            ],
        )

        # It always has to be rendered, because of the button.
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

        def register_in(a: str, b: Union[List[str], str]) -> str:
            """
            Register the given input.

            Note
            ----
            For more information, see 'register_input'.

            Parameters
            ----------
            a : str
                Specifies the id of the input.
            b : Union[List[str], str]
                Attributes which should be passed to the (dash) component, by default ("value",).

            Returns
            -------
            str
                Unique id for the input and plugin.
                This is necessary because ids are defined globally.
            """
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
        if output_layout is not None:
            components += [
                html.Div(
                    id=f"{self.id}-output",
                    className="shadow-sm p-3 bg-white rounded-lg loading-container",
                    children=output_layout,
                    style={} if not c.get("matplotlib-mode") else {"display": "none"},
                )
            ]

        def register_out(a: str, b: Union[List[str], str]) -> str:
            """
            Register the output.

            Note
            ----
            For more information, see 'register_output'

            Parameters
            ----------
            a : str
                Specifies the id of the output.
            b : Union[List[str], str]
                Attribute.

            Returns
            -------
            str
                Unique id for the output and plugin.
                This is necessary because ids are defined globally.
            """
            return self.register_output(a, b, mpl=True)

        output_layout = self.__class__.get_mpl_output_layout(register_out)
        if output_layout is not None:
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
    @interactive
    def get_run_input_layout(register: Callable) -> Component:
        """
        Generate the run selection input.

        This is only the case if `activate_run_selection` is True.

        Parameters
        ----------
        register : Callable
            The register method to register (user) variables.
            For more information, see 'register_input'.

        Returns
        -------
        Component
            The layout of the run selection input.
        """
        return html.Div(
            [
                dbc.Select(
                    id=register("run", ["value", "options"]),
                    placeholder="Select run ...",
                ),
            ]
        )

    @staticmethod
    @interactive
    def load_run_inputs(
        runs: List[AbstractRun],
        groups: List[Group],
        check_run_compatibility: Callable[[AbstractRun], bool],
    ) -> Dict[str, Any]:
        """
        Load the options for `get_run_input_layout`.

        Both runs and groups are displayed.

        Parameters
        ----------
        runs : List[AbstractRun]
            The runs to display.
        groups : List[Group]
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

        for run in runs:
            if check_run_compatibility(run):
                try:
                    run_path = run.path
                    if run_path is not None:
                        run_name = run_path.parent.name + "/" + run.name

                    values.append(run.id)
                    labels.append(run_name)
                    disabled.append(False)
                except Exception:
                    pass

        added_group_label = False
        for run in groups:
            if check_run_compatibility(run):
                if not added_group_label:
                    values.append("")
                    labels.append("Groups")
                    disabled.append(True)
                    added_group_label = True

                values.append(run.id)
                labels.append(run.name)
                disabled.append(False)

        return {
            "run": {
                "options": get_select_options(labels=labels, values=values, disabled=disabled),
                "value": None,
            }
        }

    @interactive
    def get_selected_runs(self, inputs: Dict[str, Any]) -> List[AbstractRun]:
        """
        Parse selected runs from inputs.

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
            If `activate_run_selection` is set but `run` is not available.
        """
        from deepcave import run_handler

        # Special case: If run selection is active
        # Don't update anything if the inputs haven't changed
        if self.activate_run_selection:
            if (run_id := inputs["run"]["value"]) is None:
                raise PreventUpdate()

            # Update runs
            run = run_handler.get_run(run_id)

            # Also:
            # Remove `run` from inputs_key because
            # The runs name does not need to be included
            _inputs = inputs.copy()
            del _inputs["run"]

            return [run]
        else:
            return self.all_runs

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
        run: Optional[Union[AbstractRun, List[AbstractRun]]],
        previous_inputs: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Load the content as in 'load_inputs' but called after inputs have changed.

        Provides a lot of flexibility.

        Note
        ----
        Only the changes have to be returned. The returned dictionary will be merged with the
        inputs.

        Parameters
        ----------
        run : Optional[Union[AbstractRun, List[AbstractRun]]], optional
            The selected run from the user. In case of `activate_run_selection`, only one run
            is passed. Defaults to None.
        previous_inputs : Dict[str, Any]
            Previous content of the inputs.
        inputs : Dict[str, Any]
            Current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the changes.
        """
        return inputs

    @staticmethod
    def get_input_layout(register: Callable) -> List[Component]:
        """
        Layout for the input block.

        Parameters
        ----------
        register : Callable
            The register method to register (user) variables.
            For more information, see 'register_input'.

        Returns
        -------
        List[Component]
            Layouts for the input block.
        """
        return []

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Component]:
        """
        Layout for the filter block.

        Parameters
        ----------
        register : Callable
            The register method to register (user) variables.
            For more information, see 'register_input'.

        Returns
        -------
        List[Component]
            Layouts for the filter block.
        """
        return []

    @staticmethod
    def get_output_layout(register: Callable) -> Optional[Union[Component, List[Component]]]:
        """
        Layout for the output block.

        Parameters
        ----------
        register : Callable
            The register method to register outputs.
            For more information, see 'register_input'.

        Returns
        -------
        Union[Component, List[Component]]
            Layouts for the output block.
        """
        return None

    @staticmethod
    def get_mpl_output_layout(register: Callable) -> Optional[Union[Component, List[Component]]]:
        """
        Layout for the matplotlib output block.

        Parameters
        ----------
        register : Callable
            The register method to register outputs.
            For more information, see 'register_input'.

        Returns
        -------
        Optional[Union[Component, List[Component]]]
            Layout for the matplotlib output block.
        """
        return None

    @staticmethod
    def load_outputs(
        runs: Union[AbstractRun, Dict[str, AbstractRun]],
        inputs: Dict[str, Dict[str, str]],
        outputs: Dict[str, Union[str, Dict[str, str]]],
    ) -> Union[Component, List[Component]]:
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed `inputs` are cleaned and therefore differs compared to `load_inputs` or
        `load_dependency_inputs`.
        Inputs are cleaned s.t. only the first value is used.
        Also, boolean values are casted to booleans.

        Parameters
        ----------
        runs : Union[AbstractRun, Dict[str, AbstractRun]]
            All selected runs. If `activate_run_selection` is set, only the selected run is
            returned.
        inputs : Dict[str, Dict[str, str]]
            Input and filter values from the user.
        outputs : Dict[str, Union[str, Dict[str, str]]]
            Raw outputs from the runs. If `activate_run_selection` is set,
            a Dict[str, str] is returned.

        Returns
        -------
        Union[Component, List[Component]]
            The components must be in the same position as defined in `get_output_layout`.
        """
        return []

    @staticmethod
    def load_mpl_outputs(
        runs: Union[AbstractRun, Dict[str, AbstractRun]],
        inputs: Dict[str, Dict[str, str]],
        outputs: Dict[str, Union[str, Dict[str, str]]],
    ) -> Union[Component, List[Component]]:
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed `inputs` are cleaned and therefore differs compared to `load_inputs` or
        `load_dependency_inputs`. Inputs are cleaned s.t. only the first value is used.
        Also, boolean values are casted to booleans.

        Parameters
        ----------
        runs : Union[AbstractRun, Dict[str, AbstractRun]]
            All selected runs. If `activate_run_selection` is set, only the selected run is
            returned.
        inputs : Dict[str, Dict[str, str]]
            Input and filter values from the user.
        outputs : Dict[str, Union[str, Dict[str, str]]]
            Raw outputs from the runs. If `activate_run_selection` is set,
            a Dict[str, str] is returned.

        Returns
        -------
        Union[Component, List[Component]]
            The components must be in the same position as defined in `get_output_layout`.
        """
        return []

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed `inputs` are cleaned and therefore differs compared to `load_inputs` or
        `load_dependency_inputs`. Inputs are cleaned s.t. only the first value is used.
        Also, boolean values are casted to booleans.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs : Dict[str, Any]
            Input data.

        Returns
        -------
        Dict[str, Any]
            Serialized dictionary.
        """
        return {}

    @classmethod
    def generate_outputs(
        cls, runs: Union[AbstractRun, List[AbstractRun]], inputs: Dict[str, Any]
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Check whether run selection is active and accepts either one or multiple runs at once.

        Calls `process` internally.

        Parameters
        ----------
        runs : Union[AbstractRun, List[AbstractRun]]
            Run or runs to process.
        inputs : Dict[str, Any]
            Input data. Only "real" inputs (not "filter" inputs) are necessary.

        Returns
        -------
        Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
            Returns a data dictionary with the same outputs as `process`.
            If `activate_run_selection` is set, a Dict[str, Dict[str, Any]] is returned. The first
            dictionary is keyed by the `run.id`.
        """
        if cls.activate_run_selection:
            if isinstance(runs, AbstractRun):
                return cls.process(runs, inputs)
            else:
                raise RuntimeError(
                    "The method `generate_outputs` accepts only one run because"
                    "`activate_run_selection` is set."
                )
        else:
            if not isinstance(runs, list):
                if not isinstance(runs, AbstractRun):
                    raise RuntimeError(
                        "The method `generate_outputs` accepts either one or multiple runs."
                    )

                runs = [runs]

            outputs = {}
            for run in runs:
                outputs[run.id] = cls.process(run, inputs)

            return outputs

    def generate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate inputs for the `process` and `load_outputs` required for api mode.

        The arguments are validated against the input schema.

        Note
        ----
        Arguments are only available at runtime. Therefore, no api can be shown beforehand.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            The inputs for the run.

        Raises
        ------
        ValueError
            If an unknown input is passed.
            If an input is missing.
        """
        mapping = {}
        for id, attribute, *_ in self.inputs:
            # Since `self.inputs` is ordered, the first occurring attribute is used and
            # the id is added, so it is not used again.
            if id not in mapping:
                mapping[id] = attribute

        inputs: Dict[str, Dict[str, Any]] = {}
        for k, v in kwargs.items():
            if k in mapping:
                if k not in inputs:
                    inputs[k] = {}

                inputs[k] = v
            else:
                if len(mapping) > 0:
                    raise ValueError(
                        f"Unknown input: {k}. Valid inputs are: {', '.join(list(mapping.keys()))}."
                    )
                else:
                    raise ValueError(f"Unknown input: {k}. No input needs to be passed.")

        if mapping.keys() != inputs.keys():
            raise ValueError(
                f"Missing inputs: {', '.join(list(set(mapping.keys()) - set(inputs.keys())))}."
            )

        return inputs
