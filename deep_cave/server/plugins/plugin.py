from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Union, Optional, Tuple
import os
import json
from collections import defaultdict

import pandas as pd

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_dangerously_set_inner_html
from plotly.graph_objects import Figure
import dash_html_components as html
from dash.development.base_component import Component
import dash_table
from ConfigSpace import ConfigurationSpace

from ..server import app
from ..helper import get_study_data
from ..config import models_location
from deep_cave.util.util import encode_data
from deep_cave.util.logs import get_logger
from deep_cave.util.parsing import deep_cave_data_decoder
from deep_cave.registry import get_surrogate
from deep_cave.registry.onnx_surrogate import ONNXSurrogate # only import for typing


logger = get_logger(__name__)


class Plugin(ABC):
    """
    Abstract Plugin class. A (analysis-)plugin is used to create a new analysis-tool in the UI.
    It is basically a wrapper for any python code. Through a uniform interface any kind of library can be
    integrated into the server.

    It is necessay to implement the method process and ui_elements. There are other methods and properties to
    further control the behaviour of the plugin.
    Other methods
    - ui_customization: Customize ui_elements based on available data
    Other properties
    - name: Change the display name
    - category: [missing] change the group the plugin is placed inside the UI
    - tooltip: Shortly explain what the method is doing
    - has_single_output: The plugin declares that it can analyze multiple trials in one call of process.
    - default_output: Controls what is shown, when process hasn't run yet
    - wip: [missing] A flag indicating the completeness of implementation.

    The general life cycle of a plugin is the following:
    1. The plugin is registered with the PluginManager.
    2. The PluginManager calls ui_elements and instantiates the dash Components with the id (generated
       automatically) of this plugin and registers it to the UI.
    3. The PluginManager registers the callback function process with the UI. Adding any ui_elements to the callback.
    4. The server is started.
    5. When a user opens the site, the ui_elements are displayed.
    6. When the user selects a study, the ui_customization is called. The customization defines the changes to the UI
        based on the data provided. The UI is updated according to the customization.
    7. When the user interacts with the ui_elements the Components act accordingly.
    8. The user or an external event (like new data becoming available) can trigger the analysis of the study.
    9. The data from the study, together with the user input are provided to the process method. The method performs
        the analysis and returns an UI component as a result.
    10. The result is displayed to the user.
    11. When the user ends the session the plugins state is reset.

    Up to this point the 'data' given to process and ui_customization were relatively abstract. The following
    passage deals with the content in more detail.

    On each call to process all available data is handed to process. If certain information isn't need it should be
    left to **kwargs, to prevent an error due to the argument not defined in process.

    Kwargs that are always present are:
    - data: A pandas.DataFrame with all available trials. The columns are categorized with four prefixes.
        - 'trial.': Contains the three elements to make a trial unique. The config_id, fidelity and an increment or seed
        - 'config.': The features or configurations used to train the model
        - 'metrics.': The metrics produced by this trial
        - 'trial_meta.': Additional meta data produced by the trial, e.g. tags, duration, start and end
        Also each trial has a unique trial_id, generated form the three components of a trial (config_id, fidelity,
        increment)
    - meta: A dictionary containing meta data about the study itself. The meta can contain but mustn't the following
        keys:
        - optimizer_args: Arguments used to run the AutoML optimizer
        - study_name: Name of the study.
        - objective: The metric used by the AutoML optimizer for evaluation of a model.
        - search_space: A ConfigSpace.ConfigurationSpace. For more information,
         see https://github.com/automl/ConfigSpace
        - start_time: Start time of the study.
        - end_time: End time of the study.
        - duration: The duration of the study.
        - groups: Additional key-value pairs used for grouping.
        Depending on the back_end there are also the keys
        - metrics: Lists all columns in data with the 'metrics.' prefix
        - config: Lists all columns in data with the 'config.' prefix
    - models: A dictionary containing meta data about the models logged for different fidelities.
        The key is always the fidelity. If a model was logged, then there is always the 'default' fidelity available.
        The value is has the another dict with the key-value pairs:
            - mapping: A discription of how the features ('config.') in data should be mapped to the model
            - model_id: The unique identifier of the logged model
            - format: format
        Those values shouldn't be of concern to the user. The surrogate model can be obtained by simply calling:
            self.get_surrogate_model(models, fidelity)
        with the desired fidelity, available as key in models.
    Next there are the values obtained from ui_elements. The key in kwargs is the same key defined in ui_elements.
    The value is the value returned from the dash Component.

    The user can group studies by their group-keys defined in meta['groups']. If this is the case studies with the
    same group key-value pair will be merged and send to process as one study. If the are different key-value pairs
    identical pairs are again merge, but different key-value pairs are passed to process separately.
    If the property has_single_output is set to False, the maintainer of the plugin doesn't have to consider this case,
    since the different groups are passed separately to process and their output is append.
    If the property has_single_output is set to True, then the input the process method changes (not to customize_ui,
    which only receives the first selected study). 'data', 'meta' and 'models' will be embedded into a dictionary,
    with the key being the study_name. The value is identical to the description above. The process method is
    expected to handle multiple studies at once and process one output.
    """

    prefix = 'plugin'
    allowed_outputs = Union[Component, Figure, str, dash_table.DataTable]

    @staticmethod
    @abstractmethod
    def ui_elements() -> Dict[str, Type[Component]]:
        """
        Defines additional UI elements that can be used to deliver user input to the process method.

        Returns
        -------
        Dict with key being the name of the UI element and value the dash_core_components.Component. Use the class
        and not an instance.
        For more information on components, see https://dash.plotly.com/dash-core-components

        The key is used for later reference in ui_customization and process.
        """
        return {}

    @staticmethod
    def ui_customization(**kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Based on the available kwargs specify the Components parameters.

        Parameters
        ----------
        kwargs
            Dict. Content of kwargs are 'data', 'meta', 'models'
            (for more information on the contents of these keys see class description above).

        Returns
        -------
        Return a dictionary with the keys matching the ones defined in ui_elements. The value is itself a dictionary.
        The content of those dictionary are forwarded as **kwargs to the dash Component defined in ui_elements.

        For more on the kwargs available to each Component consult https://dash.plotly.com/dash-core-components.
        """
        return {}

    @staticmethod
    @abstractmethod
    def process(**kwargs) -> allowed_outputs:
        """
        Define the analysis process of the plugin

        Parameters
        ----------
        kwargs
            Dict. Content of kwargs are 'data', 'meta', 'models' and any additional user input defined in ui_elements
            (for more information on the contents of these keys see class description above).

        Returns
        -------
        Return any of the following:
        - dash Graph
        - plotly Figure
        - dash component, like dash_table
        - string, that can be interpreted as HTML
        The content will be handled by internal logic and displayed to the user.
        """
        pass

    @property
    def name(self) -> str:
        """
        Specify the displayed name of the plugin. Default is the name of the class.

        Returns
        -------
            str. The name of the plugin.
        """
        return self.__class__.__name__

    @property
    def category(self) -> str:
        """
        Define the category of the plugin. Not yet used

        Returns
        -------
            str. The output defines the category of the plugin.
        """
        return ''

    @property
    def tooltip(self) -> str:
        """
        Shortly describe what the plugin is doing. The tooltip is displayed on mouseover to the user.

        Returns
        -------
            str. A text that shortly describes the plugin.
        """
        return 'Tooltip missing. Please contact maintainer'

    @property
    def has_single_output(self) -> bool:
        """
        Sets the flag for has_single_output. For the implications on process see the class description.

        Returns
        -------
            bool.
        """
        return False

    @property
    def default_output(self) -> allowed_outputs:
        """
        Define what is displayed in the UI before process was called.

        Returns
        -------
            A valid dash Component instance.
        """
        return dcc.Graph()

    @property
    def wip(self) -> bool:
        """
        Doesn't do anything

        Returns
        -------
            bool. Flag indicate work in progress with True.
        """
        return True

    def get_ui_id(self, name: str) -> str:
        """
        Helper method. From the plugin id and the name of the ui element it generates an id for the element

        Parameters
        ----------
        name
            str. Name of the ui element
        Returns
        -------
            str. The id for the ui element used as a reference in the dash ui.
        """
        return self.id + '-' + name

    def get_output_id(self) -> str:
        """
        Helper method, that generates the id by which the output element in the dash ui is referenced

        Returns
        -------
            str. Returns a string with the id used in the dash ui for the output element
        """
        return self.id + '-output'

    def __init__(self):
        # todo clean up
        self.id = self.prefix + '_' + self.__class__.__name__
        self.trigger_value = 0

        self.output_element = Output(self.get_output_id(), 'children')

        self.data_elements = []

        self.trigger_ids = ['trigger-' + self.id]
        self.trigger_elements = [Input(trigger_id, 'n_clicks') for trigger_id in self.trigger_ids]
        # add selected rows as trigger. Changing selected rows clears plots
        self.trigger_elements.append(Input('table_of_studies', 'derived_virtual_selected_row_ids'))
        self.trigger_ids.append('studies')
        # create a complete list of states. Including: User input and data input
        self.state = []
        # include data
        self.state.extend(self.data_elements)
        self.state.append(State('table_of_studies', 'selected_columns'))
        # include ui_elements
        ui_copy = self.ui_elements().copy()
        for k, v in ui_copy.items():
            v = v(id=self.get_ui_id(k))
            self.state.append(State(v.id, 'value'))

        # create a list of keys. They have the same ordering as presented in the callback
        # the list is later zipped with args to form kwargs and hand it over to process(**kwargs)
        self.keys = []
        # first register all triggers
        self.keys.extend(self.trigger_ids)
        self.keys.append('groupby')
        # then register all data sources
        # self.keys.append('studies')
        # then add all user inputs
        self.keys.extend([keys for keys in self.ui_elements().keys()])

    def register_callback(self) -> None:
        """
        Method called by PluginManager to register the single callback of this plugin.

        The method could be extended to increase the interactivity, but it is not recommended, since it is not
        standardized for all plugins.

        The method generates a function and sets the single output element, all trigger and state elements set
        in the constructor of the plugin.

        Returns
        -------
            None.
        """
        # todo register callback should get app through parameters and not through importing
        app.callback(
            self.output_element,
            self.trigger_elements,
            self.state,
            prevent_initial_call=True
        )(self)

    def generate_ui(self, customization: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Component]:
        """
        Called by PluginManager to instantiate the ui elements.

        Parameters
        ----------
        customization
            Dict. If a customization dict is provided, the plugins are called with the provided arguments.
            If it is None, the ui elements will be initialized only with an id.
        Returns
        -------
            List of instantiated components
        """
        elements = []
        ui_copy = self.ui_elements().copy()
        for k, v in ui_copy.items():
            if customization and k in customization:
                v = v(id=self.get_ui_id(k), **customization[k])
            else:
                v = v(id=self.get_ui_id(k))
            # add a label with name/description of the input field
            elements.append(html.Label(k))
            elements.append(v)
        return elements

    def __call__(self, *args) -> Union[Union[Component, dash_table.DataTable],
                                       List[Union[Component, dash_table.DataTable]]]:
        """
        The class itself is registered to the dash callback function. If one of the triggers is used, the
        class will be called with the args. The are defined by the trigger and state elements.
        The __call__ is a wrapper around process. It converts the the args into kwargs for more flexibility
        for the user.
        The __call__ will also load the referenced data via get_study_data().
        Based on the configuration the process method is called once will all the data or multiple times.
        The output is returned to dash, to be displayed.

        Parameters
        ----------
        args
            List of inputs, previously defined during the construction of the plugin.

        Returns
        -------

        """
        # dash provides inputs to callback functions as args
        # convert it to kwargs
        kwargs = dict(zip(self.keys, args))
        # check if button-trigger was pressed or the selected row was updated
        if kwargs['trigger-' + self.id] <= self.trigger_value:
            # if the button wasn't pressed then set the default_output
            return self.default_output
        else:
            # if the button was pressed then update state and display results
            self.trigger_value = kwargs['trigger-' + self.id]

        studies = kwargs['studies'].copy()
        groupby = kwargs['groupby'].copy()
        # remove the trigger. No value for the user
        del kwargs[self.trigger_ids[0]]
        # remove studies, it gets replaced with data, meta and models
        del kwargs['studies']
        del kwargs['groupby']

        if groupby:
            grouped = defaultdict(list)
            for study in studies:
                data, meta_data, models = get_study_data(study)
                key = []
                for group in groupby:
                    key.append(meta_data[group])
                grouped[tuple(key)].append(study)
            if self.has_single_output:
                return self._grouped_single_output(grouped, kwargs)
            else:
                return self._grouped_multi_output(grouped, kwargs)
        else:
            if self.has_single_output:
                # if multiselect is supported, then the plugin will build one graph with all
                # information in it
                return self._non_grouped_single_output(studies, kwargs)
            else:
                # if multiselect is not supported, then call the process method multiple times
                # and append the outputs
                return self._non_grouped_multi_output(studies, kwargs)

    def get_surrogate_model(self, models: Dict, fidelity: Union[float, str] = None) -> Union[ONNXSurrogate, None]:
        """
        Helper method for the user, to load the selected model. The models dict is available to every process call.
        fidelity can be a number, or a string, referencing the fidelity. The string default should always be
        available if a least one model was uploaded to the registry.

        Parameters
        ----------
        models
            Dict. Of models with its meta data, like fidelity and mapping.
        fidelity
            str or float. Referencing one of the fidelities in models.
        Returns
        -------
            ONNXSurrogate model, or None, if the operation failed for any reason
        """
        model_meta = self._resilient_model_meta_selection(models, fidelity)
        if model_meta is None:
            return None
        return get_surrogate(models_location, model_meta['model_id'], model_meta['mapping'])

    @staticmethod
    def _composite_loading(group: List[str], studies: List) -> Tuple[List[str], pd.DataFrame, Dict, Dict]:
        """
        Helper function for _grouped_ methods. Loads all specified Studies.
        Also holds the logic on how to combine the data.

        Current implementation assumes, that grouped data is similar. DataFrames are appended and Dicts are updated.
        Meaning the next study overwrites the keys in the previous study. Assuming that the meta data is mostly the
        same, it at least guarantees that all keys from the studies are present. This behaviour makes it much easier
        for the user since the data has the same format as in non grouped state.

        Parameters
        ----------
        group
            str. The key referencing the groups.
        studies
            List of studies associated with this groupd.

        Returns
        -------
            The appended DataFrame and a single meta and model meta dict
        """
        df_data, meta_data, models_meta = None, None, None
        if isinstance(studies, list):
            for study in studies:
                data, meta, models = get_study_data(study)
                if df_data is None:
                    df_data = data
                else:
                    df_data.append(data)
                # todo fix
                if meta_data:
                    meta_data.update(meta)
                else:
                    meta_data = meta
                # todo fix
                if models_meta:
                    models_meta.update(models)
                else:
                    models_meta = models
            return group, df_data, meta_data, models_meta
        else:
            raise Exception('Something went wrong. _composite_loading shouldn\'t be called with a single study'
                            'not in a list.')

    def _grouped_multi_output(self, grouped: Dict[List[str], List[str]], kwargs: Dict) -> List[allowed_outputs]:
        """
        Helper method. Loaded grouped data and calls process once for each group. Hence multiple outputs will be
        generated.

        Parameters
        ----------
        grouped
            Dict. Contains the group keys with the studies associated with this group.
        kwargs
            The kwargs handed down from __call__ to be used in process.
        Returns
        -------
            Returns a list of allowed output objects.
        """
        outputs = []
        for group, studies in grouped.items():
            virtual_study, kwargs['data'], kwargs['meta'], kwargs['models'] = self._composite_loading(group, studies)

            logger.info(f'{self.__class__.__name__} Calling process(**kwargs) with kwargs: {kwargs.keys()}')
            return_val = self.process(**kwargs)
            # depending on return value automate some aspects
            outputs.append(html.Label(virtual_study))
            outputs.append(self._interpret_output(return_val))
        return outputs

    def _grouped_single_output(self, grouped: Dict[List[str], List[str]], kwargs: Dict) -> allowed_outputs:
        """
        Similar to _grouped_multi_output, with the exception that the process method is called once with all
        the data. The plugin has to handle the data as Dict, with the keys being the associated group.

        Parameters
        ----------
        grouped
            Dict. Contains the group keys with the studies associated with this group.
        kwargs
            The kwargs handed down from __call__ to be used in process.
        Returns
        -------
            Returns one object of allowed output.
        """
        kwargs['data'], kwargs['meta'], kwargs['models'] = {}, {}, {}
        for group, studies in grouped.items():
            virtual_study, data, meta, models = self._composite_loading(group, studies)
            kwargs['data'][virtual_study], kwargs['meta'][virtual_study], kwargs['models'][
                virtual_study] = data, meta, models

        logger.info(f'{self.__class__.__name__} Calling process(**kwargs) with kwargs: {kwargs.keys()}')
        return_val = self.process(**kwargs)
        # depending on return value automate some aspects
        return self._interpret_output(return_val)

    def _non_grouped_multi_output(self, studies: List[str], kwargs: Dict) -> List[allowed_outputs]:
        """
        Called when multiple studies are selected, but no key to group by.
        This helper method handles multi output. The process method is called once for every study.

        Parameters
        ----------
        studies
            List of studies as strings.
        kwargs
            The kwargs handed down from __call__ to be used in process.
        Returns
        -------
            Returns list of allowed_outputs
        """
        outputs = []
        for study in studies:
            data, meta, models = get_study_data(study)

            kwargs['data'] = data
            kwargs['meta'] = meta
            kwargs['models'] = models

            logger.info(f'{self.__class__.__name__} Calling process(**kwargs) with kwargs: {kwargs.keys()}')
            return_val = self.process(**kwargs)
            # depending on return value automate some aspects
            outputs.append(html.Label(study))
            outputs.append(self._interpret_output(return_val))
        return outputs

    def _non_grouped_single_output(self, studies, kwargs) -> allowed_outputs:
        """
        Called when multiple studies are selected, but no key to group by.
        This helper method handles single output. The process method is called once with data, meta and models,
        being a dict, with keys being the study name.

        Parameters
        ----------
        studies
            List of studies as strings.
        kwargs
            The kwargs handed down from __call__ to be used in process.
        Returns
        -------
            Returns one object of allowed_outputs
        """
        kwargs['data'], kwargs['meta'], kwargs['models'] = {}, {}, {}

        for study in studies:
            data, meta, models = get_study_data(study)

            kwargs['data'][study] = data
            kwargs['meta'][study] = meta
            kwargs['models'][study] = models

        logger.info(f'{self.__class__.__name__} Calling process(**kwargs) with kwargs: {kwargs.keys()}')
        return_val = self.process(**kwargs)
        # depending on return value automate some aspects
        return self._interpret_output(return_val)

    def _interpret_output(self, return_val: allowed_outputs) -> Union[Component, dash_table.DataTable]:
        """
        Helper method for all _output functions.
        Makes sure the output of process is correct and converts it if necessary, so that everything is valid
        dash Component that can be displayed

        Parameters
        ----------
        return_val
            One of the allowed output data types

        Returns
        -------
            A dash Component (or DataTable)
        """
        if isinstance(return_val, dcc.Graph):
            return return_val
        if isinstance(return_val, Figure):
            return dcc.Graph(figure=return_val)
        if isinstance(return_val, Component):
            return return_val
        if isinstance(return_val, str):
            logger.warning(f'{self.name} Setting string as HTML Object')
            return dash_dangerously_set_inner_html.DangerouslySetInnerHTML(return_val)
        else:
            error = f'{self.name} returns invalid type {type(return_val)},' \
                    f' expected [string, dcc.Graph, dash.graph_objects.Figure, dcc.Component]'
            logger.exception(error)
            raise TypeError(error)

    def _resilient_model_meta_selection(self, models: Dict, fidelity: Union[str, float]) -> Union[Dict, None]:
        """
        Helper method for get_surrogate_model. Handles any kind of problem that could arise from not matching
        models content with the requested fidelity.

        Parameters
        ----------
        models
            Dict. The models meta dict from the process call
        fidelity
            An identifier for the requested fidelity matching the keys in models.
        Returns
        -------
            Dict. Returns the matching entry in models or None, if non is available.
        """
        if fidelity is None:
            fidelity = 'default'
        if models is None:
            logger.warning(f'Plugin {self.name} found no models. Returning None')
            return None
        if fidelity not in models:
            logger.warning(f'{self.name} didn\'t find selected fidelity {fidelity}. Falling back to default')
            if 'default' not in models:
                logger.warning(f'Plugin {self.name} has a dict in models but no selected fidelity {fidelity} or'
                               f'default fidelity. Returning None')
                return None
            return models['default']
        else:
            # everything works fine
            return models[fidelity]

    @staticmethod
    def encode_data(data: pd.DataFrame, cs: ConfigurationSpace = None) -> pd.DataFrame:
        """
        A helper method, that calls the helper function encode_data. When a ConfigSpace is provided the
        data will be converted based on those rules. Otherwise the class of data (numerical, ordinal, nominal) will
        be derived from the data type.

        For information see the documentation of encode_data.

        Parameters
        ----------
        data
            pd.DataFrame. A data DataFrame as provided by a process call.
        cs
            ConfigurationSpace. A ConfigurationSpace object that matches the specified data.

        Returns
        -------
            pd.DataFrame. Returns a new DataFrame with nominal values in OneHot encoding, ordinal values in Ordinal
            encoding and numerical values cast into the correct datatype.
        """
        return encode_data(data, cs)
