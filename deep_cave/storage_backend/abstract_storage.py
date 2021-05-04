import os
import json
import pathlib
from typing import Dict, Optional, Union, Tuple
import datetime
from abc import abstractmethod, ABC

import pandas as pd


class AbstractStorage(ABC):
    def __init__(self, study: str, tracking_uri: Union[str, pathlib.Path]):
        """
        AbstractStorage, used as base class for all backend storage classes.

        The purpose of this class is to provide a common interface that abstracts between the internal data
        representation and the way the data is actually persisted.

        This way it is possible to use DeepCAVE to save data as JSON to a filesystem or persist it to a SQL or
        Non-SQL DB. It won't make a difference to the user. Just by specifying a different schema, another backend
        storage will be selected to handle the saving and loading of data.

        A AbstractStorage is embedded inside a store. The store calls the callback functions below to inform the
        storage backend about changes and what these changes do. It is up to the implementation of a subclass of
        AbstractStorage to decide what and when to save the data.

        The AbstractStorage class should be able to handle JSON serializable dicts. Meaning all components of the
        dictionary are either string, float, int, list or dict. The store class will abstract away all more complex
        data types.

        The storage backend is also used by the server to retrieve data. There is one method for each part of the data:
        data, meta, model.
        All three are returned by retrieve_data, which might additional data, like appending the conigs. and metrics.
        columns to the meta data. Without this step, this additional data isn't known to meta data.

        It is currently not enforced, that any method except scheme is implemented by a subclass of AbstractStorage.
        This might lead to a behaviour, where no data is actually persisted, because the backend didn't implement the
        functionality and there is no direct way for the user to know that nothing is being logged.
        On the other hand, all the callbacks are called by the Store and AbstractStorage shouldn't be forced to
        implement behaviours for all callback methods.

        Parameters
        ----------
        study
            str. Name of the study to load
        tracking_uri
            str or path to the directory containing the studies.
        """
        self.tracking_uri = tracking_uri
        self.study = study

    @staticmethod
    @abstractmethod
    def scheme() -> str:
        """
        Returns the scheme this backend storage can handle. E.g. empty string means filesystem, sqlite://[path], means
        the implementation is used for saving and loading, to and from a sqlite file.

        Further separation between storage_backends currently not possible. There can only be one storage backend
        per schema.

        Alternatively, the naming convention in converters can be used.

        Returns
        -------
            Returns a string that matches the scheme
        """
        pass

    def on_trial_start(self, trial_id: str, trial: Dict[str, Dict]) -> None:
        """
        Callback called everytime a new trial starts.

        Parameters
        ----------
        trial_id
            The tripled of trial_id.
        trial
            The content of the trial, available at the beginning of the trial.
        Returns
        -------
            None.
        """
        pass

    def on_trial_end(self, trial_id: str, trial: Dict[str, Dict]) -> None:
        """
        Callback called everytime a trial ends.

        Parameters
        ----------
        trial_id
            The tripled of trial_id.
        trial
            The complete content of the trial.
        Returns
        -------
            None
        """
        pass

    def on_study_start(self, **study_meta) -> None:
        """
        Callback called everytime a study starts.

        Parameters
        ----------
        study_meta
            The content of the study meta data, available at the beginning of the study.
        Returns
        -------
            None.
        """
        pass

    def on_study_end(self, **study_meta) -> None:
        """
        Callback called everytime a study ends.

        Parameters
        ----------
        study_meta
            The complete content of the study meta data.
        Returns
        -------
            None
        """
        pass

    def on_model_log(self, fidelity: Union[str, float], model_entry: Dict) -> None:
        """
        Callback called, when a new model is logged.

        Parameters
        ----------
        fidelity
            The fidelity the model is associated with. Remember a fidelity can get a model assigned multiple times.
        model_entry
            The content of the model meta data.
        Returns
        -------
            None
        """
        pass

    def retrieve_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        By default just returns the composite information retrieved from trials, meta and models.

        This method can be overridden, to add additional information to either of the three data components

        Returns
        -------
            Trial data DataFrame, the meta dict and the model meta dict
        """
        return self.retrieve_trials(), self.retrieve_meta(), self.retrieve_models()

    def retrieve_meta(self) -> Dict:
        """
        Retrieve only the meta data.

        Returns
        -------
            A dictionary conforming to the standard of meta data
        """
        pass

    def retrieve_trials(self) -> pd.DataFrame:
        """
        Retrieve the trial data and convert it into a conform trials DataFrame.

        Returns
        -------
            pd.DataFrame.
        """
        pass

    def retrieve_models(self) -> Dict:
        """
        Retrieve the model meta data and return it as Dict

        Returns
        -------
            Dict, with the model meta data.
        """
        pass

    def get_studies(self) -> Dict[str, Dict]:
        """
        Return information of all studies in tracking_uri.
        Return the name of the studies and the meta information in dict format.

        Returns
        -------
            A dict with keys being the study names and the value being the meta dictionary from retrieve_meta
        """
        # the storage backend has to handle get_studies, because only the backend knows about the structure
        # return a Dict with {study_name: meta_dict}, meta_dict defined as in retrieve_data
        pass

