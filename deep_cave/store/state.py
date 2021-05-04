from typing import Any
import atexit
import os

from ..store.store import Store

"""
This file (module) manages the state of the logger object. Confusingly the file is located in the store,
because the logger is one interface to the store and the store should manage everything state related,
so that everything else, stays idempotent.

The state is inspired by MLflow, where the logger keeps track of the state.
The user can create a new study. The logger then assumes that every trial belongs to this new study.

Trials are not managed this way, to provide more thread safety. For more see store/trial.py
"""


_study = os.environ.get('STUDY', None)
_tracking_uri = os.environ.get('TRACKING_URI', '.')
_storage = None
# per default, registry and tracking uri are the same
_registry_uri = os.environ.get('REGISTRY_URI', _tracking_uri)

"""
The getter and setter functions for these global objects
"""


def get_study() -> [str, None]:
    """
    Getter for the name/id of the current study.

    Returns
    -------
        Returns string, if a study already exists, otherwise return None.
    """
    return _study


def set_study(s: [str, None]):
    """
    Setter for the name/id of the current study.

    Returns
    -------
        None
    """
    global _study
    _study = s


def get_tracking_uri() -> str:
    """
    Getter for the tracking_uri of the logger.

    Returns
    -------
        Returns string, if a uri is already defined, otherwise return None.
    """
    return _tracking_uri


def set_tracking_uri(s: str):
    """
    Setter for the tracking_uri of the logger.

    Returns
    -------
        None
    """
    global _tracking_uri
    _tracking_uri = s


def get_registry_uri():
    """
    Getter for the registry_uri of the logger.

    Returns
    -------
        Returns string, if a uri is already defined, otherwise return None.
    """
    return _registry_uri


def set_registry_uri(s: str):
    """
    Setter for the registry_uri of the logger.

    Returns
    -------
        None
    """
    global _registry_uri
    _registry_uri = s


def get_storage() -> [Store, None]:
    """
    Getter for the store current store object.

    Returns
    -------
        Returns a store object, if a study already exists, otherwise return None.
    """
    return _storage


def set_storage(s: [Store, None]):
    """
    Setter for the store current store object. Either set a new store as the global store object, or
    set the store object to None. For example, if the study has ended and the store object should be discarded,
    to avoid overwriting the data with other data.

    Returns
    -------
        None
    """
    global _storage
    if _storage is not None:
        atexit.unregister(_storage.end_study)
    if s is not None:
        atexit.register(s.end_study)

    _storage = s