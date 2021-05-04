from typing import Optional
from urllib.parse import urlparse

from .abstract_storage import AbstractStorage
from .storage_backends import available_backends


def infer_storage_backend(tracking_uri: str, study: Optional[str] = None) -> AbstractStorage:
    """
    Interface for the storage_backend. Selects the correct storage backend to return.

    Given a URI the function infers the schema and selects the backend storage.
    Study doesn't have to be specified. In this case the storage_backend will only give information about
    study meta data.

    Parameters
    ----------
    tracking_uri
        str. URI pointing to the directory where all studies are located.
    study
        str. If a study should be loaded
    Returns
    -------
        An instance of AbstractStorage.
    """
    parsed_url = urlparse(tracking_uri)
    try:
        backend = available_backends[parsed_url.scheme](study, tracking_uri)
    except KeyError as e:
        print(e)
        raise NotImplementedError('backend ' + parsed_url.scheme + ' is not yet supported')
    return backend
