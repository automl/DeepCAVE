from urllib.parse import urlparse
from typing import Optional

from .converters import available_converters

from ..storage_backend import AbstractStorage


def get_converter(converter_name: str, tracking_uri: str, study: Optional[str] = None) -> AbstractStorage:
    """
    Interface for converters. Abstracts the selection and initialization process away. Returns the correct converter
    as an instance of AbstractStorage.

    Parameters
    ----------
    converter_name
        str. Name of the converter. Every Converter has a unique name, based on which it is selected.
    tracking_uri
        str. The tracking_uri, with which the storage is initialized.
    study
        str. The name of the study with which the storage is initialized. Can be None, when information about all
            available studies is needed.

    Returns
    -------
        An instance of AbstractStorage.
    """
    parsed_url = urlparse(tracking_uri)

    if converter_name not in available_converters:
        raise KeyError(f'converter_name {converter_name} not in available converters'
                       f'Available converters are {available_converters.keys()}')
    if available_converters[converter_name].scheme() != parsed_url.scheme:
        raise ValueError(f'Not matching Converter ({converter_name}) with scheme ({parsed_url.scheme})'
                         f'expected {available_converters[converter_name].scheme()} as scheme')

    return available_converters[converter_name](study, tracking_uri)
