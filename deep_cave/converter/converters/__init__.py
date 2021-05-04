import os
import collections

from deep_cave.util.logs import get_logger
from deep_cave.util.importing import auto_import_iter
from deep_cave.converter.abstract_converter import AbstractConverter
from deep_cave.server.config import external_converters


logger = get_logger(__name__)

"""
Automated import of all converters available in this module.
"""


available_converters = {}

paths = [os.path.join(os.path.dirname(__file__), '*')]
if external_converters:
    if isinstance(external_converters, list):
        paths.extend(external_converters)
    if isinstance(external_converters, str):
        paths.append(external_converters)


for name, obj in auto_import_iter(paths):
    if not issubclass(obj, AbstractConverter):
        continue
    # Plugin itself is a subclass, filter it out
    if obj == AbstractConverter:
        continue
    available_converters[obj.name()] = obj

# the only object available from this module is the plugins dictionary
__all__ = ['available_converters']
