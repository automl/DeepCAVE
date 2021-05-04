import os

from deep_cave.util.logs import get_logger
from deep_cave.util.importing import auto_import_iter
from deep_cave.storage_backend.abstract_storage import AbstractStorage

logger = get_logger(__name__)


available_backends = {}

paths = [os.path.join(os.path.dirname(__file__), '*')]


for name, obj in auto_import_iter(paths):
    if not issubclass(obj, AbstractStorage):
        continue
    # Plugin itself is a subclass, filter it out
    if obj == AbstractStorage:
        continue
    available_backends[obj.scheme()] = obj

# the only object available from this module is the plugins dictionary
__all__ = ['available_backends']
