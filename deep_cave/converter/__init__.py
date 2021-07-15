import os

from deep_cave.util.importing import auto_import_iter
from deep_cave.converter.converter import Converter


converters = {}

paths = [os.path.join(os.path.dirname(__file__), '*')]
for name, obj in auto_import_iter("converter", paths):
    if not issubclass(obj, Converter):
        continue
    # Plugin itself is a subclass, filter it out
    if obj == Converter:
        continue

    converters[obj.name()] = obj



