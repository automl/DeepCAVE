import os
import glob
from typing import List
from importlib import import_module
import importlib.util
import sys
import inspect
import collections

from deep_cave.util.logs import get_logger


logger = get_logger(__name__)


def auto_import_iter(module, paths: List[str]):
    for path in paths:
        for f in glob.glob(path):
            if os.path.basename(f).startswith('__'):
                continue
            module_name = f'{module}.' + os.path.basename(f).replace('.py', '')
            # dynamic import
            # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            try:
                spec = importlib.util.spec_from_file_location(module_name, f)
                foo = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = foo
                spec.loader.exec_module(foo)
            except Exception as e:
                logger.exception(f'Problem when loading file {f} as {module_name} from path {path}')
            # iterate module content
            # https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
            # allow only class # inspect.isclass
            for name, obj in inspect.getmembers(sys.modules[module_name], inspect.isclass):
                yield name, obj
