import importlib.util
import inspect
import sys
from pathlib import Path
from typing import List, Iterator, Any

from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


def auto_import_iter(module: str, paths: List[Path]) -> Iterator[tuple[str, Any]]:
    for path in paths:
        logger.debug(f"Searching for files in {path.absolute()}")
        for f in path.iterdir():
            if f.name.startswith('__'):
                continue
            module_name = f'{module}.{f.stem}'

            if "pending" in module_name:
                continue

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
