import sys

from deepcave.__author__ import __author__
from deepcave.__version__ import __version__
from deepcave.runs.objective import Objective
from deepcave.runs.recorder import Recorder

if sys.version_info < (3, 9):
    raise ValueError(
        "Unsupported python version %s found. DeepCAVE requires Python "
        "3.9 or higher." % sys.version_info
    )

version = __version__
author = __author__
_exec_file = sys.argv[0]

if "server.py" in _exec_file or "worker.py" in _exec_file:
    from deepcave.config import configs
    from deepcave.queue import Queue  # noqa
    from deepcave.server import get_app  # noqa
    from deepcave.utils.cache import Cache  # noqa
    from deepcave.utils.run_caches import RunCaches  # noqa

    config = configs["default"]
    app = get_app()
    queue = Queue(config.REDIS_URL)

    # Meta cache
    c = Cache(filename=config.CACHE_DIR / "meta.json", defaults=config.META_DEFAULT)

    # Run caches
    rc = RunCaches(config.CACHE_DIR)

    __all__ = ["version", "app", "queue", "c", "rc", "config", "Recorder", "Objective"]
else:
    __all__ = ["version", "Recorder", "Objective"]
