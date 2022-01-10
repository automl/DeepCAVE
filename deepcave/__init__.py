import sys

from deepcave.__version__ import __version__
from deepcave.runs.objective import Objective
from deepcave.runs.recorder import Recorder

version = __version__
exec_file = sys.argv[0]

if "server.py" in exec_file or "worker.py" in exec_file:
    from deepcave.utils.cache import Cache  # noqa
    from deepcave.utils.run_caches import RunCaches  # noqa
    from deepcave.server import get_app  # noqa
    from deepcave.queue import Queue  # noqa
    from deepcave.config import CONFIG, META  # noqa

    app = get_app()
    queue = Queue(CONFIG["REDIS_URL"])

    # Meta cache
    c = Cache(
        filename=CONFIG["CACHE_DIR"] / "meta.json",
        defaults=META)

    # Run caches
    rc = RunCaches()

    __all__ = ["version", "app", "queue", "c", "rc", "Recorder", "Objective"]
else:
    __all__ = ["version", "Recorder", "Objective"]
