import os
import sys
from src.runs.recorder import Recorder


version = "0.0.1"
exec_file = sys.argv[0]


if "server.py" in exec_file or "worker.py" in exec_file:
    from src.utils.cache import Cache  # noqa
    from src.utils.run_caches import RunCaches  # noqa
    from src.server import get_app  # noqa
    from src.queue import Queue  # noqa
    from src.config import CONFIG, META  # noqa

    app = get_app()
    queue = Queue(CONFIG["REDIS_URL"])

    # Meta cache
    c = Cache(
        filename=os.path.join(CONFIG["CACHE_DIR"], "meta.json"),
        defaults=META)

    # Run caches
    rc = RunCaches()

    __all__ = ["version", "app", "queue", "c", "rc", "Recorder"]
else:
    __all__ = ["version", "Recorder"]
