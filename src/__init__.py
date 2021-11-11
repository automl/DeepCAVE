import os
from src.runs.recorder import Recorder

version = "0.0.1"


if __name__ == "__main__":
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
