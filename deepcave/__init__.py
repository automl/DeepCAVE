import sys

author = "René Sass and Marius Lindauer"
version = "0.2"


_exec_file = sys.argv[0]

if "server.py" in _exec_file or "worker.py" in _exec_file:
    from deepcave.config import config
    from deepcave.queue import Queue  # noqa
    from deepcave.runs.handler import RunHandler  # noqa
    from deepcave.runs.objective import Objective
    from deepcave.runs.recorder import Recorder
    from deepcave.server import get_app  # noqa
    from deepcave.utils.cache import Cache  # noqa
    from deepcave.utils.run_caches import RunCaches  # noqa

    app = get_app()
    queue = Queue(config.REDIS_ADDRESS, config.REDIS_PORT)

    # Meta cache
    c = Cache(filename=config.CACHE_DIR / "meta.json", defaults=config.META_DEFAULT)

    # Run caches
    rc = RunCaches(config)

    # Run Handler
    run_handler = RunHandler(config, c, rc)

    __all__ = [
        "version",
        "app",
        "queue",
        "c",
        "rc",
        "run_handler",
        "config",
        "Recorder",
        "Objective",
    ]
else:

    try:
        from deepcave.runs.objective import Objective
        from deepcave.runs.recorder import Recorder

        __all__ = ["version", "Recorder", "Objective"]
    except ModuleNotFoundError:
        __all__ = ["version"]
