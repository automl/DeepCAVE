import datetime
import sys
import os

name = "DeepCAVE"
package_name = "deepcave"
author = "René Sass and Marius Lindauer"
author_email = "sass@tnt.uni-hannover.de"
description = "An interactive framework to visualize and analyze your AutoML process in real-time."
url = "automl.org"
project_urls = {
    "Documentation": "https://automl.github.io/DeepCAVE/main",
    "Source Code": "https://github.com/automl/deepcave",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, René Sass and Marius Lindauer"
version = "0.4"

_exec_file = sys.argv[0]
_exec_files = ["server.py", "worker.py", "sphinx-build"]


if any(file in _exec_file for file in _exec_files):
    from deepcave.config import config
    from deepcave.queue import Queue  # noqa
    from deepcave.runs.handler import RunHandler  # noqa
    from deepcave.runs.objective import Objective  # noqa
    from deepcave.runs.recorder import Recorder  # noqa
    from deepcave.server import get_app  # noqa
    from deepcave.utils.cache import Cache  # noqa
    from deepcave.utils.run_caches import RunCaches  # noqa
    from deepcave.utils.notification import Notification

    app = get_app()
    queue = Queue(config.REDIS_ADDRESS, config.REDIS_PORT)

    # Meta cache
    c = Cache(filename=config.CACHE_DIR / "meta.json", defaults=config.META_DEFAULT)

    # Set working directory to current directory
    if c.get("working_dir") is None:
        c.set("working_dir", value=os.getcwd())

    # Run caches
    rc = RunCaches(config)

    # Run Handler
    run_handler = RunHandler(config, c, rc)

    # Notifications
    notification = Notification()

    __all__ = [
        "version",
        "app",
        "queue",
        "c",
        "rc",
        "run_handler",
        "notification",
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
