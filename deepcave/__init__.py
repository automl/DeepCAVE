from typing import Any, Callable, TypeVar, cast

import datetime
import os
import sys
from functools import wraps
from pathlib import Path

name = "DeepCAVE"
package_name = "deepcave"
author = "R. Sass and E. Bergman and A. Biedenkapp and F. Hutter and M. Lindauer"
author_email = "sass@tnt.uni-hannover.de"
description = "An interactive framework to visualize and analyze your AutoML process in real-time."
url = "automl.org"
project_urls = {
    "Documentation": "https://automl.github.io/DeepCAVE/main",
    "Source Code": "https://github.com/automl/deepcave",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, {author}"
version = "1.1"

_exec_file = sys.argv[0]
_exec_files = ["server.py", "worker.py", "sphinx-build"]

ROOT_DIR = Path(__file__).parent


def get_app(title: str):
    import dash_bootstrap_components as dbc
    from dash_extensions.enrich import (
        DashProxy,
        MultiplexerTransform,
        NoOutputTransform,
        TriggerTransform,
    )

    app = DashProxy(
        __name__,
        title=title,
        update_title="",
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        transforms=[
            # enable use of Trigger objects
            TriggerTransform(),
            # makes it possible to target an output multiple times in callbacks
            MultiplexerTransform(),
            # enable use of ServersideOutput objects
            # ServersideOutputTransform(),
            # enable callbacks without output
            NoOutputTransform(),
            # makes it possible to skip callback invocations while a callback is running
            # BlockingCallbackTransform(),
            # makes it possible to write log messages to a Dash component
            # LogTransform(),
        ],
    )
    return app


if any(file in _exec_file for file in _exec_files):
    from deepcave.custom_queue import Queue
    from deepcave.runs.handler import RunHandler
    from deepcave.runs.objective import Objective  # noqa
    from deepcave.runs.recorder import Recorder  # noqa
    from deepcave.utils.cache import Cache
    from deepcave.utils.configs import parse_config
    from deepcave.utils.notification import Notification
    from deepcave.utils.run_caches import RunCaches

    # Get config
    config_name = None
    if "--config" in sys.argv:
        config_name = sys.argv[sys.argv.index("--config") + 1]
    config = parse_config(config_name)

    # Create app
    app = get_app(config.TITLE)
    queue = Queue(config.REDIS_ADDRESS, config.REDIS_PORT)

    if "server.py" in _exec_file:
        # Meta cache
        c = Cache(
            filename=config.CACHE_DIR / "meta.json",
            defaults=config.META_DEFAULT,
            debug=config.DEBUG,
        )

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
        from deepcave.runs.objective import Objective  # noqa
        from deepcave.runs.recorder import Recorder  # noqa

        __all__ = ["version", "Recorder", "Objective"]
    except ModuleNotFoundError:
        __all__ = ["version"]


_api_mode = False if "app" in globals() else True


# This TypeVar is necessary to ensure that the decorator works with arbitrary signatures.
F = TypeVar("F", bound=Callable[..., Any])


def interactive(func: F) -> F:
    @wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Any:
        if _api_mode:
            return

        return func(*args, **kwargs)

    return cast(F, inner)
