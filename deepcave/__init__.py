# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# DeepCAVE

This module is used to initialize and set up the configuration for the DeepCAVE framework.

The Dash application gets created.
"""

from typing import Any, Callable, TypeVar, cast

import datetime
import os
import sys
import warnings
from functools import wraps
from pathlib import Path

name = "DeepCAVE"
package_name = "deepcave"
author = (
    "S. Segel and H. Graf and E. Bergman and K. Thieme and L. Fehring and A. Tornede and "
    "R. Sass and A. Biedenkapp and F. Hutter and M. Lindauer"
)
author_email = "l.fehring@ai.uni-hannover.de"
description = "A Visualization and Analysis Tool for Automated Machine Learning."
url = "automl.org"
project_urls = {
    "Documentation": "https://automl.github.io/DeepCAVE/main",
    "Source Code": "https://github.com/automl/deepcave",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, {author}"
version = "1.3.2"

_exec_file = sys.argv[0]
_exec_files = ["server.py", "worker.py", "sphinx-build"]

ROOT_DIR = Path(__file__).parent


def get_app(title: str) -> Any:
    """
    Get the Dash Proxy.

    Parameters
    ----------
    title : str
        The title of the application.

    Returns
    -------
    DashProxy
        The dash proxy.
    """
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
    from deepcave.runs.objective import Objective
    from deepcave.runs.recorder import Recorder
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

        # Supress warnings during initializing run handler to avoid showing warnings with respect
        # to previously loaded runs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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
        from deepcave.utils.configs import parse_config
        from deepcave.utils.notification import Notification

        config_name = None
        if "--config" in sys.argv:
            config_name = sys.argv[sys.argv.index("--config") + 1]
        config = parse_config(config_name)

        # Notifications
        notification = Notification()

        __all__ = ["version", "Recorder", "Objective", "notification", "config"]
    except ModuleNotFoundError:
        __all__ = ["version"]


_api_mode = False if "app" in globals() else True

# This TypeVar is necessary to ensure that the decorator works with arbitrary signatures.
F = TypeVar("F", bound=Callable[..., Any])


def interactive(func: F) -> F:
    """
    Define the interactive decorator.

    Parameters
    ----------
    func : F
        The function to be decorated.

    Returns
    -------
    F
        The decorated function.
    """

    @wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Any:
        """
        Inner function of the decorator.

        Parameters
        ----------
        *args : Any
            Arguments to be passed to the wrap function.
        **kwargs : Any
            Keyword arguments to be passed to the wrap function.

        Returns
        -------
        Any
            The result of the function.
        """
        if _api_mode:
            return

        return func(*args, **kwargs)

    return cast(F, inner)
