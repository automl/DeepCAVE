from typing import Any, List

import multiprocessing
import subprocess
from pathlib import Path

from absl import app, flags

from deepcave.utils.configs import parse_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean("open", False, "Opens DeepCAVE directly in the browser.")
flags.DEFINE_integer(
    "n_workers",
    1,
    "How many workers should be specified. In your case, the maximum number of workers should be "
    f"{multiprocessing.cpu_count() - 1}.",
)
flags.DEFINE_boolean("docker", False, "Uses docker image to start DeepCAVE. Not supported yet.")
flags.DEFINE_string("config", None, "Filename to a user-specific config.")
flags.DEFINE_string(
    "get_config_value", None, "Prints the value of a given config key. Useful for bash scripts."
)


def execute(_) -> None:
    if (config_key := FLAGS.get_config_value) is not None:
        config = FLAGS.config
        if config is not None:
            config = str(FLAGS.config)
        config = parse_config(config)
        print(getattr(config, config_key))
        return

    HERE = Path(__file__).parent

    if FLAGS.docker:
        exit("The command is not supported yet.")
        # subprocess.call('./start_docker.sh')
    else:
        start = HERE / "start.sh"
        open = "true" if FLAGS.open else "false"
        n_workers = str(FLAGS.n_workers)

        if FLAGS.config is not None:
            subprocess.call([start, open, n_workers, str(FLAGS.config)])
        else:
            subprocess.call([start, open, n_workers])


def main() -> None:
    app.run(execute)
