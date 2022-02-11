from typing import Any, List

import subprocess

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("start", False, "Starts DeepCAVE.")
flags.DEFINE_boolean("docker", False, "Uses docker images to start DeepCAVE.")
flags.DEFINE_string("get_config", None, "Returns the value of a given config key.")


def execute(args: List[Any]) -> None:
    if (config_key := FLAGS.get_config) is not None:
        from deepcave.config import config

        print(getattr(config, config_key))
        return

    if FLAGS.start:
        if FLAGS.docker:
            # TODO: Implement docker execution
            exit("The command is not supported yet.")

            # subprocess.call('./start_docker.sh')
        else:
            subprocess.call("./start.sh")


def main() -> None:
    app.run(execute)
