from typing import Any, List

import subprocess

from absl import app, flags

from deepcave.config import parse_config, configs

FLAGS = flags.FLAGS
flags.DEFINE_boolean("start", False, "Starts DeepCAVE.")
flags.DEFINE_boolean("docker", False, "Uses docker images to start DeepCAVE.")
flags.DEFINE_string("get_config", None, "Returns the value of a given config key.")
flags.DEFINE_enum("config", "default", configs.keys(), help="Use selected config")


def execute(args: List[Any]) -> None:
    if (config_key := FLAGS.get_config) is not None:
        config = parse_config(FLAGS.config)
        print(getattr(config, config_key))
        return

    if FLAGS.start:
        if FLAGS.docker:
            # TODO: Implement docker execution
            exit("The command is not supported yet.")

            # subprocess.call('./start_docker.sh')
        else:
            subprocess.call(["./start.sh", FLAGS.config])


def main() -> None:
    app.run(execute)
