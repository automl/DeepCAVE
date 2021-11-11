import os

from absl import app
from absl import flags

from src.utils.cache import Cache
from src.utils.run_caches import RunCaches
from src.server import get_app
from src.queue import Queue
from src.config import CONFIG, META
from src.runs.recorder import Recorder

FLAGS = flags.FLAGS
flags.DEFINE_boolean('interface', False, 'Produces debugging output.')

version = "0.0.1"


if __name__ == "__main__":
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
