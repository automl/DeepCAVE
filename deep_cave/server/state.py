from functools import lru_cache


from .. import store

from .config import studies_location, converter
from ..util.logs import get_logger

logger = get_logger(__file__)

# intended for holding the global state of the sever
# caution with this global state. There can be multiple users on the side simultaneously.
# The action of one user shouldn't alter the state of the other

# generate the new state of studies each time it is requested. To always be up to date.
def get_studies():
    try:
        return store.get_studies(studies_location, converter=converter)
    except Exception as e:
        logger.error(str(e) + ': couldnt set studies')
        raise e


# manage the cache from here. The cache object can be cleared, to force reloading of the data
cache = lru_cache(maxsize=10, typed=False)
