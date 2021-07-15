"""

.. code:: python
    import deep_cave

    config = {'lr': 1e8}
    fidelity = 2

    with deep_cave.start_trial(config, fidelity) as trial:
        trial.log_metric(.5)

"""

# mlflow style access to API via __init__.py

'''
from .logger import (
    start_trial,
    end_trial,
    start_study,
    end_study,
    log_surrogate
)

from .store.state import (
    set_study,
    get_study,
    set_tracking_uri,
    get_tracking_uri,
    set_registry_uri,
    get_registry_uri
)


__all__ = [
    'start_trial',
    'end_trial',
    'start_study',
    'end_study',
    'set_study',
    'get_study',
    'set_tracking_uri',
    'get_tracking_uri',
    'set_registry_uri',
    'get_registry_uri',
    'log_surrogate'
]
'''

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
