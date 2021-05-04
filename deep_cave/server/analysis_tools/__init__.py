import collections
import os

from deep_cave.server.config import external_plugins
from deep_cave.server.plugins.plugin import Plugin
from deep_cave.util.importing import auto_import_iter
from deep_cave.util.logs import get_logger

logger = get_logger(__name__)


plugins = {}

paths = [os.path.join(os.path.dirname(__file__), '*')]
if external_plugins:
    if isinstance(external_plugins, list):
        paths.extend(external_plugins)
    if isinstance(external_plugins, str):
        paths.append(external_plugins)


for name, obj in auto_import_iter(paths):
    if not issubclass(obj, Plugin):
        continue
    # Plugin itself is a subclass, filter it out
    if obj == Plugin:
        continue
    # class should have the correct signature, so this redundant
    if hasattr(obj, 'wip'):
        try:
            plugin = obj()
        except TypeError as e:
            logger.exception(e)
        # if the plugin is WIP, then don't add it. Later with production and development env, maybe change this.
        if not plugin.wip:
            # logger.info('Loaded plugin ' + plugin.name)
            plugins[plugin.name] = plugin

# order the plugins, so that they are always displayed in the same order
plugins = collections.OrderedDict(sorted(plugins.items()))
# the only object available from this module is the plugins dictionary
__all__ = ['plugins']
