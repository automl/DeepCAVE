from pathlib import Path

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.plugins.plugin import Plugin
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.utils.importing import auto_import_iter


def read_plugins():
    plugins = {}

    paths = [Path(__file__).parent]

    for _, obj in auto_import_iter("deepcave.plugins", paths):

        if issubclass(obj, Plugin):
            # Plugin itself is a subclass, filter it out
            if any(obj.__module__ in c.__module__ for c in [DynamicPlugin, StaticPlugin, Plugin]):
                continue

            pos = obj.position()
            if pos not in plugins:
                plugins[pos] = []

            plugins[pos].append(obj)

    plugins_ordered = []
    for pos in sorted(plugins):
        plugins_ordered += plugins[pos]

    return plugins_ordered


plugins = read_plugins()

plugin_layouts: dict[str, Plugin] = {plugin.id(): plugin() for plugin in plugins}
plugin_names: dict[str, str] = {plugin.id(): plugin.name() for plugin in plugins}
plugin_categories: dict[str, str] = {plugin.id(): plugin.category() for plugin in plugins}

__all__ = [plugin_layouts, plugin_names, plugin_categories]
