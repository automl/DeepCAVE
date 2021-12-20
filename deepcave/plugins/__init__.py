import os

from deepcave.plugins.dynamic_plugin import DynamicPlugin
from deepcave.plugins.plugin import Plugin
from deepcave.plugins.static_plugin import StaticPlugin
from deepcave.utils.importing import auto_import_iter


def read_plugins():
    plugins = {}

    paths = [os.path.join(os.path.abspath(os.curdir),
                          'deepcave', 'plugins', '*')]

    for _, obj in auto_import_iter("deepcave.plugins", paths):

        if issubclass(obj, Plugin):
            # Plugin itself is a subclass, filter it out
            if obj.__module__ == DynamicPlugin.__module__ or \
               obj.__module__ == StaticPlugin.__module__ or \
               obj.__module__ == Plugin.__module__:
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


def get_plugin_layouts():
    layouts = {}

    for plugin in plugins:
        layouts[plugin.id()] = plugin()

    return layouts


def get_plugin_names():
    names = {}

    for plugin in plugins:
        names[plugin.id()] = plugin.name()

    return names


def get_plugin_categories():
    categories = {}

    for plugin in plugins:
        categories[plugin.id()] = plugin.category()

    return categories


plugin_layouts = get_plugin_layouts()
plugin_names = get_plugin_names()
plugin_categories = get_plugin_categories()

__all__ = [plugin_layouts, plugin_names, plugin_categories]
