import os
from abc import ABC, abstractmethod 
from deep_cave.util.importing import auto_import_iter


class Layout(ABC):

    instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Layout, cls).__new__(cls)

            cls.instance.variables_defined = False
            cls.instance.define_variables()
            cls.instance.callbacks_registered = False
            cls.instance.register_callbacks()

        return cls.instance.get_layout()

    def define_variables(self):
        if not self.variables_defined:
            self._define_variables()
            self.variables_defined = True
    
    def register_callbacks(self):
        if not self.callbacks_registered:
            self._register_callbacks()
            self.callbacks_registered = True

    def get_layout(self):
        return self._get_layout()

    def _define_variables(self):
        pass

    def _register_callbacks(self):
        pass

    def _get_layout(self):
        pass
