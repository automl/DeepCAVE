import os
from abc import ABC, abstractmethod 
from deep_cave.util.importing import auto_import_iter


class Layout(ABC):

    '''
    instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Layout, cls).__new__(cls)

            cls.instance.variables_defined = False
            cls.instance.define_variables()
            cls.instance.callbacks_registered = False
            cls.instance.register_callbacks()

        return cls.instance.get_layout()
    '''

    def __init__(self):
        self.register_callbacks()

    def register_callbacks(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
