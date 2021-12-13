from abc import abstractmethod


class Layout:
    def __init__(self):
        self.register_callbacks()

    def register_callbacks(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
