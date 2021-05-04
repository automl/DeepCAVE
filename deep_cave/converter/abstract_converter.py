from abc import abstractmethod

from ..storage_backend import AbstractStorage


class AbstractConverter(AbstractStorage):
    """
    AbstractConverter is a subclass of AbstractStorage. Thereby, converters can be used where storage is required.
    Adds another the static method name. Name is used to select the converter from the configured config variable
    CONVERTER.

    """
    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Name of the converter child class.

        Returns
        -------
            The name as string.
        """
        pass

