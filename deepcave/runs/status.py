from enum import IntEnum


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    MEMORYOUT = 3
    CRASHED = 4
    ABORTED = 5
    NOT_EVALUATED = 6

    def to_text(self) -> str:
        """
        Convert name to simpler, lower case text format.

        Returns
        -------
        str
            The converted name in lower case with spaces added
        """
        return self.name.lower().replace("_", " ")
