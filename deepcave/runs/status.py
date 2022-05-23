from enum import IntEnum


class Status(IntEnum):
    SUCCESS = 1
    TIMEOUT = 2
    MEMORYOUT = 3
    CRASHED = 4
    ABORTED = 5
    RUNNING = 6
    NOT_EVALUATED = 7

    def to_text(self) -> str:
        return self.name.lower().replace("_", " ")
