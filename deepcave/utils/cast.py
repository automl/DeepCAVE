from typing import Any


def optional_int(value: Any):
    if value is None:
        return None

    return int(value)
