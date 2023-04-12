from typing import Any, Optional


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    return int(value)
