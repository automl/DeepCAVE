from typing import Any

import datetime
import random
import string


def get_random_string(length: int) -> str:
    if length < 0:
        raise ValueError("Length has to be greater than 0")
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def short_string(value: Any, length: int = 30, *, mode: str = "prefix") -> str:
    value = str(value)
    if len(value) <= length:
        return value

    cut_length = length - 3  # For 3 dots (...)
    if mode == "prefix":
        return f"...{value[-cut_length:]}"
    elif mode == "suffix":
        return f"{value[:cut_length]}..."
    raise ValueError(f"Unknown mode '{mode}'")


def get_latest_change(st_mtime: int) -> str:
    t = datetime.datetime.fromtimestamp(st_mtime)
    s_diff = (datetime.datetime.now() - t).seconds
    d_diff = (datetime.datetime.now() - t).days

    if s_diff < 60:
        return "Some seconds ago"
    elif s_diff < 3600:
        return f"{int(s_diff / 60)} minutes ago"
    elif s_diff < 86400:
        return f"{int(s_diff / 60 / 60)} hours ago"
    elif d_diff < 7:
        return f"{d_diff} days ago"
    else:
        return t.strftime("%Y/%m/%d")
