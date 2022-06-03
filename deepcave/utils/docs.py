from __future__ import annotations

import re
from pathlib import Path


def rst_to_md(filename: str | Path) -> str:
    if isinstance(filename, Path):
        filename = str(filename)

    with open(filename, "r") as file:
        data = file.read()

    # Remove reference
    result = re.finditer(r":ref:`(.*?)<(.*?)>`", data)
    for match in result:
        a = match.group(0)
        b = f"``{match.group(1)}``"
        data = data.replace(a, b)

    # Remove terms
    result = re.finditer(r":term:`(.*?) <(.*?)>`", data)
    for match in result:
        a = match.group(0)
        b = f"``{match.group(1)}``"
        data = data.replace(a, b)

    # Changing links
    result = re.finditer(r"`(.*?) <(.*?)>`_", data)
    for match in result:
        a = match.group(0)
        b = f"[{match.group(1)}]({match.group(1)})"
        data = data.replace(a, b)

    # Remove images
    result = re.finditer(r".. image::(.*?)\n", data)
    for match in result:
        a = match.group(0)
        data = data.replace(a, "")

    # Remove notes/warnings (not the best implementation but sufficient for now)
    data = data.replace(".. note::", "#### Note\n-----")
    data = data.replace(".. note ::", "#### Note\n-----")
    data = data.replace(".. warning::", "#### Warning\n-----")
    data = data.replace(".. warning ::", "#### Warning\n-----")
    data = data.replace(".. code::", "#### Code\n-----")
    data = data.replace(".. code ::", "#### Code\n-----")

    # Remove last \n
    if data.endswith("\n"):
        data = data[:-2]

    return data
