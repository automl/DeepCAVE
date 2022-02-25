from typing import Dict


def update_dict(a: Dict[str, Dict], b: Dict[str, Dict]):
    """
    Updates a from b inplace.
    """

    for k1, v1 in b.items():
        if k1 not in a:
            a[k1] = {}

        for k2, v2 in v1.items():
            a[k1][k2] = v2
