from typing import Dict


def update_dict(a: Dict[str, Dict], b: Dict[str, Dict]) -> None:
    """
    Updates dictionary a with dictionary b inplace.

    Parameters
    ----------
    a : Dict[str, Dict]
        Dictionary to be updated.
    b : Dict[str, Dict]
        Dictionary to be added.
    """
    for k1, v1 in b.items():
        if k1 not in a:
            a[k1] = {}

        for k2, v2 in v1.items():
            a[k1][k2] = v2
