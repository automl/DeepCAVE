import numpy as np


def categorical_map_fn(value, mapping, reverse=False, nan=np.nan):
    # Take care about nan cases
    if not reverse and (value is None or (not isinstance(value, str) and np.isnan(value))):
        return -1
    elif reverse and value == -1:
        return nan

    if not reverse:
        return mapping[value]
    else:
        mapping_reversed = {v:k for k,v in mapping.items()}
        return mapping_reversed[value]


def numerical_map_fn(value, mn, mx, log=False, reverse=False, nan=np.nan):
    # Take care about nan cases
    if not reverse and (value is None or np.isnan(value)):
        return -1.
    elif reverse and (value == -1. or value == -1):
        return np.nan

    if not reverse:
        return (value-mn)/(mx-mn)
    else:
        return value*(mx-mn)+mn


if __name__ == "__main__":
    #v = map_fn(5, 0, 10, reverse=False, categorical_mapping=None)
    #assert v == 0.5

    v = categorical_map_fn(2, mapping={"tanh": 1, "sin": 2}, reverse=True)
    print(v)
    #assert v == 0

    #v = map_fn(0.5, 0, 2, reverse=True, categorical_mapping=None)
    #assert v == 1

    #v = map_fn(0.5, 0, 2, reverse=True, categorical_mapping={"tanh": 0, "sin": 1})
    #assert v == "sin"

    #mapping = {}
    #hp_names = ["a", "b", "c"]
    #cat_mapping = [{}, {"test1": 0, "test2": 1, "test3": 2}, {}]
    #mns = [0, 0, 100.]
    #mxs = [10, 2, 200.]

    #for hp_name, mn, mx, cat in zip(hp_names, mns, mxs, cat_mapping):
    #    mapping[hp_name] = lambda v, reverse=False, mn=mn, mx=mx, cat=cat: map_fn(v, mn, mx, reverse, cat)

    #pairs = [(5, 0.5), ("test1", 0.), (150., 0.5)]

    #for id, (input, output) in enumerate(pairs):
    #    assert mapping[hp_names[id]](input, False) == output
    #    assert mapping[hp_names[id]](output, True) == input
