# noqa
"""from typing import Any, Dict, List, Optional


import numpy as np
from ConfigSpace import ConfigurationSpace
from random_forest import RandomForest

import deepcave.utils.styled_plot as styled_plot
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.run import AbstractRun
from deepcave.utils import layout

test_configspace = ConfigurationSpace(
    name="test_space",
    space={
        "uniform_integer": (1, 10),
        "uniform_float": (1.0, 10.0),
        "categorical": ["a", "b", "c"],
        "constant": 1337,
    },
)
test_forest = RandomForest(test_configspace)
test_config = test_configspace.sample_configuration()

test_run = SMAC3v1Run("test_smac", test_configspace)
# print(test_config.config_id)
# print(type(test_run.get_trial_key(test_config.config_id, 10)))


# For testing the get leaf values, but it returns nothing?
# It returns a Tuple but of what exactly?
x = np.array(test_config.get_array())
print(test_forest.get_leaf_values(x))


test_dict: Dict[str, List[Any]]
test_dict = {"hp_1": [], "hp_2": [], "hp_3": [], "hp_4": []}
test_dict["hp_2"].append([1, 2, 3, 4])
test_dict["hp_2"].append(0.2)
# print(test_dict)

config1 = test_configspace.sample_configuration()
config2 = test_configspace.sample_configuration()
config3 = test_configspace.sample_configuration()
config_dict_ = {0: config1, 5: config2, 7: config3}
config_dict_1 = {0: config1, 5: config2, 7: config3}
dict_list = [config_dict_, config_dict_1]
# print(len(dict_list))
# print(layout.get_slider_marks(dict_list))
# print(dict(zip(list(range(len(config_dict_))), list(range(len(config_dict_))))))
# print(len(config_dict_))
# print(type(range(len(config_dict_))))
# print(layout.get_slider_marks(list(range(len(config_dict_)))))
# print(layout.get_slider_marks(steps=3))

# print(layout.get_slider_marks(config_dict_))
yaxis = "hallo"
layout_kwargs = {
    "margin": 1,
    "xaxis": {"title": "Budget", "domain": 5},
}
print(layout_kwargs)
layout_kwargs[yaxis] = {
    # "title": objective.name,
    "titlefont": {"color": 0},
    "tickfont": {"color": 0},
    "range": [3, 2],
}

print(layout_kwargs)
layout_kwargs[yaxis] |= {
    "anchor": "free",
    "overlaying": "y",
    "side": "left",
    "position": 0,
}

print(layout_kwargs)

# Consider two lists
languages1 = [
    "Python",
    "PHP",
    "Java",
]
languages2 = ["C", "C++", "C#"]
print("Language List 1: ", languages1)
print("Language List 2: ", languages2)

# Append list into another list
languages1.append([language for language in languages2])
print("Result: ", languages1)

tuple_test = ("arg2", "arg3", "arg4")
print(tuple_test)
print(list(tuple_test))

import seaborn as sns

pal = sns.color_palette()
hex_codes = pal.as_hex()
print(type(hex_codes[5 % len(hex_codes)]))

styled_plot = styled_plot.StyledPlot()
plt = styled_plot.plt

print(type(plt.render()))

list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
list_comp = [1.5 for i in list]


def test(hallo: Optional[int] = 2):
    print(type(hallo))

test(5)
test()"""
