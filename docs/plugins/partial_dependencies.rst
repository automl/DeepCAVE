Partial Dependencies
====================

The quality of a machine learning algorithm is highly dependent on the choice of hyperparameters. PdP plots aim to visualize how chaning one or two hyperparameters affects the objective function. To reduce comptutational complexity the plugin utilizes a random forest surrogate model.

* In the one hyperparameter case, random samples or the selected hyperparameters are combined with each combination of the othre hyperparameters present in the dataset resulting in a multitude of ice curves. The resulting ice curves are than averaged to a PD plot. The hyperparameter is plotted along the x axis, and the respective objective function value along the y-axis

* The two hyperparameter case works as the one hyperparameter case, but the then two dimensional ice-curves are disregarded in the plot. The resulting plot has one target hyperparameter plotted along the x-axis and the other hyperparameter plotted along the y-axis. The color in the plot indicates the respective objective function value.


This plugin is capable of answering following questions:

* How does the objective change with respect to one or two hyperparameters? For example, does the
  accuracy increase if the learning rate decreases?
* Do multiple trials show similar behavior?
Options
-------
* **Objective**: Here you can select the objective you wish to analyze for. In the one hyperparameter case the objective is plotted on the y-axis, in the two hyperparameter case it is represented by the color.

* **Budget**: Here you can select the budget used to compute ICE. TODO 

* **Hyperparameter 1**: Here you can select the first hyperparameter. It is plotted along the x-axis.

* **Hyperparameter 2**: Here you can optionally select the second hyperparameter.

* **Show Confidence**: Selection for whether or not to show certainty estimates (based on the difference of ice curves).

* **Show ICE Curves**: Selection for whether or not to to show the ICE curves used during computation


.. image:: ../images/plugins/partial_dependencies.png
