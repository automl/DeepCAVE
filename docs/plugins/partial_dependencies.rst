Partial Dependencies
====================

The quality of a machine learning algorithm is highly dependent on the choice of hyperparameters.
Partial dependence plots (PDP) aim to visualize how changing one or two hyperparameters affects the objective function.
The PDP is calculated using a random forest surrogate model trained on the run data, i.e. the evaluated hyperparameter configurations and corresponding objective values.

* In the **one hyperparameter case**, random samples or the selected hyperparameters are combined with each combination of the other hyperparameters present in the dataset, resulting in a multitude of individual conditional expectation (ICE) curves.
  The resulting ICE curves are than averaged to a PDP.
  The hyperparameter is plotted along the x axis, and the respective objective function value along the y-axis.

* The **two hyperparameter case** works as the one hyperparameter case, but the then two dimensional ICE-curves are disregarded in the plot.
  The resulting plot has one target hyperparameter plotted along the x-axis and the other hyperparameter plotted along the y-axis.
  The color in the heatmap plot indicates the respective objective function value.


This plugin is capable of answering the following questions:

* How does the objective change with respect to one or two hyperparameters? For example, does the
  accuracy increase if the learning rate decreases?
* Do multiple trials show similar behavior?

Options
-------
* **Objective**: Here you can select the objective you wish to analyze. In the one hyperparameter case the objective is plotted on the y-axis, in the two hyperparameter case it is represented by the color.

* **Budget**: Here you can select the multi-fidelity budget to be used. The plugin will only consider trials evaluated on the selected budget.

* **Hyperparameter 1**: Here you can select the first hyperparameter. It is plotted along the x-axis.

* **Hyperparameter 2**: Here you can optionally select the second hyperparameter.

* **Show Confidence**: Selection for whether or not to show confidence estimates (based on the surrogate model's uncertainty estimate).

* **Show ICE Curves**: Selection for whether or not to to show the ICE curves used during computation (only for the one hyperparameter case).


.. image:: ../images/plugins/partial_dependencies.png
