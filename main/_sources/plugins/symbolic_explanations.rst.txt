Symbolic Explanations
====================

Symbolic Explanations allow to obtain explicit formulas quantifying the relation between
hyperparameter values and model performance by applying symbolic regression to meta-data collected
during hyperparameter optimization.

The plugin is capable of answering similar questions as the Partial Dependencies plugin, i.e.:

* How does the objective change with respect to one or two hyperparameters? For example, does the
  accuracy increase if the learning rate decreases?
* What explicit formula describes the relationship between one or two hyperparameters and the
  model's objective function?

While the Partial Dependencies plugin provides a plot describing the effects of hyperparameters on
the model performance, the Symbolic Explanations plugin additionally allows to obtain an explicit
formula capturing these effects.

To learn more about Symbolic Explanations, please see the paper
`Symbolic Explanations for Hyperparameter Optimization
<https://openreview.net/forum?id=JQwAc91sg_x>`_.

.. image:: ../images/plugins/symbolic_explanations.png
.. image:: ../images/plugins/symbolic_explanations2.png


Options
-------
* **Objective:** Select the objective function you wish to analyze. In the one hyperparameter case, the objective is plotted on the y-axis, in the two hyperparameter case, it is represented by the color.

* **Budget**: Select the multi-fidelity budget to be used. The plugin will only consider trials evaluated
  on the selected budget. The *Combined* budget option displays all configurations but shows scores only
  from the highest budget if a configuration was evaluated with multiple budgets.

* **Hyperparameter #1**: Select the first hyperparameter. It is plotted along the x-axis.

* **Hyperparameter #2**: Optionally select the second hyperparameter.

Furthermore, you can adjust the following parameters of the symbolic regression:

* **Parsimony coefficient:** Penalizes the complexity of the resulting formulas. The higher the value, the higher the penalty on the complexity will be, resulting in simpler formulas.

* **Generations:** The number of generations to evolve the symbolic regression.

* **Population Size:** The number of formulas competing in each generation.

* **Random seed:** The random seed to be used in the symbolic regression.

* **Metric:** The metric to evaluate the fitness of the formulas.
