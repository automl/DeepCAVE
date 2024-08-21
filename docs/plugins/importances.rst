Importances
===========

The Importances plugin provides insights into how individual hyperparameters influence the
objective performance of your models.
It generates a bar plot displaying the importance scores for each hyperparameter.

This plugin is capable of answering following questions:

* Which individual hyperparameters have the greatest impact on the objective performance?
* How does the importance of each hyperparameter vary across different budgets?
* How does the local parameter importance compare to the global importance as determined by fANOVA?

.. image:: ../images/plugins/importances.png

Importance Methods
------------------

**Local Parameter Importance** (LPI) focuses on understanding the impact of individual
hyperparameters in the context of a specific configuration (here, the incumbent). Leveraging a
random forest surrogate model, it calculates the variance in performance caused by changes in each
hyperparameter, and expresses this as a fraction of the total variance.
This method is especially useful for pinpointing which parameters are crucial in the context of the
incumbent their local effects. For more details on LPI, please see the paper
`CAVE: Configuration Assessment, Visualization and Evaluation
<https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-LION12-CAVE.pdf>`_.

**fANOVA Importance** Functional ANOVA (fANOVA) evaluates hyperparameter importance by decomposing
the performance variance into contributions from individual hyperparameters and their interactions.
It uses a surrogate model, such as a random forest, to predict performance across different
hyperparameter configurations, and then analyzes how much each hyperparameter and its interactions
contribute to the overall variance in performance. This method provides a comprehensive view of
hyperparameter importance across the entire configuration space, thus helping in understanding the
global impact of hyperparameters. For more details on fANOVA, please refer to the paper
`An Efficient Approach for Assessing Hyperparameter Importance
<https://proceedings.mlr.press/v32/hutter14.html>`_.


Options
-------

* **Method:** Whether to calculate the local parameter importance or the fANOVA importance.

* **Objective:** Select the objective function you wish to analyze.

* **Trees:** Specify the number of trees for the random forest surrogate model used in calculating importance scores.

To refine your analysis, you can apply various filters after calculation:

* **Hyperparameters:** Display only the importance scores for specific hyperparameters of interest.

* **Limit Hyperparameters**: Show only the top N most important hyperparameters.

* **Budgets**: Filter the results to view importance scores specific to certain multi-fidelity budgets, allowing you to analyze how hyperparameter importance varies with budget changes.

.. warning::
    The fANOVA implementation relies on pyrfr, which cannot be applied to constant hyperparameters.
    As a workaround, we remove constant hyperparameters before calculation.
    This will break if the configuration space contains any conditions or forbiddens including
    constant hyperparameters.