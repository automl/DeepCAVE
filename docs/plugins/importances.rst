Importances
===========

Hyperparameter importance tells you how large the influence of single hyperparameters are.
This plugin is capable of answering following questions:

* Which hyperparameters are the most important?
* How much differs the importance between the budgets?


.. warning::
    This page is under construction.


.. image:: ../images/plugins/importances.png

.. warning::
    As the fANOVA implementation relies on pyrfr and pyrfr cannot be applied to constant hyperparameters,
    as a workaround we remove constant hyperparameters before calculation.
    This will break if there are conditions or forbiddens including constant hyperparameters.

.. note::
    Ablation Importance: If a plot is not showing for a specific budget, you might have to expand your configuration space in order to 
    improve the performance of the surrogate model. You can also increase the number of trees.
    Please note the order the hyperparameters are in. The performance increase 
    or decrease of each hypterparameter depends on the hyperparameter(s) prior to it. If the performance decreases, ablation importance
    chooses the hyperparameter with the smallest decrease. If two hyperparameters increase or decrease the performance equally, 
    they are chosen randomly.
    The results of fANOVA and Ablation Importance can be complementary. This is because fANOVA is a global method and Ablation 
    Importance might only see the interactions partially.
    Biedenkapp, A., Lindauer, M., Eggensperger, K., Hutter, F., Fawcett, C., & Hoos, H. (2017). 
    Efficient Parameter Importance Analysis via Ablation with Surrogates. 
    Proceedings of the AAAI Conference on Artificial Intelligence, 31(1). https://doi.org/10.1609/aaai.v31i1.10657

.. warning::
    Ablation Importance: It is not recommended to limit the number of hyperparameters for this method. This is because all 
    hyperparameters depend on each other and ignoring some can cause a loss of information.