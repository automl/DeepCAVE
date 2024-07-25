Ablation Paths
===========

.. warning::
    This page is under construction.


.. image:: ../images/plugins/ablation_importances.png


.. note::
    If a plot is not showing for a specific budget, you might have to expand your configuration space in order to
    improve the performance of the surrogate model. You can also increase the number of trees.
    Please note the order the hyperparameters are in. The performance increase
    or decrease of each hypterparameter depends on the hyperparameter(s) prior to it. If the performance decreases, ablation importance
    chooses the hyperparameter with the smallest decrease. If two hyperparameters increase or decrease the performance equally,
    they are chosen randomly.

    Biedenkapp, A., Lindauer, M., Eggensperger, K., Hutter, F., Fawcett, C., & Hoos, H. (2017).
    Efficient Parameter Importance Analysis via Ablation with Surrogates.
    Proceedings of the AAAI Conference on Artificial Intelligence, 31(1). https://doi.org/10.1609/aaai.v31i1.10657

.. warning::
    It is not recommended to limit the number of hyperparameters for this method. This is because all
    hyperparameters depend on each other and ignoring some can cause a loss of information.

.. warning::
    It is not recommended to display multiple budget at once. The ranking of the hyperparameters is important and will get
    mixed up, because for different budgets the ranking can change.
