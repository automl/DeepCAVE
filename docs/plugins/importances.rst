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