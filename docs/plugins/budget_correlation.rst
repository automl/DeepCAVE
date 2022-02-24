Budget Correlation
==================

When multiple :term:`budgets <budget>` are used, it is not clear how well lower budgets perform 
in contrast to higher budgets. The budget correlation is one method to find out if a low budget
is sufficient enough to replace a high budget.

Especially in Neural Networks, you do not need to train longer if convergence was already reached.
If the convergence already was reached in budget 1 but the training still continues till budget 2,
the correlation between budget 1 and budget 2 will be high.


.. image:: ../images/plugins/budget_correlation.png
