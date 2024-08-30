Pareto Front
============

Pareto efficiency, or Pareto optimality, occurs when no single configuration can be deemed superior
because multiple important objectives must be balanced. For instance, you might seek a configuration
that offers both high performance and low computational cost. The Pareto front helps by highlighting
the best configurations relative to two given objectives.

.. note::
    You can enable or disable specific runs if you click on the name right to the plot.
    Clicking on a configuration will redirect you to the configuration plugin, where you can
    view detailed information.

This plugin can help answer the following questions:

* Which configuration should I choose if I slightly favor objective A over objective B?
* How does optimizer A compare to optimizer B? Is optimizer A more effective than optimizer B?

.. image:: ../images/plugins/pareto_front.png

Options
-------

* **Objective 1 / 2:** Select the objective functions you wish to analyze.

* **Budget**: Select the multi-fidelity budget to be used. The plugin will only consider trials evaluated
  on the selected budget. The *Combined* budget option displays all configurations but shows scores only
  from the highest budget if a configuration was evaluated with multiple budgets.

To refine your analysis, you can apply various filters after calculation:

* **Show all configurations:** Additionally to the pareto front, also the other configurations are
  displayed. This makes it easier to see the performance differences.

* **Show error bars**: Show error bars in the case of non-deterministic runs with multiple seeds
  evaluated per configuration.

* **Show Runs / Show Groups**: You can choose to display either individual runs, groups of runs, or both.
